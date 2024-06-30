import argparse
import math
import os
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor, QUANT_DESC_8BIT_PER_TENSOR
from pytorch_quantization.nn.modules import _utils
from pytorch_quantization import quant_modules
quant_modules.initialize()

import onnx
import onnx_graphsurgeon as gs

class QuantConv(torch.nn.Module, _utils.QuantMixin):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                weight,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode='zeros',
                **kwargs):            
        super().__init__()
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8))
        self._weight_quantizer = quant_nn.TensorQuantizer(QuantDescriptor(num_bits=8, axis=(0)))
        # self =
        self.in_channels = in_channels
        self.out_channels =out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride=_pair(stride)
        self.padding=_pair(padding)
        self.dilation=_pair(dilation)
        self.groups=groups
        self.bias=bias
        self.padding_mode=padding_mode
        self.weight = weight
    def forward(self, x):
        quant_input = self._input_quantizer(x)
        quant_weight = self._weight_quantizer(self.weight)
        
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            output = F.conv2d(F.pad(quant_input, expanded_padding, mode='circular'),
                            quant_weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else:
            # output = nn.Conv2d
            output = F.conv2d(quant_input, quant_weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation,
                            groups=self.groups)

        return output

class QuantAdd(torch.nn.Module, _utils.QuantMixin):
    default_quant_desc_input = QUANT_DESC_8BIT_PER_TENSOR
    def __init__(self, quantization):
        super().__init__()
        if quantization:
            self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
            self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.quantization = quantization

    def forward(self, x, y):
        if self.quantization:
            # print(f"QAdd {self._input0_quantizer}  {self._input1_quantizer}")
            return self._input0_quantizer(x) + self._input1_quantizer(y)
        return x + y
    
class QuantC2fChunk(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.c = c
    def forward(self, x, chunks, dims):
        return torch.split(self._input0_quantizer(x), (self.c, self.c), dims)
    
class QuantConcat(torch.nn.Module): 
    def __init__(self, dim):
        super().__init__()
        self._input0_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self._input1_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        self.dim = dim

    def forward(self, x, dim):
        x_0 = self._input0_quantizer(x[0])
        x_1 = self._input1_quantizer(x[1])
        return torch.cat((x_0, x_1), self.dim) 

class QuantUpsample(torch.nn.Module): 
    def __init__(self, size, scale_factor, mode):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self._input_quantizer = quant_nn.TensorQuantizer(QuantDescriptor())
        
    def forward(self, x):
        return F.interpolate(self._input_quantizer(x), self.size, self.scale_factor, self.mode)


def export_onnx(best, model, imgsz):    
    onnx_f = best.with_suffix('.onnx')
    
    model.to("cuda")
    model.eval()                    
    model.model[-1].export = True
    model.model[-1].format = "onnx"
    # model.fuse() # conv & bn fuse

    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    quant_modules.initialize()

    dummy_input = torch.randn(1, 3, imgsz, imgsz).to('cpu')
    input_names = ["input" ]
    output_names = [ "pred" ]

    dynamic_axes = {'input': {0: 'batch'}}  # shape(1,3,640,640)
    dynamic_axes['pred'] = {0: 'batch'}  # shape(1,25200,85)

    torch.onnx.export(
        model.to('cpu'), 
        dummy_input.to('cpu'), 
        onnx_f, 
        do_constant_folding=True, 
        verbose=False, 
        opset_version=13,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
        )
    
    model.to("cuda")
    model.train()
    model.model[-1].export = False

    return onnx_f
    
def modify_onnx(onnx_f):
    def get_many_outputs_act_nodes(nodes, act_node_name = "Silu"):
        if act_node_name == "Silu":
            act_nodes = [node for node in nodes if node.op == "Mul" and node.i(0).op == "BatchNormalization" and node.i(1).op == "Sigmoid"]
            many_outputs_act_nodes = []
            
        elif act_node_name == "Relu":
            act_nodes = [node for node in nodes if node.op == "Relu" and node.i(0).op == "BatchNormalization"]
            many_outputs_act_nodes = []

        for node in act_nodes: # convolution mul node for silu activation.
            try:
                for i in range(99):
                    node.o(i)
            except:
                if i > 1:
                    nodename_outnum = {"node": node, "out_num": i}
                    many_outputs_act_nodes.append(nodename_outnum)
        
        return many_outputs_act_nodes

    def get_many_outputs_add_nodes(nodes):
        add_nodes = [node for node in nodes if node.op == "Add"]
        many_outputs_add_nodes = []

        for node in add_nodes: # convolution mul node for silu activation.
            # mul 노드의 output tensor 개수 파악.
            try:
                for i in range(99):
                    node.o(i)
            except:
                if i > 1 and node.o().op == "QuantizeLinear":
                    add_nodename_outnum = {"node": node, "out_num": i}
                    many_outputs_add_nodes.append(add_nodename_outnum)
        return many_outputs_add_nodes

    def merge_qdq_act_nodes(many_outputs_act_nodes):
        if len(many_outputs_act_nodes) != 0:
            for node_dict in many_outputs_act_nodes:
                if node_dict["out_num"] == 2:
                    if node_dict["node"].o(0).op == "QuantizeLinear" and node_dict["node"].o(1).op == "QuantizeLinear":
                        if node_dict["node"].o(1).o(0).o(0).op == "Concat":
                            concat_dq_out_name = node_dict["node"].o(1).o(0).outputs[0].name
                            for i, concat_input in enumerate(node_dict["node"].o(1).o(0).o(0).inputs):
                                if concat_input.name == concat_dq_out_name:
                                    # print(node_dict["node"].o(1).o(0).outputs[0].name, i)
                                    node_dict["node"].o(1).o(0).o(0).inputs[i] = node_dict["node"].o(0).o(0).outputs[0] # concat 4개
                        else:
                            node_dict["node"].o(1).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0] # 그 외
                            
                    elif node_dict["node"].o(0).op == "QuantizeLinear" and node_dict["node"].o(1).op == "Concat":
                        concat_dq_out_name = node_dict["node"].outputs[0].outputs[0].inputs[0].name
                        for i, concat_input in enumerate(node_dict["node"].outputs[0].outputs[1].inputs):
                            if concat_input.name == concat_dq_out_name:
                                node_dict["node"].outputs[0].outputs[1].inputs[i] = node_dict["node"].outputs[0].outputs[0].o().outputs[0] # concat 4개
                
                elif node_dict["out_num"] == 3:
                    node_dict["node"].o(2).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
                    node_dict["node"].o(1).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]

                elif node_dict["out_num"] == 4: # shape node not merged
                    node_dict["node"].o(3).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]
                    node_dict["node"].o(2).o(0).o(0).inputs[0] = node_dict["node"].o(0).o(0).outputs[0]

    def merge_qdq_add_nodes(many_outputs_add_nodes):
        for node_dict in many_outputs_add_nodes:
            if node_dict["node"].outputs[0].outputs[0].op == "QuantizeLinear" and node_dict["node"].outputs[0].outputs[1].op == "Concat":
                concat_dq_out_name = node_dict["node"].outputs[0].outputs[0].inputs[0].name
                for i, concat_input in enumerate(node_dict["node"].outputs[0].outputs[1].inputs):
                    if concat_input.name == concat_dq_out_name:
                        node_dict["node"].outputs[0].outputs[1].inputs[i] = node_dict["node"].outputs[0].outputs[0].o().outputs[0] # concat 4개
        
    def delete_dfl_conv_qdq(conv_nodes):
        conv_nodes[-1].inputs[0] = conv_nodes[-1].i().i().inputs[0] # dfl block input
        conv_nodes[-1].inputs[1] = conv_nodes[-1].i(1).i().inputs[0] # dfl block weight

    model_onnx = onnx.load(onnx_f)
    graph = gs.import_onnx(model_onnx)
    nodes = graph.nodes
    tensors = graph.tensors()
    
    many_outputs_silu_nodes = get_many_outputs_act_nodes(nodes, "Silu")
    many_outputs_relu_nodes = get_many_outputs_act_nodes(nodes, "Relu")
    many_outputs_add_nodes = get_many_outputs_add_nodes(nodes)

    merge_qdq_act_nodes(many_outputs_silu_nodes)
    merge_qdq_act_nodes(many_outputs_relu_nodes)
    merge_qdq_add_nodes(many_outputs_add_nodes)

    # ---- 첫번째, 마지막 Conv의 q/dq 노드 삭제 ---- #
    # ---- we dont need this now ---- #
    conv_nodes = [node for node in nodes if node.op == "Conv"]
    delete_dfl_conv_qdq(conv_nodes)
    
    graph.cleanup()
    onnx.checker.check_model(gs.export_onnx(graph))

    return gs.export_onnx(graph)

def cal_model(model, data_loader, device, num_batch=1024):
    num_batch = num_batch
    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(**kwargs)

                    module._amax = module._amax.to(device)
        
    def collect_stats(model, data_loader, device, num_batch=128):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        # Feed data to the network for collecting stats
        with torch.no_grad():
            for i, datas in tqdm(enumerate(data_loader), total=num_batch, desc="Collect stats for calibrating"):
                # imgs = datas[0].to(device, non_blocking=True).float() / 255.0
                imgs = datas['img'].to(device, non_blocking=True).float() / 255.0
                model(imgs)

                if i >= num_batch:
                    break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    collect_stats(model, data_loader, device, num_batch=num_batch)
    compute_amax(model, method="mse")
    
def add_qdq_model(model):
    def conv2d_quant_forward(self, x):
        if hasattr(self, "conv2dop"):
            return self.conv2dop(x)
        return F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

    def bottleneck_quant_forward(self, x):
        if hasattr(self, "addop"):
            return self.addop(x, self.cv2(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

    def concat_quant_forward(self, x):
        if hasattr(self, "concatop"):
            return self.concatop(x, self.d)
        return torch.cat(x, self.d)

    def upsample_quant_forward(self, x):
        if hasattr(self, "upsampleop"):
            return self.upsampleop(x)
        return F.interpolate(x, self.size, self.scale_factor, self.mode)

    def c2f_qaunt_forward(self, x):
        if hasattr(self, "c2fchunkop"):
            y = list(self.c2fchunkop(self.cv1(x), 2, 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))
            
        else:
            y = list(self.cv1(x).split((self.c, self.c), 1))
            y.extend(m(y[-1]) for m in self.m)
            return self.cv2(torch.cat(y, 1))
        
    def conv_quant_forward(self, x):
        if hasattr(self, "convop"):
            return self.act(self.bn(self.convop(x)))
        else:
            return self.act(self.bn(self.conv(x)))

    for name, module in model.named_modules():
        if module.__class__.__name__ == "Conv2d":
            if not hasattr(module, "conv2dop"):
                print(f"Add QuantConv to {name}")
                # print("module.bias", module.bias)
                module.conv2dop = QuantConv(in_channels=module.in_channels, \
                                                out_channels=module.out_channels, \
                                                kernel_size=module.kernel_size, \
                                                stride=module.stride, \
                                                padding=module.padding, \
                                                dilation=module.dilation,\
                                                groups=module.groups, \
                                                bias=module.bias, \
                                                padding_mode=module.padding_mode,
                                                weight=module.weight)
            module.__class__.forward = conv2d_quant_forward
        
        # if module.__class__.__name__ == "Conv":
        #     if not hasattr(module, "convop"):
        #         print(f"Add QuantConv to {name}")
        #         # print("module.bias", module.bias)
        #         module.convop = QuantConv(in_channels=module.conv.in_channels, \
        #                                         out_channels=module.conv.out_channels, \
        #                                         kernel_size=module.conv.kernel_size, \
        #                                         stride=module.conv.stride, \
        #                                         padding=module.conv.padding, \
        #                                         dilation=module.conv.dilation,\
        #                                         groups=module.conv.groups, \
        #                                         bias=module.conv.bias, \
        #                                         padding_mode=module.conv.padding_mode,
        #                                         weight=module.conv.weight)
        #     module.__class__.forward = conv_quant_forward
            
        if module.__class__.__name__ == "C2f":
            if not hasattr(module, "c2fchunkop"):
                print(f"Add C2fQuantChunk to {name}")
                module.c2fchunkop = QuantC2fChunk(module.c)
            module.__class__.forward = c2f_qaunt_forward

        if module.__class__.__name__ == "Bottleneck":
            if module.add:
                if not hasattr(module, "addop"):
                    print(f"Add QuantAdd to {name}")
                    module.addop = QuantAdd(module.add)
                module.__class__.forward = bottleneck_quant_forward
                
        if module.__class__.__name__ == "Concat":
            if not hasattr(module, "concatop"):
                print(f"Add QuantConcat to {name}")
                module.concatop = QuantConcat(module.d)
            module.__class__.forward = concat_quant_forward

        if module.__class__.__name__ == "Upsample":
            if not hasattr(module, "upsampleop"):
                print(f"Add QuantUpsample to {name}")
                module.upsampleop = QuantUpsample(module.size, module.scale_factor, module.mode)
            module.__class__.forward = upsample_quant_forward
    return model
    

if __name__=="__main__":
    model_onnx = onnx.load("/home/maro.jeon/autonomy-deep-learning-YOLOv8/runs/detect/train55/weights/best_qat.onnx")
    model_onnx = graphsurgeon_model(model_onnx)
    model_onnx = onnx.save(model_onnx, "/home/maro.jeon/autonomy-deep-learning-YOLOv8/runs/detect/train55/weights/best_qat_modified.onnx")

    print('aaaa')