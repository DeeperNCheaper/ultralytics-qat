from modelopt.torch.quantization import nn as quant_nn
from modelopt.torch.quantization import calib
from modelopt.torch.quantization.tensor_quant import QuantDescriptor, QUANT_DESC_8BIT_PER_TENSOR
from modelopt.torch.quantization.nn.modules import _utils
from modelopt.torch.quantization import quant_modules
quant_modules.initialize()

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