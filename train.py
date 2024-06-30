from ultralytics import YOLO
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Training Arguments")
    parser.add_argument("--model", type=str, default=None, help="Path to model file, e.g., yolov8n.pt, yolov8n.yaml")
    parser.add_argument("--data", type=str, default=None, help="Path to data file, e.g., coco128.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--patience", type=int, default=50, help="Epochs to wait for no observable improvement for early stopping")
    parser.add_argument("--batch", type=int, default=16, help="Number of images per batch (-1 for AutoBatch)")
    parser.add_argument("--imgsz", type=int, default=640, help="Size of input images as integer or w,h")
    parser.add_argument("--save", default=True, help="Save train checkpoints and predict results")
    parser.add_argument("--save_period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--cache", type=str, default=False, help="True/ram, disk or False. Use cache for data loading")
    parser.add_argument("--device", type=str, default=None, help="Device to run on, e.g., cuda device=0 or device=0,1,2,3 or device=cpu")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads for data loading (per RANK if DDP)")
    parser.add_argument("--project", type=str, default=None, help="Project name")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--exist_ok", action="store_true", help="Whether to overwrite existing experiment")
    parser.add_argument("--pretrained", action="store_true", help="Whether to use a pretrained model")
    parser.add_argument("--optimizer", type=str, default="auto", choices=["SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "auto"], help="Optimizer to use")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--deterministic", type=bool, default=True, help="Whether to enable deterministic mode")
    parser.add_argument("--single_cls", action="store_true", help="Train multi-class data as single-class")
    parser.add_argument("--rect", action="store_true", help="Rectangular training with each batch collated for minimum padding")
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine learning rate scheduler")
    parser.add_argument("--close_mosaic", type=int, default=0, help="(int) disable mosaic augmentation for final epochs")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--amp", type=bool, default=True, help="Automatic Mixed Precision (AMP) training")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction to train on")
    parser.add_argument("--profile", action="store_true", help="Profile ONNX and TensorRT speeds during training for loggers")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate (lr0 * lrf)")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum/Adam beta1")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Optimizer weight decay")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="Warmup epochs (fractions ok)")
    parser.add_argument("--warmup_momentum", type=float, default=0.8, help="Warmup initial momentum")
    parser.add_argument("--warmup_bias_lr", type=float, default=0.1, help="Warmup initial bias lr")
    parser.add_argument("--box", type=float, default=7.5, help="Box loss gain")
    parser.add_argument("--cls", type=float, default=0.5, help="Cls loss gain (scale with pixels)")
    parser.add_argument("--dfl", type=float, default=1.5, help="Dfl loss gain")
    parser.add_argument("--pose", type=float, default=12.0, help="Pose loss gain (pose-only)")
    parser.add_argument("--kobj", type=float, default=2.0, help="Keypoint obj loss gain (pose-only)")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing (fraction)")
    parser.add_argument("--nbs", type=int, default=64, help="Nominal batch size")
    parser.add_argument("--overlap_mask", type=bool, default=True, help="Masks should overlap during training (segment train only)")
    parser.add_argument("--mask_ratio", type=int, default=4, help="Mask downsample ratio (segment train only)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Use dropout regularization (classify train only)")
    parser.add_argument("--val", type=bool, default=True, help="Validate/test during training")
    # add mlflow run_name argument default to None
    parser.add_argument('--qat', action='store_true', help='fine tuning for QAT')
    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)

    args_dict = vars(args)

    model.train(**args_dict) # model -> DetectionTrainer
    metrics = model.val()  # evaluate model performance on the validation set
    
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format