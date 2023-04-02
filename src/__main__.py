from argparse import ArgumentParser
import sys
from pytorch.model_torch import *
from myonnx.model_onnx import *
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


def main(*argv) -> None:
    parser = ArgumentParser()
    parser.add_argument(
        '-m',
        "--mode",
        dtype=int, 
        default=0,
        help="(0: store torch model as onnx format, 1: inference using onnx model)"
    )
    parser.add_argument('--cuda',dtype=int,default=0, help = "")
    parser.add_argument('-o','--output', type=str, help="")
    parser.add_argument('-i','--input', type=str, help = "")
    args = parser.parse_args()
    
    if args.mode == 0:
        Model().store()
    elif args.mode == 1:
        labels = FasterRCNN_ResNet50_FPN_V2_Weights.meta["categories"]
        onnx_model = Onnx_Model(
            "./onnx_model.onnx",
            labels).run("../test/grace_hopper_517x606.jpg")
    else:
        pass

    return

if __name__ == "__main__":
    main(*sys.argv[1:])