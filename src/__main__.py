from argparse import ArgumentParser
import torch
import sys
from torchmodel.pretrained import *
from onnxmodel.model import *
from torchvision.models.detection import *

class Command:
    def __init__(self):
        self.parser = ArgumentParser()
        self.setup_argument()
        self.args = self.parser.parse_args()

    def setup_argument(self) -> None:
        self.parser.add_argument(
            '-m',
            "--mode",
            dtype=int,
            default=0,
            help="(0: store torch model as onnx format, 1: inference using onnx model)"
        )

        self.parser.add_argument(
            '-c',
            "--cuda",
            dtype=int,
            default=0,
            help="(0: use cpu, 1: use gpu)"
        )

        """ Options for saving torch model """
        self.parser.add_argument(
            '-p',
            "--pretrained",
            dtype=str,
            default="fasterrcnn_resnet50_fpn",
            help="Pretrained models supported by pytorch (default: fasterrcnn_resnet50_fpn)"
        )

        self.parser.add_argument(
            '-o',
            "--output",
            dtype=str,
            default="./",
            help="Directory path to store onnx model (default: ./)"
        )

        """ Options for Inference using onnx model """
        self.parser.add_argument(
            '-f',
            "--file",
            default=None,
            help="Path to onnx model for infference"
        )
        self.parser.add_argument(
            '-i',
            "--input",
            dtype=str,
            default="../test",
            help="Path to images for inferrence of onnx model (default: ../test)"
        )
        return

    def run(self) -> int:
        use_cuda = False
        if self.args.cuda and torch.cuda.is_available():
            use_cuda = True

        if self.args.mode == 0:
            if not os.path.exists(self.args.output) and os.path.isdir(self.args.output):
                os.makedirs(self.args.output)
            return Model(self.args.pretrained, use_cuda).store(self.args.output)
        elif self.args.mode == 1:
            return Onnx_Model(self.args.file, use_cuda).run(self.args.input)
        else:
            """ undefined mode """
            return 126

def main(*argv) -> None:
    try:
        exit(Command().run())
    except Exception as e:
        print(f"Error: {e.message}", file=sys.stderr)
        exit(255)

if __name__ == "__main__":
    main(*sys.argv[1:])