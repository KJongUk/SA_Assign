from argparse import ArgumentParser
import torch
import sys
from utils.file import *
from torchmodel.pretrained import *
from onnxmodel.model import *

class Command:
    def __init__(self):
        self.parser = ArgumentParser()
        self.setup_argument()
        self.args = self.parser.parse_args()

    def setup_argument(self) -> None:
        self.parser.add_argument(
            '-m',
            "--mode",
            type=int,
            default=0,
            help="(0: store torch model as onnx format, 1: inference using onnx model)"
        )

        self.parser.add_argument(
            '-c',
            "--cuda",
            type=int,
            default=0,
            help="(0: use cpu, 1: use gpu)"
        )

        """ Options for saving torch model """
        self.parser.add_argument(
            '-p',
            "--pretrained",
            type=str,
            default="fasterrcnn_resnet50_fpn",
            help="Pretrained models supported by pytorch (default: fasterrcnn_resnet50_fpn)"
        )

        self.parser.add_argument(
            '-o',
            "--output",
            type=str,
            default="./",
            help="Directory path to store onnx model (default: ./)"
        )

        """ Options for Inference using onnx model """
        self.parser.add_argument(
            '-f',
            "--file",
            type=str,
            default=None,
            help="Path to onnx model for infference"
        )
        self.parser.add_argument(
            '-i',
            "--input",
            default="../test",
            help="Path to images for inferrence of onnx model (default: ../test)"
        )
        return

    def run(self) -> int:
        use_cuda = False
        if self.args.cuda and torch.cuda.is_available():
            use_cuda = True

        if self.args.mode == 0:
            if not os.path.exists(self.args.output):
                os.makedirs(self.args.output)
            return Model(self.args.pretrained, use_cuda).store(self.args.output)
        elif self.args.mode == 1:
            images = get_images(self.args.input)
            return OnnxModel(self.args.file,use_cuda).run(images)
        else:
            """ undefined mode """
            return 126

def main(*argv) -> None:
    try:
        exit(Command().run())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        exit(255)

if __name__ == "__main__":
    main(*sys.argv[1:])