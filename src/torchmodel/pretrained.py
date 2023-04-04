import os
import torch
from torchvision.models.detection import *
from torch.onnx import export
from utils.info import MODEL_WEIGHT

"""
Pretrained Model List:
- fcos_resnet50_fpn
- fasterrcnn_mobilenet_v3_large_320_fpn
- fasterrcnn_mobilenet_v3_large_fpn
- fasterrcnn_resnet50_fpn_v2
- fasterrcnn_resnet50_fpn
- retinanet_resnet50_fpn_v2
- retinanet_resnet50_fpn
- ssd300_vgg16
- ssdlite320_mobilenet_v3_large
"""

class Model:
    def __init__(self, model, use_cuda=False):
        self.model = model
        self.cuda = use_cuda
        
    def store(self, output):
        torch_model = self._get_models()
        input_dummy = torch.empty(1, 3, 320, 320, dtype = torch.float32)
        if self.cuda:
            input_dummy = input_dummy.cuda()
            torch_model = torch_model.cuda()

        torch_model.eval()

        if os.path.isdir(output):
            output+= self.model+".onnx"

        dynamic_axes = {
            'input' : {0: 'batch_size', 1: 'channel', 2: 'y', 3:'x'},
            'output' : {0: 'batch_size'}
        }
        
        export(
            torch_model,
            input_dummy,
            output,
            verbose=True,
            export_params = True,
            input_names = ['input'], 
            output_names = ['output'],
            dynamic_axes = dynamic_axes
        )
        return 0

    def _get_models(self):
        """
        Use pretrained models with weight trained COCO dataset.
        """
        weights = eval(MODEL_WEIGHT[self.model]).DEFAULT    
        return globals()[self.model](weights=weights, box_score_thresh=0.8)
