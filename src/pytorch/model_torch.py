import torch
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
"""
Model List
"""

class Model:
    def __init__(self, model = None):
        self.model = model
        
    def store(self, output = None):
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        torch_model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        # 
        torch_model.eval()

        input_dummy = torch.empty(1, 1, 224, 224, dtype = torch.float32)

        if not output:
            output = "onnx_model.onnx"

        torch.onnx.export(
            torch_model,
            input_dummy,
            output,
            export_params = True,
            input_names = ['input'], 
            output_names = ['cls_score','bbox_pred']
        )
        return 
