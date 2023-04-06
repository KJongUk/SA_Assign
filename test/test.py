"""
Tests to verify ONNX model 
"""
import pytest, sys, logging, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.torchmodel.pretrained import *
from src.onnxmodel.model import *
from src.utils.file import *

BENCHMARKS = "./benchmarks"
MODELS = [
    "fcos_resnet50_fpn",
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "retinanet_resnet50_fpn",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large"
]
ONNX_MODELS = [
    "./onnx/fcos_resnet50_fpn.onnx",
    "./onnx/fasterrcnn_resnet50_fpn_v2.onnx",
    "./onnx/fasterrcnn_resnet50_fpn.onnx",
    "./onnx/retinanet_resnet50_fpn_v2.onnx",
    "./onnx/retinanet_resnet50_fpn.onnx",
    "./onnx/ssd300_vgg16.onnx",
    "./onnx/ssdlite320_mobilenet_v3_large.onnx"
]

#---------------------------------
# System Test
#---------------------------------

def differential_testing(model_idx):
    """
    Differential Testing Code
    - ONNX model vs Pytorch model when using cpu
    """

    images = get_images(BENCHMARKS)
    """
    Get object detection result from ONNX and Pytorch
    """
    onnx_results = OnnxModel(ONNX_MODELS[model_idx],0).test(images)
    torch_results = Model(MODELS[model_idx]).test(images)
    
    res = False
    for idx, onnx_res in enumerate(onnx_results):
        torch_res = torch_results[idx]
        """
        Compare the results (labels, boxes, scores)
        """

        torch_labels = torch_res["labels"].detach().numpy()
        onnx_labels = onnx_res[1]
        flag1 = compare_results(onnx_labels,torch_labels)
        res = res or flag1

        torch_boxes = torch_res["boxes"].detach().numpy()
        onnx_boxes = onnx_res[0]
        flag2 = compare_results(onnx_boxes,torch_boxes, 1e-2)
        res = res or flag2

        torch_scores = torch_res["scores"].detach().numpy()
        onnx_scores = onnx_res[2]
        flag3 = compare_results(onnx_scores,torch_scores, 0.2)
        
        res = res or flag3

    return res
   
   
def test_answer_model_2():
    assert differential_testing(2) == False

