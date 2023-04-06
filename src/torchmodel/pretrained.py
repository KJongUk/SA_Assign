import os
import torch
import onnx
import onnx.numpy_helper as numpy_helper
from onnx import shape_inference
from torchvision.models.detection import *
from torch.onnx import export
from utils.info import MODEL_WEIGHT
from utils.file import *

"""
Pretrained Model List:
- fcos_resnet50_fpn
- fasterrcnn_resnet50_fpn_v2
- fasterrcnn_resnet50_fpn
- retinanet_resnet50_fpn_v2
- retinanet_resnet50_fpn
- ssd300_vgg16
- ssdlite320_mobilenet_v3_large
"""


class Model:
    def __init__(self, model):
        self.model = model
        
    def store(self, output):
        torch_model = self._get_models()
        input_dummy = torch.empty(1, 3, 320, 320, dtype = torch.float32)

        torch_model.eval()


        if os.path.isdir(output):
            output = os.path.join(output, self.model+".onnx")


        dynamic_axes = {
            'input' : {0: 'batch_size', 1: 'channel', 2: 'y', 3:'x'},
            'output' : {0: 'batch_size'}
        }
        
        export(
            torch_model,
            input_dummy,
            output,
            verbose=True,
            opset_version=11,
            export_params = True,
            input_names = ['input'],
            output_names = ['output'],
            dynamic_axes = dynamic_axes
        )
        
        self._update_onnx_model(torch_model, output)
        return 0


    def _get_models(self):
        """
        Use pretrained models with weight trained COCO dataset.
        """
        weights = eval(MODEL_WEIGHT[self.model]).DEFAULT
        return globals()[self.model](weights=weights, box_score_thresh=0.8)

    def _update_onnx_model(self,torch_model, onnx_path):
        """
        Compare ONNX model with torch model
        - Reference: https://gaussian37.github.io/dl-pytorch-deploy/
        """
    
        onnx_model = onnx.load(onnx_path)
        onnx_layers = dict()
        for layer in onnx_model.graph.initializer:
            onnx_layers[layer.name] = numpy_helper.to_array(layer)

        torch_layers = dict()

        for layer_name, layer_value in torch_model.named_modules():
            torch_layers[layer_name] = layer_value

        """
        e.g., 
        - onnx_layer : torch.nn.modules.linear.Linear.weight
        - torch_layer : torch.nn.modules.linear.Linear
        """
        onnx_layers_set = set(onnx_layers.keys())
        torch_layers_set = set(layer_name + ".weight" for layer_name in list(torch_layers.keys()))

        """ Get intersection set of layers (ONNX /\ Pytorch) """
        inter_layers = list(onnx_layers_set.intersection(torch_layers_set))



        different_layers = []

        for name in inter_layers:
            """ Check if a layer's tensor is different with torch layer """
            torch_layer_name = name.replace(".weight", "")
            onnx_weight = onnx_layers[name]
            torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
            flag = compare_two_array(onnx_weight, torch_weight, name)
            if flag:
                different_layers.append(name)

        if len(different_layers)>=1:
            print("Update ONNX Weight from Pytroch Model.")
            graph = onnx_model.graph
            for idx, layer in enumerate(graph.initializer):
                layer_name = layer.name
                if layer_name in different_layers:
                    onnx_layer_name = layer_name
                    torch_layer_name = layer_name.replace(".weight", "")
                    onnx_weight = onnx_layers[onnx_layer_name]
                    torch_weight = torch_layers[torch_layer_name].weight.detach().numpy()
                    copy_tensor = numpy_helper.from_array(torch_weight, onnx_layer_name)
                    onnx_model.graph.initializer[idx].CopyFrom(copy_tensor)
            print("Save Updated ONNX Model")
            os.remove(onnx_path)
            """ Stroe New ONNX Model """
            onnx.save(onnx_model, onnx_path)

        """ Store shape information """
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)
        return 0