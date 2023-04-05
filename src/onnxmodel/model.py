import os, sys
import cv2
import onnx
import time
import torchvision.transforms as transforms
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from PIL import Image
from utils.info import *


class OnnxModel:
    def __init__(self, model, cuda):
        """
        Initialize OnnxModel
        - session: Load ONNX Model for inference
        - labels: categories for COCO dataset
        - cuda: usability of gpu
        - resize: Input demension for ONNX model
        """
        #self.onnx_model = onnx.load(model)
        self.session = onnxruntime.InferenceSession(
            model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.labels = labels
        self.cuda = cuda
        self.resize = MODEL_INPUT[self._get_model(model)]
        self.model_res = MODEL_RES[self._get_model(model)]

    def run(self, images):
        """
        Run ONNX model and Draw an image with bounding boxes.
        - (1) Inference with ONNX model.
        - (2) Print image with bounding box.
        - (3) Print inference time each image.
        """
        inference_times = []
        imgs = []
        outs = []

        total = len(images)

        for i in tqdm(range(total), desc="Obj Detection with Onnx Model"):
            img, img_ = self._preprocess(images[i])
            start = time.time()
            out = self._inference(img)
            inference_time = time.time()-start
            
            inference_times.append(inference_time)
            imgs.append(img_)
            outs.append(out)

        self._draw_results(outs, imgs, images, inference_times)
        self._print_time(images, inference_times)
        return 0

    def _preprocess(self,image):
        img = Image.open(image)
        if self.resize:
            resize = transforms.Resize([320,320])
            img = resize(img)
        to_tensor = transforms.ToTensor()
        img_ = to_tensor(img)

        # Add Batch dimenssion
        img_.unsqueeze_(0)
        return img_, img

    def _inference(self, img):
        ort_input = {self.session.get_inputs()[0].name: self._to_numpy(img)}
        ort_out = self.session.run(None, ort_input)
        return ort_out

    def _to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def _get_model(self,model):
        """ get model name """
        return (model.split('/')[-1]).split('.')[0]

    def _draw_results(self, outs, imgs, images, times):
        plt.figure()

        for i, out in enumerate(outs):
            img = imgs[i]
            boxes = out[0].flatten()

            if self.model_res == 1:
                res = out[1].flatten()   
                scores = out[2].flatten()
            else:
                res = out[2].flatten()   
                scores = out[1].flatten()

            fig, ax = plt.subplots(1, figsize=(12,9))
            ax.imshow(img)

            plt.title(
                images[i].split('/')[-1],
                fontdict={
                    'fontsize': 16,
                    'fontweight': 'bold'
                }
            )
            plt.xlabel("Inference Time: xxxxs")
            for idx, score in enumerate(scores):
                base_idx = idx*4
                y1, x1, y2, x2 = boxes[base_idx:base_idx+4]
                color = 'red'
                box_h = (y2-y1)
                box_w = (x2-x1)
                bbox = patches.Rectangle((y1, x1), box_h, box_w, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(bbox)
                
                label_text = "{} ({:0.2f})".format(self.labels[res[idx]], score)
                plt.text(y1, x1, s=label_text, color='white', verticalalignment='top', bbox={'color': "red", 'pad': 0})
            plt.axis('off')
        plt.show() 
        return

    def _print_time(self, images, inference_times):
        """
        """
        for i, image in enumerate(images):
            name = image.split('/')[-1]
            print(f"Image File: {name}, "
                f"inference-time: {inference_times[i]:.4f}s ",
                file=sys.stdout)
        return

