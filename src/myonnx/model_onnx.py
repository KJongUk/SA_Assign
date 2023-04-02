import onnx
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


class Onnx_Model:
    def __init__(self, model, labels):
        self.session = onnxruntime.InferenceSession(model)
        self.labels =labels

    def run(self, path):
        img = Image.open(path)
        resize = transforms.Resize([224, 224])
        img = resize(img)
        img_ycbcr = img.convert('YCbCr')
        img_y, img_cb, img_cr = img_ycbcr.split()
        to_tensor = transforms.ToTensor()
        img_y = to_tensor(img_y)
        img_y.unsqueeze_(0)

        ort_inputs = {self.session.get_inputs()[0].name: img_y.detach().cpu().numpy()}
        ort_outs = self.session.run(None, ort_inputs)

        # Plot the bounding boxes on the image
        self._draw_box(ort_outs, img)
        return

    def _draw_box(self,outs, img):
        bboxes = outs[0].flatten()
        labels = outs[1].flatten()
        scores = outs[2].flatten()

        plt.figure()
        fig, ax = plt.subplots(1, figsize=(12,9))
        ax.imshow(img)

        for idx in range(len(scores)):
            if scores[idx] < 0.2:
                continue
            
            base_index = idx * 4
            y1, x1, y2, x2 = bboxes[base_index], bboxes[base_index + 1], bboxes[base_index + 2], bboxes[base_index + 3] 
            color = 'red'
            box_h = (y2 - y1)
            box_w = (x2 - x1)
            bbox = patches.Rectangle((y1, x1), box_h, box_w, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(bbox)
            label_text = "{} ({:0.2f})".format(self.labels[labels[idx]], scores[idx])
            plt.text(y1, x1, s=label_text, color='white', verticalalignment='top', bbox={'color': "red", 'pad': 0})

        plt.axis('off')
        plt.show()
        return