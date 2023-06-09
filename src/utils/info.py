MODEL_WEIGHT = {
    "fcos_resnet50_fpn": "FCOS_ResNet50_FPN_Weights", 
    "fasterrcnn_resnet50_fpn_v2": "FasterRCNN_ResNet50_FPN_V2_Weights", 
    "fasterrcnn_resnet50_fpn": "FasterRCNN_ResNet50_FPN_Weights", 
    "retinanet_resnet50_fpn_v2": "RetinaNet_ResNet50_FPN_V2_Weights", 
    "retinanet_resnet50_fpn": "RetinaNet_ResNet50_FPN_Weights", 
    "ssd300_vgg16": "SSD300_VGG16_Weights", 
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights", 
}

MODEL_INPUT = {
    "fcos_resnet50_fpn": None, 
    "fasterrcnn_resnet50_fpn_v2": None, 
    "fasterrcnn_resnet50_fpn": None, 
    "retinanet_resnet50_fpn_v2": None, 
    "retinanet_resnet50_fpn": None, 
    "ssd300_vgg16": None, 
    "ssdlite320_mobilenet_v3_large": [320,320], 
}

MODEL_RES = {
    "fcos_resnet50_fpn": 2, 
    "fasterrcnn_resnet50_fpn_v2": 1, 
    "fasterrcnn_resnet50_fpn": 1, 
    "retinanet_resnet50_fpn_v2": 2, 
    "retinanet_resnet50_fpn": 2, 
    "ssd300_vgg16": 2, 
    "ssdlite320_mobilenet_v3_large": 2, 
}

labels =[
    "__background__","person","bicycle","car","motorcycle",
    "airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","N/A","stop sign","parking meter","bench","bird",
    "cat","dog","horse","sheep","cow","elephant",
    "bear","zebra","giraffe","N/A","backpack","umbrella",
    "N/A","N/A","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","N/A","wine glass",
    "cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","couch","potted plant",
    "bed","N/A","dining table","N/A","N/A","toilet",
    "N/A","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster",
    "sink","refrigerator","N/A","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"
]