# SA_Assign
This repository contains a tool for running object detection using [ONNX model](https://onnx.ai/). It provides functions which transforms Pytorch pretrained model to ONNX format and which performs object detection using ONNX model.

## Installation
* Requirements:
- Python(>= 3.8)
- Pytorch(>= 1.9)

* Install Python libraries:
```bash
pip install -r requirements.txt
```

If you want to use GPU, please install python library:
```bash
pip install onnxruntime-gpu
```

* Other libraries for OpenCV:
```bash
apt-get -y install libgl1-mesa-glx libglib2.0-0
```

## Usage
To run our system, we provide a script file.

1. Convert Pytorch's pretrained model to ONNX format. 
```bash
sh ./run.sh store
```

2. Perform inference using ONNX model with [10 images](./benchmarks).
```bash
sh ./run.sh run
```

