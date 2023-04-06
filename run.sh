#!/bin/bash

# Pytorch Model List
MODEL="
fcos_resnet50_fpn
fasterrcnn_mobilenet_v3_large_fpn
fasterrcnn_resnet50_fpn_v2
fasterrcnn_resnet50_fpn
retinanet_resnet50_fpn_v2
retinanet_resnet50_fpn
ssd300_vgg16
ssdlite320_mobilenet_v3_large
"

if [ $# -eq 0 ] ; then
    echo "USAGE: ./run.sh (store|run|test)\n"
    echo "-store: Convert pretrained pytorch model to onnx model\n"
    echo "-run: Run object detection with onnx model\n"
    echo "-test: Perform with provided settings\n"
    exit 0
fi

if [ ${1} = "store" ];then
    
    echo "Pretrained Pytorch Model List:\n"
    for model in $MODEL
    do
        echo "- "$model;
    done
    echo "Enter the Model selected from the list above: "
    read TORCH
    echo "Enter Output Path for ONNX Model: "
    read OUTPUT
    echo "Use GPU (y/n)?: "
    read GPU
    USE=0
    if [ $GPU = "y" ];then
        USE=1
    fi
    python3 ./src/__main__.py -m 0 -c $USE -p $TORCH -o $OUTPUT
elif [ ${1} = "run" ]; then
    BENCHMARK="./benchmarks"
    echo "Enter Path of ONNX Model: "
    read ONNX
    echo "Use GPU (y/n)?: "
    read GPU
    USE=0
    if [ $GPU = "y" ]; then
        USE=1
    fi
    python3 ./src/__main__.py -m 1 -c $USE -f $ONNX -i $BENCHMARK
elif [ ${1} = "test" ]; then
    BENCHMARK="./benchmarks"
    ONNX="./onnx/fasterrcnn_resnet50_fpn.onnx"
    python3 ./src/__main__.py -m 1 -f $ONNX1 -i $BENCHMARK
elif [ ${1} = "alltest" ]; then
    BENCHMARK="./benchmarks"
    ONNX1="./onnx/fcos_resnet50_fpn.onnx"
    ONNX3="./onnx/fasterrcnn_mobilenet_v3_large_fpn.onnx"
    ONNX4="./onnx/fasterrcnn_resnet50_fpn_v2.onnx"
    ONNX5="./onnx/fasterrcnn_resnet50_fpn.onnx"
    ONNX6="./onnx/retinanet_resnet50_fpn_v2.onnx"
    ONNX7="./onnx/retinanet_resnet50_fpn.onnx"
    ONNX8="./onnx/ssd300_vgg16.onnx"
    ONNX9="./onnx/ssdlite320_mobilenet_v3_large.onnx"
    python3 ./src/__main__.py -m 1 -f $ONNX1 -i $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 1 -f $ONNX3 -i $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 1 -f $ONNX4 -i $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 1 -f $ONNX5 -i $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 1 -f $ONNX6 -i $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 1 -f $ONNX7 -i $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 1 -f $ONNX8 -i $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 1 -f $ONNX9 -i $BENCHMARK
elif [ ${1} = "all" ]; then
    BENCHMARK="./onnx"
    ONNX1="fcos_resnet50_fpn"
    ONNX3="fasterrcnn_mobilenet_v3_large_fpn"
    ONNX4="fasterrcnn_resnet50_fpn_v2"
    ONNX5="fasterrcnn_resnet50_fpn"
    ONNX6="retinanet_resnet50_fpn_v2"
    ONNX7="retinanet_resnet50_fpn"
    ONNX8="ssd300_vgg16"
    ONNX9="ssdlite320_mobilenet_v3_large"
    python3 ./src/__main__.py -m 0 -p $ONNX1 -o $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 0 -p $ONNX3 -o $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 0 -p $ONNX4 -o $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 0 -p $ONNX5 -o $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 0 -p $ONNX6 -o $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 0 -p $ONNX7 -o $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 0 -p $ONNX8 -o $BENCHMARK
    echo "\n\n"
    python3 ./src/__main__.py -m 0 -p $ONNX9 -o $BENCHMARK
else
   echo "USAGE: ./run.sh (store|run|test)\n"
   echo "-store: Convert pretrained pytorch model to onnx model\n"
   echo "-run: Run object detection with onnx model\n"
   echo "-test: Perform with provided settings\n"
   exit 0
fi

