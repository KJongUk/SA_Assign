import os 
import numpy as np

def get_images(path):
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split('.')[-1]=="jpg":
                images.append(os.path.join(root,file))
    return images

def compare_two_array(r, c, layer):
    """
    Compare ONNX model layer and Pytorch model layer
    """

    f = False
    try : 
        np.testing.assert_allclose(r, c, rtol=1e-5, atol=0)
        print(layer + ": no difference.")
    except layer as msg:
        print(layer + ": Error.")
        print(msg)
        f = True
    return f


def compare_results(r, c, rtol=1e-5):
    """
    Compare ONNX model layer and Pytorch model layer
    """
    f = False
    try : 
        np.testing.assert_allclose(r, c, rtol=rtol, atol=0)
    except Exception as e:
        f = True
    return f