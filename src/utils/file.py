import os 

def get_images(path):
    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split('.')[-1]=="jpg":
                images.append(os.path.join(root,file))
    return images