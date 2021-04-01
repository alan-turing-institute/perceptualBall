# Utilities to make running code simpler
from PIL import Image
from torchvision import transforms


def open_and_resize(image_name):
    im = Image.open(image_name).convert("RGB")
    im = im.resize(size=(224, 224))
    return im


def normalise_and_preprocess(im):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize
    ])
    return preprocess(im)


def open_and_preprocess(image_name):
    im = open_and_resize(image_name)
    return normalise_and_preprocess(im)
