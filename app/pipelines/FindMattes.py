from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models
import numpy as np

fcn = None
_trf = None
_label_colors = np.array([
    (0, 0, 0),       # 0=background
    (128, 0, 0),     # 1=aeroplane
    (0, 128, 0),     # 2=bicycle
    (128, 128, 0),   # 3=bird
    (0, 0, 128),     # 4=boat
    (128, 0, 128),   # 5=bottle
    (0, 128, 128),   # 6=bus
    (128, 128, 128), # 7=car
    (64, 0, 0),      # 8=cat
    (192, 0, 0),     # 9=chair
    (64, 128, 0),    # 10=cow
    (192, 128, 0),   # 11=dining table
    (64, 0, 128),    # 12=dog
    (192, 0, 128),   # 13=horse
    (64, 128, 128),  # 14=motorbike
    (192, 128, 128), # 15=person
    (0, 64, 0),      # 16=potted plant
    (128, 64, 0),    # 17=sheep
    (0, 192, 0),     # 18=sofa
    (128, 192, 0),   # 19=train
    (0, 64, 128),    # 20=tv/monitor
], dtype=np.uint8)


def getRotoModel():
    global fcn
    fcn = models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
    ).eval()
    torch.set_num_threads(4)


def _get_transform(size):
    global _trf
    if _trf is None or not hasattr(_trf, '_size'):
        _trf = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        _trf._size = size
    return _trf


def decode_segmap(image):
    h, w = image.shape
    rgb = _label_colors[image.reshape(-1)].reshape(h, w, 3)
    return rgb


def createMatte(filename, matteName, size):
    img = Image.open(filename).convert('RGB')
    trf = _get_transform(size)
    inp = trf(img).unsqueeze(0)
    if fcn is None:
        getRotoModel()
    with torch.no_grad():
        out = fcn(inp)['out']
    om = torch.argmax(out.squeeze(), dim=0).cpu().numpy()
    rgb = decode_segmap(om)
    im = Image.fromarray(rgb)
    im.save(matteName)


def createMatteBatch(filenames, matteNames, size):
    if fcn is None:
        getRotoModel()
    trf = _get_transform(size)
    batch = []
    for fn in filenames:
        img = Image.open(fn).convert('RGB')
        batch.append(trf(img))
    inp = torch.stack(batch, dim=0)
    with torch.no_grad():
        out = fcn(inp)['out']
    preds = torch.argmax(out, dim=1).cpu().numpy()
    for i, matteName in enumerate(matteNames):
        rgb = decode_segmap(preds[i])
        Image.fromarray(rgb).save(matteName)
