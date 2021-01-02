import os
import sys
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

a = 0

def show(mask, b, h, w, m):
    torch.set_printoptions(threshold=10000)
    name = str(random.randint(1, 1000000)) + '.jpg'
    path = os.path.join('/home/ziyechen/files', name)
    mask = mask.view(b, -1, h, w)           # (b, g*m, h, w)
    mask = F.interpolate(mask, (256, 128))  # (b, g*m, h, w)
    mask = mask.view(-1, m, 256, 128)       # (b*g, m, h, w)
    mask = F.softmax(mask, dim=1)           # (b*g, m, h, w)
    #mask = mask.permute(0, 2, 1, 3).contiguous().view(-1, m * 128) # (b*g*h, m*w)
    mask = mask[0].view(-1, 128)
    mask = (mask.numpy() * 255.0).astype(dtype=np.uint8)
    cv2.imwrite(path, mask)
    print('yes!')
    return