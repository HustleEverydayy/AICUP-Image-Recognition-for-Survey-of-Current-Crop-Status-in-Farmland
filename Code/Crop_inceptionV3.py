import PIL
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from torchvision import models
def argmax(lst): return max(range(len(lst)), key=lst.__getitem__)

classes=['banana', 'bareland', 'carrot', 'corn', 'dragonfruit', 'garlic', 'guava', 'peanut', 'pineapple', 'pumpkin', 'rice', 'soybean', 'sugarcane', 'tomato', 'building', 'mountain', 'sky']
num_classes = len(classes)
folders=['banana', 'bareland', 'carrot', 'corn', 'dragonfruit', 'garlic', 'guava', 'peanut', 'pineapple', 'pumpkin', 'rice', 'soybean', 'sugarcane', 'tomato']
nonfield=['building', 'mountain', 'sky']
class_to_index={'banana': 0, 'bareland': 1, 'building': 2, 'carrot': 3,
 'corn': 4, 'dragonfruit': 5, 'garlic': 6, 'guava': 7, 'mountain': 8,
 'peanut': 9, 'pineapple': 10, 'pumpkin': 11, 'rice': 12,
 'sky': 13, 'soybean': 14, 'sugarcane': 15, 'tomato': 16}
index_to_class = {v: k for k, v in class_to_index.items()}
#model = models.inception_v3(pretrained=False)
model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=False)
model.aux_logits = False
model.fc =  nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('fbl_pt2-E61.pth'))
model.eval()
model = model.cuda()

mydir = "subs"

test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


from pathlib import Path
def scan_image(fname, landtype_index):
    count = 0
    img_name = Path(fname).stem
    ext_name = Path(fname).suffix
    im = Image.open(fname)
    width, height = im.size   # Get dimensions
    new_width = 512
    new_height = 512
    stride = 512
    for startx in range(width//8, width*7//8-new_width, stride):
      for starty in range(height//2, height*7//8-new_height, stride):
          left = startx
          top = starty
          right = startx + new_width
          bottom = starty + new_height
          subname = f"{img_name}-{count:03}{ext_name}"
          count +=1
          im2 = im.crop((left, top, right, bottom))
          input_tensor = test_transform(im2)
          input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
          input_batch = input_batch.to('cuda')
          output = model(input_batch)
          probabilities = torch.nn.functional.softmax(output[0], dim=0).tolist()
          det = argmax(probabilities)
          if det ==2 or det ==8 or det ==13:#building, mountain sky
             subfolder = index_to_class[det]
          elif probabilities[landtype_index] > 0.3:
             subfolder = index_to_class[landtype_index]
          else:
             subfolder = index_to_class[landtype_index]+'_out' 
          name2 = os.path.join(mydir, subfolder, subname)
          print(name2)
          im2.save(name2)
          break

import glob
from PIL import Image 
for f in folders:
    landtype_index=class_to_index[f]
    subpath=os.path.join(f, "*")
    files = glob.glob(subpath)
    print(f"Categroy:{f}")
    for fname in files:
      print(fname)
      scan_image(fname, landtype_index)