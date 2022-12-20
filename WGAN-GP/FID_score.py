import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
_ = torch.manual_seed(123)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from glob import glob
import concurrent.futures
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
def reader(start_index,end_index,pth,IMGS):
    for idx in range(start_index,end_index):
        fname= pth[idx]
        IMGS[idx]=np.transpose(np.array(Image.open(fname).resize((64,64)).convert('RGB')) , (2,0,1))
    print("FINISH {0} to {1}".format(start_index,end_index))
def creator(PATH= "data/cars/stanford_cars/cars_train/*.jpg"):
    SIZE = 500#len(list(glob(PATH)))
    IMGS = np.zeros((SIZE,3,64,64))
    INDEX =0 
    step = 20
    PATH = list(glob(PATH))[0:500]
    print("started .. .   .    .")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(reader, start,min(start+step, len(PATH)),PATH, IMGS): start for start in range(0,len(PATH),20)}
    imgs = torch.from_numpy(IMGS)
    imgs= imgs.type(torch.uint8)
    return imgs
AllEpoch = 200
Fake_path= "dataset/result_global_view/Fake/Epoch{0}/*.jpeg"
gt_loader= creator()
writer = open("writeing_128color.txt","w")
fid = FrechetInceptionDistance(feature=64)
Z = copy.deepcopy(gt_loader[0:320])
fid.update(Z, real=True)
for i in range(AllEpoch):
    All = 0
    summation = 0
    fake_img = creator(Fake_path.format(i))
    Y= copy.deepcopy(fake_img[0:320])
    fid.update(Y, real=False)
    Value = fid.compute().numpy()
    print(Value)
    #summation+=fid.compute()
    print("Epoch {0} : {1}".format(i,Value))
    writer.write("Epoch {0} : {1}\n".format(i,Value))
writer.close()
