"""
Training of WGAN-GP
"""

from tkinter.tix import IMAGE
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights
import numpy as np
import copy
from PIL import ImageFile
from glob import glob
import os
import cv2 as cv
import time

import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 100
NUM_EPOCHS = 110
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10


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
c=0
#TOTAL_LENGTH= len(list(glob("dataset/train/nudes/*.jpg")))
#IMAGES=np.zeros((TOTAL_LENGTH,3,64,64)) 
transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)
dataset1 = torchvision.datasets.StanfordCars(root="./data/cars", split = 'train', transform= transforms, download=True)
dataset2 = torchvision.datasets.StanfordCars(root="./data/cars", split = 'test', transform= transforms, download=True)
dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=torch.cuda.is_available()
)

for batch_idx, (data, target) in enumerate(loader):
    print('Batch idx {}, data shape {}, target shape {}'.format(
        batch_idx, data.shape, target.shape))
    inp= data[0]
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    #plt.imshow(inp)
    #plt.show()
    break

#dataset = datasets.ImageFolder("dataset/train/",transform=transforms)
c=0
# comment mnist above and uncomment below for training on CelebA
#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)


# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"datasets_reduced/logs_orginal/real")
writer_fake = SummaryWriter(f"datasets_reduced/logs_orginal/fake")
writer= open("resport_110Epoch_simplealgorithm.txt","w")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    timer = time.time()
    # Target labels not needed! <3 unsupervised
    print("The {0} epoch started".format(epoch))
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        print()
        # Print losses occasionally and print to tensorboard
        if batch_idx % 5 == 0 and batch_idx > 0:
            message = f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            print(message)
            writer.write(message+"\n")

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                if os.path.exists("dataset/result") == False:
                    os.mkdir("dataset/result")
                    os.mkdir("dataset/result/Real_theoriginalgorithm")
                    os.mkdir("dataset/result/Fake_theoriginalgorithm")
                SAVER_NAME = {"Fake":fake}
                for k in SAVER_NAME.keys():
                    X= copy.deepcopy(SAVER_NAME[k])
                    for batch_ireator in range(BATCH_SIZE-1):
                        try:
                            IMG= copy.deepcopy(X[batch_ireator].cpu().numpy().transpose((1, 2, 0)))
                            mean = np.array([0.5, 0.5, 0.5])
                            std = np.array([0.5, 0.5, 0.5])
                            IMG = std * IMG + mean
                            IMG = np.clip(IMG, 0, 1)
                            plt.imshow(IMG)
                            plt.imsave("dataset/result/Fake_theoriginalgorithm/IMG_{0}_{1}_{2}.jpeg".format(epoch,batch_idx,batch_ireator),IMG)
                            #plt.show()
                            #input()
                        except:
                            break
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
    writer.write("\n EPOCH TIME IS : {0}".format(timer - time.time()))

writer_fake.close()
writer_real.close()
writer.close()