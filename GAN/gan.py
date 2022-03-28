#todo here is an example https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

#Import packages
#--------------------------
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import math

import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("../output", exist_ok=True)


#Define (hyper-)parameters
#--------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training") #default=200
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches") #default=64
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension") #default=28; attention: if this is changed, then the architecture of the discriminator and generator must be changed! #todo
parser.add_argument("--channels", type=int, default=3, help="number of image channels") #default=1; color images is 3
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between saving model checkpoints")

opt = parser.parse_args()
print(opt)
img_shape = (opt.channels, opt.img_size, opt.img_size) #todo image size siehe display größe...
print("Image Shape: " + str(img_shape))

ngpu = 0 #amount of GPUs; 0=CPU
cuda = True if torch.cuda.is_available() else False

#Define generator model
#--------------------------
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


#Define discriminator model
#--------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


#Loss function
#--------------------------
adversarial_loss = torch.nn.BCELoss()

#Initialize generator and discriminator
#--------------------------
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()



#Load dataset
#--------------------------

# os.makedirs("../dataset/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../dataset/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

#todo transforn resize usw.
os.makedirs("../dataset/landscape_complete", exist_ok=True)
dataset = dset.ImageFolder(root="../dataset/landscape_complete",
                           transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.CenterCrop(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))



#Configure the data loader
#--------------------------
dataloader = torch.utils.data.DataLoader(
    dataset= dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)


device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#Plot some exemplary training images
#--------------------------
real_batch = next(iter(dataloader))
plt.figure(figsize=(6,6))
plt.axis("off")
plt.title("Exemplary Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()


#Optimizers
#--------------------------
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#Training
#--------------------------
for epoch in range(opt.n_epochs): #for each epoch
    for i, (imgs, _) in enumerate(dataloader): #for each batch

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        #--------------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i

        #save images regularly
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "../output/%d.png" % batches_done, nrow=5, normalize=True)

        # save models regularly
        # if batches_done % opt.checkpoint_interval == 0:
        #     torch.save(optimizer_D.state_dict(), "models/Discriminator%d.pth" % batches_done)
        #     torch.save(optimizer_G.state_dict(), "models/Generator%d.pth" % batches_done)