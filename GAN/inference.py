#Import packages
#--------------------------
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torchvision.datasets as dset
from torch.autograd import Variable
import torch
import model_architecture
from model_architecture import opt

os.makedirs("output_training", exist_ok=True)

#todo remove packages not used


#Define (hyper-)parameters
#--------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension") #default=28; attention: if this is changed, then the architecture of the discriminator and generator must be changed! #todo
parser.add_argument("--channels", type=int, default=3, help="number of image channels") #default=1; color images is 3
opt = parser.parse_args()
img_shape = (opt.channels, opt.img_size, opt.img_size) #todo image size siehe display größe...


#Initialize generator
#--------------------------
generator = model_architecture.Generator()


#Hardware settings for the model training/inference
#--------------------------
ngpu = 0 #amount of GPUs; 0=CPU
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



#Optimizers
#--------------------------
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))



# load existing model
generator.load_state_dict(torch.load("models/existing_generator.pth"))
generator.eval()
optimizer_G.load_state_dict(torch.load("models/existing_G_optimizer.pth"))

img_list = []

#Inference
#--------------------------



optimizer_G.zero_grad()

# Sample noise as generator input
z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
# Generate a batch of images
gen_img = generator(z)


optimizer_G.step()

save_image(gen_img.data[:25], "output_inference/inference.png", nrow=5, normalize=True)

#todo show image instead of save
plt.figure(figsize=(9,9))
plt.axis("off")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()