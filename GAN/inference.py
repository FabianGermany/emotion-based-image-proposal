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
from model_architecture import opt, img_shape

os.makedirs("output_training", exist_ok=True)

#todo remove packages not used


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


# load existing model
#--------------------------
generator.load_state_dict(torch.load("models/existing_generator.pth"))
generator.eval()

#Inference
#--------------------------
z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim)))) # Sample noise as generator input
gen_img = generator(z) # Generate image

#display image
#--------------------------
plt.figure(figsize=(9,9))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(gen_img[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

#save image
#--------------------------
save_image(gen_img.data[:1], "output_inference/inference.png", nrow=1, normalize=True)
