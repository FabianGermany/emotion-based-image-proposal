#Import packages
#--------------------------
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
load_existing_model = True #decide whether to use an existing model (True) or to create a new one (False)



#Loss function
#--------------------------
adversarial_loss = torch.nn.BCELoss()

#Initialize generator and discriminator
#--------------------------
generator = model_architecture.Generator()
discriminator = model_architecture.Discriminator()

#Hardware settings for the model training/inference
#--------------------------
ngpu = 0 #amount of GPUs; 0=CPU
cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#Load dataset
#--------------------------
os.makedirs("dataset/landscape_complete", exist_ok=True)
dataset = dset.ImageFolder(root="dataset/landscape_complete",
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


#Load existing model if desired
#--------------------------
if(load_existing_model):
    generator.load_state_dict(torch.load("models/existing_generator.pth"))
    generator.eval()
    discriminator.load_state_dict(torch.load("models/existing_discriminator.pth"))
    discriminator.eval()
    optimizer_G.load_state_dict(torch.load("models/existing_G_optimizer.pth"))
    optimizer_D.load_state_dict(torch.load("models/existing_D_optimizer.pth"))


#Train models
#--------------------------
for epoch in range(opt.n_epochs): #for each epoch
    for i, (imgs, _) in enumerate(dataloader): #for each batch

        #Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        #Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #Train generator
        # -----------------

        optimizer_G.zero_grad()


        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        #Generate a batch of images
        gen_imgs = generator(z)

        #Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #Train discriminator
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

        #save exemplary images regularly
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "output_training/%d.png" % batches_done, nrow=5, normalize=True)
            # Up to 25 images will be displayed in that image. If the batch size is lower, then less images.

        #save models regularly
        if batches_done % opt.checkpoint_interval == 0:
            torch.save(discriminator.state_dict(), "models/existing_discriminator%d.pth" % batches_done)
            torch.save(generator.state_dict(), "models/existing_generator%d.pth" % batches_done)
            torch.save(optimizer_D.state_dict(), "models/existing_D_optimizer%d.pth" % batches_done)
            torch.save(optimizer_G.state_dict(), "models/existing_G_optimizer%d.pth" % batches_done)