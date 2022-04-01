#Import packages
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import GAN.model_architecture as model_architecture
from GAN.model_architecture import opt, img_shape

def generate_new_GAN_img(model_path, output_path):

    #Initialize generator
    generator = model_architecture.Generator()

    #Hardware settings for the model training/inference
    ngpu = 0 #amount of GPUs; 0=CPU
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # load existing model
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    #Inference
    z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim)))) # Sample noise as generator input
    gen_img = generator(z) # Generate image

    #display image
    # plt.figure(figsize=(9,9))
    # plt.axis("off")
    # plt.imshow(np.transpose(vutils.make_grid(gen_img[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()

    #save image
    save_image(gen_img.data[:1], output_path, nrow=1, normalize=True)

#Run the function
#generate_new_GAN_img()