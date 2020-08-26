import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse 

import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML

parser = argparse.ArgumentParser(description='Simple GAN network')
parser.add_argument('--nz', type=int,
                    help='an integer for the latent vector', default=100)
parser.add_argument('--epochs', type=int,
                    help='no of epochs', default=10)

args = parser.parse_args()

# Creating Custom Dataset to load npy file
class CamelDataset(Dataset):
	"""
	args

	root_dir: pass the directory of the .npy file
	transform: pass the list of transforms to be applied on images. Default is None

	"""
	def __init__(self, root_dir, transform=None):
		self.data = np.load(root_dir).reshape(-1, 1, 28, 28)
		self.transform = transform

	def __getitem__(self, index):
		x = self.data[index]
		if self.transform:
			x = self.transform(x)
			x = x.permute(1,2,0)
		return x

	def __len__(self):
		return len(self.data)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x


class Discriminator(nn.Module):
    def __init__(self, nc, ndc):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndc, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(num_features=ndc),
            nn.Dropout2d(p=0.25),
            #PrintLayer(),
            nn.Conv2d(ndc, ndc, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(num_features=ndc),
            nn.Dropout2d(p=0.25),
            #PrintLayer(),
            nn.Conv2d(ndc, ndc*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(num_features=ndc*2),
            nn.Dropout2d(p=0.25),
            #PrintLayer(),
            nn.Conv2d(ndc*2, ndc*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(num_features=ndc*2),
            nn.Dropout2d(p=0.25),
            #PrintLayer(),
        )
        self.fc1 = nn.Linear(2048, 1)
          
    def forward(self, input):
        x = self.main(input)
        x = x.view(-1, 2048)
        x = self.fc1(x)
        out = torch.sigmoid(x)
        return out


class Generator(nn.Module):
    def __init__(self, nz, ngc):
        super().__init__()
        self.nz = nz
        self.fc = nn.Linear(nz, 3136)
        self.bn = nn.BatchNorm1d(3136)
        self.main = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            #PrintLayer(),
            nn.Conv2d(ngc, ngc*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngc*2),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            #PrintLayer(),
            nn.Conv2d(ngc*2, ngc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngc),
            nn.ReLU(),
            #PrintLayer(),
            nn.Conv2d(ngc, ngc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngc),
            nn.ReLU(),
            #PrintLayer(),
            nn.Conv2d(ngc, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            #PrintLayer(),
            )
        
   
    def forward(self, input):
        
        z = F.relu(self.bn(self.fc(input)))
        z = z.view(-1, 64, 7, 7)
        z = self.main(z)
        
        return z


class GAN(nn.Module):
	def __init__(self, nz, epochs):
		super().__init__()
		self.device = ('cuda' if torch.cuda.is_available else 'cpu')
		self.epochs = epochs
		self.nz = nz

		ngpu = 1
		ndc = 64
		ngc = 64
		nc = 1

		transform = transforms.Compose([transforms.ToTensor()])
		camel_dataset = CamelDataset(root_dir='./data/camel_images/full_numpy_bitmap_camel.npy', transform=transform)
		self.dataloader = DataLoader(camel_dataset,batch_size=256,shuffle=True, num_workers=0)

		self.fixed_noise = torch.randn(128, nz, device=self.device)


		self.G = Generator(nz, ngc).to(self.device)
		self.D = Discriminator(nc, ndc).to(self.device)
		
		"""
		args

		weights_init: initialize the weights of the conv and BatchNorm layer of the network

		"""
		self.G.apply(weights_init)
		self.D.apply(weights_init)

		# Establish convention for real and fake labels during training
		self.real_label = 1.
		self.fake_label = 0.

		# Loss Function
		self.criterion = nn.BCELoss()

		# Setup RMSPROP optimizers for both G and D
		self.optimizerD = optim.RMSprop(self.D.parameters(), lr=0.0008)
		self.optimizerG = optim.RMSprop(self.G.parameters(), lr=0.0004)

	def train(self):
		# Lists to keep track of progress
		self.img_list = []
		self.G_losses = []
		self.D_losses = []
		iters = 0

		print("Starting Training Loop...")
		# For each epoch
		for epoch in range(self.epochs):
		    # For each batch in the dataloader
		    for i, data in enumerate(self.dataloader, 0):

		        ############################
		        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		        ###########################
		        ## Train with all-real batch
		        self.D.zero_grad()
		        # Format batch
		        real_cpu = data.to(self.device)
		        label = torch.full((data.shape[0],), self.real_label, dtype=torch.float, device=self.device)
		        # Forward pass real batch through D
		        output = self.D(real_cpu).view(-1)
		        # Calculate loss on all-real batch
		        errD_real = self.criterion(output, label)
		        # Calculate gradients for D in backward pass
		        errD_real.backward()
		        D_x = output.mean().item()

		        ## Train with all-fake batch
		        # Generate batch of latent vectors
		        noise = torch.randn(data.shape[0], self.nz, device=self.device)
		        # Generate fake image batch with G
		        fake = self.G(noise)
		        label.fill_(self.fake_label)
		        # Classify all fake batch with D
		        output = self.D(fake.detach()).view(-1)
		        # Calculate D's loss on the all-fake batch
		        
		        errD_fake = self.criterion(output, label)
		        # Calculate the gradients for this batch
		        errD_fake.backward()
		        D_G_z1 = output.mean().item()
		        # Add the gradients from the all-real and all-fake batches
		        errD = errD_real + errD_fake
		        # Update D
		        self.optimizerD.step()

		        ############################
		        # (2) Update G network: maximize log(D(G(z)))
		        ###########################
		        self.G.zero_grad()
		        label.fill_(self.real_label)  # fake labels are real for generator cost
		        # Since we just updated D, perform another forward pass of all-fake batch through D
		        output = self.D(fake).view(-1)
		        # Calculate G's loss based on this output
		        errG = self.criterion(output, label)
		        # Calculate gradients for G
		        errG.backward()
		        D_G_z2 = output.mean().item()
		        # Update G
		        self.optimizerG.step()

		        # Output training stats
		        if i % 50 == 0:
		            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
		                  % (epoch, self.epochs, i, len(self.dataloader),
		                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

		        # Save Losses for plotting later
		        self.G_losses.append(errG.item())
		        self.D_losses.append(errD.item())

		        # Check how the generator is doing by saving G's output on fixed_noise
		        if (iters % 200 == 0) or ((epoch == self.epochs-1) and (i == len(self.dataloader)-1)):
		            with torch.no_grad():
		                fake = self.G(self.fixed_noise).detach().cpu()
		            self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

		        iters += 1

	def create_animator(self):
		fig = plt.figure(figsize=(10,10))
		plt.axis("off")
		ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in self.img_list]
		ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
		ani.save('gan.gif', writer='imagemagick', fps=30)
		HTML(ani.to_jshtml())

	def plots(self):
		plt.figure(figsize=(10,5))
		plt.title("Generator and Discriminator Loss During Training")
		plt.plot(self.G_losses,label="G")
		plt.plot(self.D_losses,label="D")
		plt.xlabel("iterations")
		plt.ylabel("Loss")
		plt.legend()
		plt.savefig('loss_vs_iteration.png')



if __name__ == '__main__':
	gan = GAN(nz=args.nz, epochs=args.epochs)
	gan.train()
	gan.create_animator()
	gan.plots()

