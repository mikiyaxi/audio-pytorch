

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms 
import matplotlib.pyplot as plt 
# plt.use('Agg')

# transform method, a class that you can pass in with different transform method 
# later it can take in the data(images or audio) and transform the data into tensor 
# class "transform" -> method 1: ToTensor() -> specify it so that transform module can use
transform = transforms.ToTensor() 
'''
# this is a different transform method that will then value into [-1, 1] range tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
'''



# define the dataset 
mnist_data = datasets.MNIST(root='./data', train=True,
                            download=True, transform=transform) 

# dataloader 
data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=64,
                                          shuffle=True)

# inspect -> tensor range: tensor(0.) tensor(1.)
# later in the output layer, we need to make sure that all data fits in this range
images, labels = next(iter(data_loader))
print(torch.min(images), torch.max(images))

# AutoEncoder Class 
class Autoencoder_linear(nn.Module):
    def __init__(self):
        # N, 784: N is the batch size; 784 is the pixel size 
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128), # [N, 784] reduce to [N, 128]
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3) # from shape[N, 784] to [N, 3]
            # don't need an activation function for last layer
        )

        # reconstruct the latent space to original 
        # switch the size 
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), # convert back the dimensions
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28), # from shape[N, 784] to [N, 128]
            nn.Sigmoid() # keeping the output data value in between 0 and 1
        )

    # forward
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
'''
if the data range is between [-1, +1], we then apply nn.Tanh()
it could happen if the transform was calculate differently -> back to the top
'''

# build the mdoel 
if __name__ == "__main__":

    # initialize model 
    model = Autoencoder_linear()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # training:
    num_epochs = 10 
    outputs = []
    for epoch in range(num_epochs):
        for img, _ in data_loader:
            img = img.reshape(-1, 28*28)
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 


        print(f"Epoch:{epoch+1}:, Loss: {loss.item():.4f}")
        outputs.append((epoch, img, recon))


# display image 
for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i > 9: break 
        plt.subplot(2, 9, i+1)
        item = item.reshape(-1, 28, 28)
        # item: 1, 28, 28 
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break 
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1 
        item = item.reshape(-1, 28, 28)
        # item: 1, 28, 28 
        plt.imshow(item[0])
    
    plt.show()
