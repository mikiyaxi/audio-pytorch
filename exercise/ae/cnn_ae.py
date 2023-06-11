
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


class Autoencoder_cnn(nn.Module):
    def __init__(self):
        # N, 1, 28, 28
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # (channel, first num of output channel, kernel) = (N, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7), # N, 64, 7: increase the channel, but decrease the image sizes
        )

        # reconstruct the latent space to original 
        # given shape: N, 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), # N, 32, 7, 7
            nn.ReLU(),
            # N, 16, 13, 13 -- output_padding --> N, 16, 14, 14
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid() # keeping the output data value in between 0 and 1
        )

    # forward
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# you can use nn.MaxPool2d | nn.MaxUnpool2d if needed


# adding device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build the mdoel 
if __name__ == "__main__":

    print("training on:", device)
    # initialize model 
    model = Autoencoder_cnn()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # training:
    num_epochs = 10 
    outputs = []
    for epoch in range(num_epochs):
        for img, _ in data_loader:
            img = img.to(device)
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 


        print(f"Epoch:{epoch+1}:, Loss: {loss.item():.4f}")
        outputs.append((epoch, img, recon))


# Plotting 
for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i > 9: break 
        plt.subplot(2, 9, i+1)
        # item: 1, 28, 28 
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9: break 
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1 
        item = item.reshape(-1, 28, 28)
        # item: 1, 28, 28 
        plt.imshow(item[0])
    
    plt.show()

