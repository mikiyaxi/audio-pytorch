import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


'''
just so you know, real world data audio/image sometimes could get very large 
loading them all at once it's impossbile for memory sometimes 
Therefore, loading them with certain batch by batch could release that problem 
to achieve this is to create dataLoader in PyTorch 
'''
BATCH_SIZE = 128 
EPOCHS = 10
learning_rate = .001

class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__() 
        self.flatten = nn.Flatten() # store the information in layers 
        self.dense_layers = nn.Sequential(
            # a way to patch the data in a sequences of layers, so data float naturally
            nn.Linear(28*28, 256), # mnist image are 28*28 size 
            nn.ReLU(),
            nn.Linear(256, 10) # 10 is the number of class
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        '''indicate how the pytorch model will process the data'''
        flattened_data = self.flatten(input_data)
        # passing flatten data to dense layer, and get back output(logits)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions



# download dataset
def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",  # where the dataset will be installed or saved
        download=True,
        train=True,
        # apply transformation, take a image in and reshape the a new tensor that
        #  each value is normalize between 0 and 1
        transform=ToTensor()  
    )
    validation_data = datasets.MNIST(
        root="data",  # where the dataset will be installed or saved
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data


# train 
def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    ''' 
    create a loop go through all the sample in the dataset 
    each iteration will get a new batch of sample, both input(x) and target(y) per batch
    '''
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device) # send input to GPUs

        # calculate the loss 
        predictions = model(inputs) # pass the inputs to the model, get back prediction 
        loss = loss_fn(predictions, targets) # comparing 

        # backpropagate loss and update weights 
        optimizer.zero_grad()    # each iteration loss fn calculate gradients stored 
                                 # for each batch calculation, we want to start from zero
        loss.backward()  # backpropagation
        optimizer.step() # update gradient 

    print(f"Loss: {loss.item()}") # printing the loss at the end, for one epoch


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    '''going through multiple epochs, and each tun train one epoch'''
    for i in range(epochs):
        print("Epoch", i+1)
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("--------------")
    print("training is done")




if __name__ == "__main__":
    # download MNIST dataset 
    train_data, _ = download_mnist_datasets()
    print("MNIST dataset download")

    # create a data loader for the train set 
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = "mps"

    print("Using Device:", device)

    # build model, right now we are assigning which device to pass on, very simple just the name 
    feed_forward_net = FeedForwardNet().to(device)


    # instantiate loss function + optimizer 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(), 
                                 lr=learning_rate)

    # train model 
    train(feed_forward_net, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Model trained and store at feedforwardnet.pth")


