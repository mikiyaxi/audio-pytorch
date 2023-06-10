
from torch import nn 
from torchsummary import summary # pip install torchsummary if not exist



#  vanilla convolutional neural network 
class CNN(nn.Module):

    def __init__(self):
        super().__init__() 
        # structure: 4 conv blocks / flatten / linear / softmax 
        '''
        think of nn.Sequential() as a contain that make the calculationg smooth from one  to one sequentially 
        each nn.Sequential() contain is a layer
        '''
        # first convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1, 
                out_channels = 16, # 16 filers in our convolutional network 
                kernel_size = 3,
                stride = 1, 
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,  # input channel has to be compatible with outout channel in the previous layers
                out_channels = 32, # increase the channel
                kernel_size = 3,
                stride = 1, 
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        #  thrid convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 64, # same here
                kernel_size = 3,
                stride = 1, 
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        #  forth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 128,
                kernel_size = 3,
                stride = 1, 
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # 
        self.flatten = nn.Flatten() 
        # 128 is from last conv1 layer, * 5 * 4 together is what we will get from nn.flatten()
        # 10 is the number of class label in the dataset 
        # you have to match the dimension with the next layer
        # self.linear = nn.Linear(128 * 5 * 4, 10) 
        self.linear = nn.Linear(128 * 5 * 4, 10) 
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_data):
        '''
        tell the pytorch structure how to pass data from layers to layers 
        '''

        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        logits = self.linear(x)
        predictions = self.softmax(logits)

        return predictions


if __name__ == "__main__":
    cnn = CNN() 
    # summary(cnn, (1, 64, 44))
