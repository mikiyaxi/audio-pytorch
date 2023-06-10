
from train import FeedForwardNet, download_mnist_datasets 
import torch

class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]



# prediction 
def predict(model, input, target, class_mapping):
    model.eval() # torch builtin: dropout or anything that's not needed will turn down 
    with torch.no_grad():
        predictions = model(input) # pass input to the model 
        # Tensor Object with specific dim: Tensor(1, 10) => (one sample, 10 classes)
        # [[0.1, 0.01, ...., 0.6]], interested in the index that has highest value
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]

    return predicted, expected



if __name__ == "__main__":

    # load back the model 
    feed_forward_net = FeedForwardNet() 
    state_dict = torch.load("feedforwardnet.pth") # loading model 

    feed_forward_net.load_state_dict(state_dict) # load the state dictionary back to the model 

    # load MNIST validation dataset 
    _, validation_data = download_mnist_datasets()

    # get a sample from the validation dataset for inference 
    '''
    both the input(x) and the target(y)
    for the following equation, [0][0]: looks like this (input, target)
    input sample of the first sample in the validation_data
    [0][1]: target from first sample
    '''
    input, target = validation_data[0][0], validation_data[0][1]

    # make an inference 
    predicted, expected = predict(feed_forward_net, input, target, 
                                  class_mapping) # mapping string label with digits

    print("Predicted:", predicted, "Expected:", expected)
