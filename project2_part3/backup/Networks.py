import numpy
import torch
import torch.nn as nn
import Data_Loaders

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        self.input_size = 6
        self.hidden_size = 200
        self.output_size = 1
        super(Action_Conditioned_FF,self).__init__()
        self.input_to_hidden = nn.Linear(self.input_size, self.hidden_size)
        self.nonlinear_ativation = nn.Sigmoid()
        self.hidden_to_output = nn.Linear(self.hidden_size,self.output_size)

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden(input)
        hidden = self.nonlinear_ativation(hidden)
        network_output = self.hidden_to_output(hidden)
        return network_output


    def evaluate(self, model, test_loader, loss_function):
        
        predictions = []
        actual = []
        for idx, sample in enumerate(test_loader):
                #print(sample['input'])
                yhat = model(sample['input'])
                y = sample['label']
                #print(yhat)
                #print(y)
                predictions.append(yhat)
                actual.append(y)
        
        #print(torch.stack(predictions))
        #print(torch.stack(actual))

        predictions = torch.stack(predictions)
        actual =  torch.stack(actual)
        
        actual_shape = actual.shape

        
        actual = actual.reshape([actual_shape[0],1])

        #print(predictions.shape)
        #print(actual.shape)

        #print(predictions[0].dtype)
        #print(actual[0].dtype)
        #print(numpy.shape(predictions))
        #print(numpy.shape(actual))
        #print(type(predictions[0]))
        #print(type(actual[0]))
        #ll = nn.MSELoss()
        loss = loss_function(predictions,actual)        
        #print(loss.item())
        return loss.item()
        #_, _ = sample['input'], sample['label']
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        #return loss

def main():
    batch_size = 16
    model = Action_Conditioned_FF()
    data_loaders = Data_Loaders.Data_Loaders(10)
    model.evaluate(model,data_loaders.test_loader, nn.MSELoss())


if __name__ == '__main__':
    main()
