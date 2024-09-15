import torch
from torch._C import device, dtype
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data.sampler import SubsetRandomSampler

class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',',dtype="float32")
        # STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
        # STUDENTS: __len__() returns the length of the dataset
        return len(self.data)

    
    def __getitem__(self, idx):
        # STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
        # x and y should both be of type float32. There are many other ways to do this, but to work with autograding
        # please do not deviate from these specifications.
        if not isinstance(idx, int):
            idx = idx.item()
                                                
        input = torch.from_numpy(self.data[idx][0:6])
        label = torch.tensor(self.data[idx][6])                                
        sample = {'input':input,'label':label}        
        return sample



class Data_Loaders():
    def __init__(self, batch_size):        
        # STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
        # make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        self.nav_dataset = Nav_Dataset()        
        dataset_size = self.nav_dataset.__len__()        
        train_size = int(0.8 * len(self.nav_dataset))
        test_size = len(self.nav_dataset) - train_size                
        self.train_loader, self.test_loader = torch.utils.data.random_split(self.nav_dataset, [train_size, test_size])


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):        
        _, _ = sample['input'], sample['label']        
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
