import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


def training_loader_construct(dataset, batch_num, Shuffle):

    # construct the train loader given the dataset and batch size value
    # this function can be used for all different cases 

    train_loader = DataLoader(
        dataset,
        batch_size=batch_num,
        shuffle=Shuffle,                     # change the sequence of the data every time
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader

# define training loader construction function
class MyDataset(Dataset):
    def __init__(self, data, input_length, target_length, mean, std, transform=None):
        '''
        Encoder input length is input_length
        decoder input length is label_length
        output length is target_length
        '''

        '''
        input
            data: np.array, [simulation_number, timesteps, state of system]
            mean: scalar
            std: scalar
        '''

        '''
        return data loader of :(Batch, Timestep, Feature_dim)
        '''

        print('data shape:', data.shape)
        num_simulation, timesteps, feature_dim = data.shape

        # starting point and endding point of the data samples
        starting_point = input_length
        endding_point = data.shape[1] - target_length

        # store the input and target value, return list of (T,F)
        self.x = []
        self.y = []
        for p in range(num_simulation):
            for j in range(starting_point, endding_point):
                self.x.append((data[p,j-input_length:j, :] - mean) / std)
                self.y.append(data[p,j:j+target_length, :])
        self.x = torch.from_numpy(np.array(self.x)).float()
        self.y = torch.from_numpy(np.array(self.y)).float()

        self.transform = transform
        
    def __getitem__(self, index):

        x = self.x[index]
        y = self.y[index]

        return x, y
    
    def __len__(self):

        assert self.x.shape[0]==self.y.shape[0], 'length of input and output are not the same'   

        return self.x.shape[0]  

def trainingset_construct(data, batch_val, input_length, target_length, Shuffle, mean, std):
    dataset = MyDataset(data, input_length, target_length, mean, std)
    train_loader = training_loader_construct(dataset = dataset, batch_num = batch_val, Shuffle=Shuffle)

    return train_loader