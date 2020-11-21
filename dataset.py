import torch
import numpy as np

def norm(x, m, s):
    return (x - m) / s

def augment(X):
    return torch.roll(X * np.random.uniform(low=0.8, high=1.1), np.random.randint(-150,150))

class SwingDataset(torch.utils.data.Dataset):
    def __init__(self, X_data, y_data, mean=None, std=None, y_mean=None, y_std=None, augment=False):
        super().__init__()
        self.X_data = torch.Tensor(X_data)
        self.y_data = torch.Tensor(y_data)
        self.augment = augment
        
        
        if mean is None: # assume that all the other values are None as well, this is the training set and we need to compute the mean/std
            self.mean = X_data.mean(axis=0)
            self.std = X_data.std(axis=0)
            self.y_dist_mean = y_data[:, -1].mean()
            self.y_dist_std = y_data[:, -1].std()
        else:
            self.mean = mean
            self.std = std
            self.y_dist_mean = y_mean
            self.y_dist_std = y_std
        
    def __len__(self):
        return self.X_data.shape[0]
    
    def __getitem__(self, idx):
        X = self.X_data[idx].clone()  # clone or else the normalization overwrites the actual tensor
        y = self.y_data[idx].clone()
        
        if self.augment:
            X = augment(X)
        
        X = norm(X, self.mean, self.std)
        y[-1] = norm(y[-1], self.y_dist_mean, self.y_dist_std) 
        
        return X, y 