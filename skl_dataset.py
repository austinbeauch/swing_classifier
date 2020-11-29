import numpy as np


def norm(x, m, s):
    return (x - m) / s



class SKLSwingDataset():
    def __init__(self, X_data, y_data, mean=None, std=None, y_mean=None, y_std=None):
        super().__init__()
        self.X_data = X_data
        self.y_data = y_data
                
        
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
        X = self.X_data[idx].clone()  
        y = self.y_data[idx].clone()
        
                
        X = norm(X, self.mean, self.std)
        y[-1] = norm(y[-1], self.y_dist_mean, self.y_dist_std) 
        
        return X, y 