import os
import glob

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

import utils
from model import BadModel, BadModel2
from dataset import SwingDataset, augment, oversample_minority
from sklearn.model_selection import KFold


def get_accuracy(predictions, truth, mean, std):
    acc = 0
    for pred, true in zip(predictions, truth):
        pred_argmax = torch.argmax(pred[:-1])
        pred_shot_type = shot_types[pred_argmax]
        
        true_argmax = torch.argmax(true[:-1])
        true_shot_type = shot_types[true_argmax]
        
        acc+=true_argmax == pred_argmax
    return acc / len(predictions)



def get_MSE(predictions, truth, mean, std):
    mse = 0
    for pred, true in zip(predictions, truth):
        pred_distance = pred[-1] * std + mean
        true_distance = true[-1] * std + mean
    mse += (pred_distance - true_distance)**2
    return mse**0.5
        


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path = "data/"
shot_types = ["Pull-hook", "Hook", "Pull", "Fade", "Straight", "Draw", "Push", "Slice" , "Push-slice"]

X_data, y_data = utils.load_data(path)

use_oversampling = True
if use_oversampling:
    X_data, y_data = oversample_minority(X_data, y_data)

k_fold = 3

n = X_data.shape[0]
kf = KFold(n_splits=k_fold)
kf.get_n_splits(X_data)

total_acc = 0
total_MSE = 0
for k_i, (train_index, test_index) in enumerate(kf.split(X_data)):
    
    
    print(train_index,test_index)
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    print(X_train.shape)
    # GENERATE DATA SPLITS

    # /GENERATE DATA SPLITS

    train_set = SwingDataset(X_train, y_train, augment=True, oversample = True)
    test_set = SwingDataset(X_test, y_test, mean=train_set.mean, std=train_set.std, y_mean=train_set.y_dist_mean, y_std=train_set.y_dist_std)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, drop_last=False, shuffle=True)

    bestmodel_file = os.path.join("weights", "best_model.pth")

    model = BadModel2().to(device)

    swing_type_loss = nn.CrossEntropyLoss()
    distance_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    losses = []
    iter_idx = -1
    train_interval = 50
    report_interval = 10

    loss_history = []
    epochs = 100

    train_losses = []
    train_losses_swing = []
    train_losses_dist = []

    for e in range(epochs):
        prefix = "Epoch {:3d}: ".format(e)
    #     for data in tqdm(train_loader, desc=prefix):
        for data in train_loader:
            print(e, end="\r", flush=True)
            iter_idx += 1
            X_train, y = data
            
            X_train = X_train.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_train.float())
            swing_loss = swing_type_loss(outputs[:, :-1], torch.max(y[:, :-1], 1)[1])
            dist_loss =  distance_loss(outputs[:, -1:], y[:, -1:])
            loss = 0.8 * swing_loss + 0.2 * dist_loss

            if iter_idx % train_interval == 0:
                train_losses_swing.append(swing_loss)
                train_losses_dist.append(dist_loss)
                train_losses.append(loss)
                
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())

    # Test
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, drop_last=False)

    swing_losses, dist_losses, total_losses, = [], [], []
    model = model.eval()
    idx = 0

    test_out = []
    true_out = []
    test_shot_types = []
    true_shot_types = []

    with torch.no_grad():
        for data in test_loader:
            idx += 1
            x_test, y_test = data
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            outputs = model(x_test.float())
            
            swing_loss = swing_type_loss(outputs[:, :-1], torch.max(y_test[:, :-1], 1)[1])
            # swing_loss = swing_type_loss(outputs[:, :-1], y_test[:, :-1])
            dist_loss =  distance_loss(outputs[:, -1:], y_test[:, -1:])
            total_loss = swing_loss + dist_loss
            
            swing_losses += [swing_loss.cpu().numpy()]
            dist_losses += [dist_loss.cpu().numpy()]
            total_losses += [total_loss.cpu().numpy()]
            
            # print("Test example %d: swing_loss = %f, dist_loss = %f" % (idx, swing_losses[idx-1], dist_losses[idx-1]))
    
            test_out.append(outputs[0])
            true_out.append(y_test[0])
            test_shot_types.append(np.argmax(outputs[:, :-1]).item())
            true_shot_types.append(np.argmax(y_test[:, :-1]).item())
    #         test_shot_dist.append(outputs[:,0].item())
    #         true_shot_dist.append(y_test[:,0].item())
        
        avg_swing_loss = np.mean(swing_losses)
        avg_dist_loss = np.mean(dist_losses)
        avg_total_loss = np.mean(total_losses)
        #print()
        #print("Out of %d test examples: avg_swing_loss = %f, avg_dist_loss = %f, avg_total_loss = %f" % (idx, avg_swing_loss, avg_dist_loss, avg_total_loss))


    accuracy = get_accuracy(test_out, true_out, train_set.y_dist_mean, train_set.y_dist_std)
    MSE = get_MSE(test_out, true_out, train_set.y_dist_mean, train_set.y_dist_std)

    total_acc += accuracy
    total_MSE += MSE
    print("Fold %d: accuracy = %.2f, MSE = %.2f" % (k_i, accuracy, MSE))

print("On average: accuracy = %.2f, MSE = %.2f" % (total_acc/k_fold, total_MSE/k_fold))