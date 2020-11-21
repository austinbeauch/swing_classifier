import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_swing(swing_data, shot_type=None, dist=None):
    """
    swing_data: Dx6 array of IMU data
    """
    
    columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
    plt.figure(figsize=(15,7))
    for idx, line in enumerate(swing_data):
        plt.subplot(2,3,idx+1)
        plt.title(columns[idx])
        plt.plot(line)
    if shot_type is not None:
        plt.suptitle(f"{shot_type}, {dist}yds")

        
def load_data(path):
    shot_types = 9
    other_metrics = 1
    total_metics = 10
    X_data = []
    y_data = [] 
    # for i in range(100):
    for csv in glob.glob(path + "*.csv"):
        x = pd.read_csv(csv).drop(columns="Unnamed: 0")
        y = np.zeros(total_metics)
        y[x["shot_type"][0]] = 1
        y[-1] = x["distance"][0]
        x_values = x.values[:, :-2].T
        X_data.append(x_values)
        y_data.append(y)

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    return X_data, y_data

def plot_counts(y):
    pd.Series(np.argmax(y[:, :-1],axis=1)).value_counts(sort=False).plot.bar().set(ylabel="Count")