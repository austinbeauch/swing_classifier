'''
        Read Gyro and Accelerometer by Interfacing Raspberry Pi with MPU6050 using Python
    http://www.electronicwings.com
'''

from MPU6050 import MPU6050

try:
    import smbus
except ModuleNotFoundError:
    print("smbus not found, generating fake data.")
    pass

import time
import numpy as np
import matplotlib.pyplot as plt

from pi_utils import *

#some MPU6050 Registers and their Address

HELP_STRING = """
0 - Pull-hook  (left->left)
1 - Hook       (right->left)
2 - Pull       (left)
3 - Fade       (left->middle)
4 - Straight
5 - Draw       (right->middle)
6 - Push       (left)
7 - Slice      (left->right)
8 - Push-slice (left->left)
"""

def main():
    try:
        mpu = MPU6050(0x68)
        mpu.set_accel_range(MPU6050.ACCEL_RANGE_8G)
        mpu.set_gyro_range(MPU6050.GYRO_RANGE_500DEG)
        data_func = mpu.get_movement_data

    except NameError as ex:
        raise ex
        print("Using fake data")
        data_func = get_fake_data

    live = False
    verbose = False
    print_interval = 1
    draw_interval = 25 # how many collections between plots
    draw_window = 1000 # window size to plot
    # plt.ion()

    columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
    data = []

    try:
        print("Starting collection.")
        count = 0
        start = time.time()
        
        while True:

            values = data_func()

            data.append(values)

            if live and count % draw_interval == 0:
                plot_data(data)

            if verbose and count % print_interval == 0:
                print_values(values, count)
            
            count += 1

    except KeyboardInterrupt:
        print("Stopping collection.")
        pass

    runtime = time.time() - start
    print(runtime)
    data_matrix = np.array(data)
    print(data_matrix.shape)
    print(count/runtime)

    plot_data(data, columns)
    plt.show()
    
    if input("\nSave run? y/[n]: ").lower() == "y":

        shot_type = input(HELP_STRING + "\nShot type?: ")
        distance = input("Distance (yards)?: ")
        save_to_csv(data_matrix, columns, shot_type, distance)
    else:
        print("Not saved.")
    
    
def plot_data(data, columns):
    data_matrix = np.array(data).T
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.title(columns[i])
        # plt.plot(data_matrix[i][-draw_window:])
        plt.plot(data_matrix[i][:])
    #plt.draw()
    #plt.pause(0.0000001)
    #plt.clf()

if __name__ == "__main__":
    main()
