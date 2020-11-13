'''
        Read Gyro and Accelerometer by Interfacing Raspberry Pi with MPU6050 using Python
    http://www.electronicwings.com
'''

from MPU6050 import MPU6050

try:
    import smbus
except ModuleNotFoundError:
    print("smbus not found, generating fake data.")
    

import time
import numpy as np
import matplotlib.pyplot as plt

from pi_utils import *


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

HIT_THRESH = 4
WINDOW_SIZE = 50

def save_swing(data_matrix, columns):
    if input("\nSave run? y/[n]: ").lower() == "y":
        shot_type = input(HELP_STRING + "\nShot type?: ")
        distance = input("Distance (yards)?: ")
        save_to_csv(data_matrix, columns, shot_type, distance)
    else:
        print("Not saved.")

def main():
    try:
        mpu = MPU6050(0x68)
        mpu.set_accel_range(MPU6050.ACCEL_RANGE_8G)
        mpu.set_gyro_range(MPU6050.GYRO_RANGE_500DEG)
        data_func = mpu.get_movement_data

    except NameError as ex:
        print("#############################")
        print("##### MPU6050 NOT FOUND #####")
        print("#####  USING FAKE DATA  #####")
        print("#############################")
        data_func = get_fake_data

    live = True
    verbose = False
    print_interval = 1
    draw_interval = 1 # how many collections between plots
    # plt.ion()

    columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
    data = []
    detect_swing = False
    try:
        print("Starting collection.")
        count = 0
        start = time.time()
        
        while True:

            # ax, ay, az, gx, gy, gz
            values = data_func()            
            data.append(values)
            
            if live and count % draw_interval == 0:
                plot_data(data, columns)

            if verbose and count % print_interval == 0:
                print_values(values, count)
            
            # if we've hit the max window size and there is no swing detected, discard first entry
            # if there is a swing detected and our list is full, then we've collected the full sample and can save
            if len(data) >= WINDOW_SIZE:
                if detect_swing:
                    # time to save
                    print("Saving...")
                    data_matrix = np.array(data)
                    save_swing(data_matrix, columns)
                    print("Continuing...")
                    detect_swing = False
                    data = []
                    count = 0  # TODO: REMOVE probably
                else:
                    data.pop(0)            
            
            if abs(values[0]) >= HIT_THRESH:
                print("swing detected")
                 # detected a swing, during the midpoint (probably)
                detect_swing = True 
                
                # empty the first half of the data list, allow recording for the second half of the swing signal
                data = data[-WINDOW_SIZE//2:]                 

            count += 1

    except KeyboardInterrupt:
        print("Stopping collection.")
        pass

    runtime = time.time() - start
    print("Sample rate (Hz):", count/runtime)
    
#     plot_data(data, columns)
#     plt.show()
    
    
def plot_data(data, columns):
    data_matrix = np.array(data).T
    for i in range(1):
        plt.subplot(2,3,i+1)
        plt.title(columns[i])
        plt.plot(data_matrix[i])
    plt.draw()
    plt.pause(0.0000001)
    plt.clf()

    
if __name__ == "__main__":
    main()
