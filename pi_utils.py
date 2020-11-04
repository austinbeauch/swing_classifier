from datetime import datetime

import numpy as np
import pandas as pd

PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47


def MPU_Init(bus, dev_address=0x68):
    #write to sample rate register
    bus.write_byte_data(dev_address, SMPLRT_DIV, 7)
    
    #Write to power management register
    bus.write_byte_data(dev_address, PWR_MGMT_1, 1)
    
    #Write to Configuration register
    bus.write_byte_data(dev_address, CONFIG, 0)
    
    #Write to Gyro configuration register
    bus.write_byte_data(dev_address, GYRO_CONFIG, 24)
    
    #Write to interrupt enable register
    bus.write_byte_data(dev_address, INT_ENABLE, 1)


def read_raw_data(bus, addr, dev_address=0x68):
    #Accelero and Gyro value are 16-bit
        high = bus.read_byte_data(dev_address, addr)
        low = bus.read_byte_data(dev_address, addr+1)
    
        #concatenate higher and lower value
        value = ((high << 8) | low)
        
        #to get signed value from mpu6050
        if(value > 32768):
                value = value - 65536
        return value

def get_pi_data(bus):
    #Read Accelerometer raw value
    acc_x = read_raw_data(bus, ACCEL_XOUT_H)
    acc_y = read_raw_data(bus, ACCEL_YOUT_H)
    acc_z = read_raw_data(bus, ACCEL_ZOUT_H)

    #Read Gyroscope raw value
    gyro_x = read_raw_data(bus, GYRO_XOUT_H)
    gyro_y = read_raw_data(bus, GYRO_YOUT_H)
    gyro_z = read_raw_data(bus, GYRO_ZOUT_H)

    #Full scale range +/- 250 degree/C as per sensitivity scale factor
    Ax = acc_x/16384.0
    Ay = acc_y/16384.0
    Az = acc_z/16384.0

    Gx = gyro_x/131.0
    Gy = gyro_y/131.0
    Gz = gyro_z/131.0

    return [Ax, Ay, Az, Gx, Gy, Gz]

def get_fake_data():
    return list(np.random.random(6))

def print_values(vals, count):
    Ax, Ay, Az, Gx, Gy, Gz = vals 
    print ("Gx=%.2f" % Gx, 
        u'\u00b0'+ "/s", 
        "\tGy=%.2f" % Gy, 
        u'\u00b0'+ "/s", 
        "\tGz=%.2f" % Gz, 
        u'\u00b0'+ "/s", 
        "\tAx=%.2f g" % Ax, 
        "\tAy=%.2f g" % Ay, 
        "\tAz=%.2f g" % Az, 
        "iter: %d" % count)     

def save_to_csv(data, columns):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S_%d_%m_%y")
    filename = "data/" + current_time + ".csv"
    pd.DataFrame(data, columns=columns).to_csv(filename)
    print("Saved", filename)
