"""This program handles the communication over I2C
between a Raspberry Pi and a MPU-6050 Gyroscope / Accelerometer combo.
Made by: MrTijn/Tijndagamer
Released under the MIT License
Copyright 2015
"""
# https://github.com/nickcoutsos/MPU-6050-Python/blob/master/MPU6050.py

try:
    import smbus
except ModuleNotFoundError:
    print("smbus not found, generating fake data.")
    

class MPU6050:

    # Global Variables
    GRAVITIY_MS2 = 9.80665

    # Scale Modifiers
    ACCEL_SCALE_MODIFIER_2G = 16384.0
    ACCEL_SCALE_MODIFIER_4G = 8192.0
    ACCEL_SCALE_MODIFIER_8G = 4096.0
    ACCEL_SCALE_MODIFIER_16G = 2048.0

    GYRO_SCALE_MODIFIER_250DEG = 131.0
    GYRO_SCALE_MODIFIER_500DEG = 65.5
    GYRO_SCALE_MODIFIER_1000DEG = 32.8
    GYRO_SCALE_MODIFIER_2000DEG = 16.4

    # Pre-defined ranges
    ACCEL_RANGE_2G = 0x00
    ACCEL_RANGE_4G = 0x08
    ACCEL_RANGE_8G = 0x10
    ACCEL_RANGE_16G = 0x18

    GYRO_RANGE_250DEG = 0x00
    GYRO_RANGE_500DEG = 0x08
    GYRO_RANGE_1000DEG = 0x10
    GYRO_RANGE_2000DEG = 0x18

    accel_dict = {
        ACCEL_RANGE_2G: ACCEL_SCALE_MODIFIER_2G,
        ACCEL_RANGE_4G: ACCEL_SCALE_MODIFIER_4G,
        ACCEL_RANGE_8G: ACCEL_SCALE_MODIFIER_8G,
        ACCEL_RANGE_16G: ACCEL_SCALE_MODIFIER_16G
    }

    gyro_dict = {
        GYRO_RANGE_250DEG: GYRO_SCALE_MODIFIER_250DEG,
        GYRO_RANGE_500DEG: GYRO_SCALE_MODIFIER_500DEG,
        GYRO_RANGE_1000DEG: GYRO_SCALE_MODIFIER_1000DEG,
        GYRO_RANGE_2000DEG: GYRO_SCALE_MODIFIER_2000DEG
    }

    # MPU-6050 Registers
    PWR_MGMT_1 = 0x6B
    PWR_MGMT_2 = 0x6C

    ACCEL_XOUT0 = 0x3B
    ACCEL_XOUT1 = 0x3C
    ACCEL_YOUT0 = 0x3D
    ACCEL_YOUT1 = 0x3E
    ACCEL_ZOUT0 = 0x3F
    ACCEL_ZOUT1 = 0x40

    TEMP_OUT0 = 0x41
    TEMP_OUT1 = 0x42

    GYRO_XOUT0 = 0x43
    GYRO_XOUT1 = 0x44
    GYRO_YOUT0 = 0x45
    GYRO_YOUT1 = 0x46
    GYRO_ZOUT0 = 0x47
    GYRO_ZOUT1 = 0x48

    ACCEL_CONFIG = 0x1C
    GYRO_CONFIG = 0x1B

    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    INT_ENABLE = 0x38

    def __init__(self, address):
        self.address = address

        self.bus = smbus.SMBus(1)

        # write to sample rate register
        self.bus.write_byte_data(self.address, self.SMPLRT_DIV, 0)
        
        # Write to power management register
        self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 1)
        
        # Write to Configuration register
        self.bus.write_byte_data(self.address, self.CONFIG, 0)

        # Write to interrupt enable register
        self.bus.write_byte_data(self.address, self.INT_ENABLE, 1)

        self.accel_scale_modifier = MPU6050.ACCEL_SCALE_MODIFIER_2G
        self.gyro_scale_modifier = MPU6050.GYRO_RANGE_250DEG

    # I2C communication methods

    def read_i2c_word(self, register):
        """Read two i2c registers and combine them.

        register -- the first register to read from.
        Returns the combined read results.
        """
        # Read the data from the registers
        high = self.bus.read_byte_data(self.address, register)
        low = self.bus.read_byte_data(self.address, register + 1)

        value = (high << 8) + low

        if (value >= 0x8000):
            return -((65535 - value) + 1)
        else:
            return value

    def get_temp(self):
        """Reads the temperature from the onboard temperature sensor of the MPU-6050.

        Returns the temperature in degrees Celcius.
        """
        # Get the raw data
        raw_temp = self.read_i2c_word(self.TEMP_OUT0)

        # Get the actual temperature using the formule given in the
        # MPU-6050 Register Map and Descriptions revision 4.2, page 30
        actual_temp = (raw_temp / 340) + 36.53

        # Return the temperature
        return actual_temp

    def set_accel_range(self, accel_range):
        """Sets the range of the accelerometer to range.

        accel_range -- the range to set the accelerometer to. Using a
        pre-defined range is advised.
        """
        # First change it to 0x00 to make sure we write the correct value later
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0x00)

        # Write the new range to the ACCEL_CONFIG register
        self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, accel_range)

        self.accel_scale_modifier = self.accel_dict[accel_range]


    def get_accel_data(self, g = False):
        """Gets and returns the X, Y and Z values from the accelerometer.

        If g is True, it will return the data in g
        If g is False, it will return the data in m/s^2
        Returns a dictionary with the measurement results.
        """
        # Read the data from the MPU-6050
        x = self.read_i2c_word(self.ACCEL_XOUT0)
        y = self.read_i2c_word(self.ACCEL_YOUT0)
        z = self.read_i2c_word(self.ACCEL_ZOUT0)

        x = x / self.accel_scale_modifier
        y = y / self.accel_scale_modifier
        z = z / self.accel_scale_modifier

        return x, y, z

    def set_gyro_range(self, gyro_range):
        """Sets the range of the gyroscope to range.

        gyro_range -- the range to set the gyroscope to. Using a pre-defined
        range is advised.
        """
        # First change it to 0x00 to make sure we write the correct value later
        self.bus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)

        # Write the new range to the ACCEL_CONFIG register
        self.bus.write_byte_data(self.address, self.GYRO_CONFIG, gyro_range)

        self.gyro_scale_modifier = self.gyro_dict[gyro_range]

    def get_gyro_data(self):
        """Gets and returns the X, Y and Z values from the gyroscope.

        Returns the read values in a dictionary.
        """
        # Read the raw data from the MPU-6050
        x = self.read_i2c_word(self.GYRO_XOUT0)
        y = self.read_i2c_word(self.GYRO_YOUT0)
        z = self.read_i2c_word(self.GYRO_ZOUT0)

        x = x / self.gyro_scale_modifier
        y = y / self.gyro_scale_modifier
        z = z / self.gyro_scale_modifier

        return x, y, z
        # return {'x': x, 'y': y, 'z': z}

    def get_all_data(self):
        """Reads and returns all the available data."""
        temp = self.get_temp()
        accel = self.get_accel_data()
        gyro = self.get_gyro_data()

        return [accel, gyro, temp]

    def get_movement_data(self):
        ax, ay, az = self.get_accel_data()
        gx, gy, gz = self.get_gyro_data()
        return ax, ay, az, gx, gy, gz

    
if __name__ == "__main__":
    mpu = MPU6050(0x68)
    print(mpu.get_temp())
    accel_data = mpu.get_accel_data()
    print(accel_data[0])
    print(accel_data[1])
    print(accel_data[2])
    gyro_data = mpu.get_gyro_data()
    print(gyro_data[0])
    print(gyro_data[1])
    print(gyro_data[2])
