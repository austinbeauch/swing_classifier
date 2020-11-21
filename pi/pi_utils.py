from datetime import datetime

import numpy as np
import pandas as pd


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

    
def save_to_csv(data, columns, shot_type, distance):
    now = datetime.now()
    current_time = now.strftime("%d_%m_%H:%M:%S")
    filename = f"data/{current_time}.csv"
    df = pd.DataFrame(data, columns=columns)
    df["distance"] = distance
    df["shot_type"] = shot_type
    df.to_csv(filename)
    print("Saved", filename)
