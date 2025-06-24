import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

#import data 
#every pitch Bryan Woo has thrown this season
data = pd.read_csv('693433_data.csv')

#delete irrelevant columns
columns_to_keep = [
    "release_speed",
    "release_pos_x",
    "release_pos_z",
    "launch_speed",
    "effective_speed",
    "release_spin_rate",
    "release_extension",
    "pitch_name"
]

#Overwrite DataFrame with only selected columns
data = data[columns_to_keep] 

#delete missing values
data = data[data['launch_speed'].notna()].copy()

data.to_csv('cleaned_woo.csv', index=False)
