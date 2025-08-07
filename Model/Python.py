import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# For full view in pycharm
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

RealData = pd.read_csv("Dataset/Bengaluru_House_Data.csv")
print(RealData.head())

print(RealData.shape)  # 13320, 9
print(RealData.info())

