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

print(RealData.groupby('area_type')['area_type'].agg('count'))

# Dropping unnecessary columns
copyData = RealData.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
print(copyData.head())

print(copyData.isnull().sum())

copyData2 = copyData.dropna()
print(copyData2.isnull().sum())
print(copyData2.shape)

print(copyData2['size'].unique())

copyData2['bhk'] = copyData2['size'].apply(lambda x: int(x.split(' ')[0]))
print(copyData2.head())

print(copyData2['bhk'].unique())

print(copyData2[copyData2['bhk'] > 20])   # 2 rows

print(copyData2.total_sqft.unique())

def is_float(x) :
    try:
        float(x)
    except:
        return False
    return True

print(copyData2[~copyData2['total_sqft'].apply(is_float)].head(15))

# Convert sqft to num
def convert_sqft_into_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try :
        return float(x)
    except:
        return None

copyData2['total_sqft'] = copyData2['total_sqft'].apply(convert_sqft_into_num)
print(copyData2.head(10))



