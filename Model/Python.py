import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

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

# Adding new col
copyData2['price_per_sqft'] = copyData2['price'] * 100000 / copyData2['total_sqft']
print(copyData2.head())

print(len(copyData2['location'].unique()))

copyData2['location'] = copyData2['location'].apply(lambda x : x.strip())
loc = copyData2.groupby('location')['location'].agg('count').sort_values(ascending = False)
print(loc)

print(len(loc[loc <= 10]))
print(loc[loc <= 10])

locLessThan10 = loc[loc <= 10]
copyData2['location'] = copyData2['location'].apply(lambda x: 'other' if x in locLessThan10 else x)
print(len(copyData2['location'].unique()))

# Outlier removal
print(copyData2[copyData2['total_sqft']/copyData2['bhk'] < 300].head())

print(copyData2.shape)

copyData2 = copyData2[~(copyData2['total_sqft']/copyData2['bhk'] < 300)]
print(copyData2.shape)

print(copyData2['price_per_sqft'].describe())

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key , subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<= (m+st))]
        df_out = pd.concat([df_out,reduced_df], ignore_index = True)
    return df_out

copyData2 = remove_pps_outliers(copyData2)
print(copyData2.shape)

print(copyData2.head())


def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color = 'blue', label = '2 BHK', s = 50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color = 'green', marker = "+", label = '3 BHK', s = 50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price per Square Feet")
    plt.title(location)
    plt.legend()

plot_scatter_chart(copyData2,'Hebbal')


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        for bhk, bhk_df, in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis = 'index')

copyData3 = remove_bhk_outliers(copyData2)
print(copyData3.shape)

plot_scatter_chart(copyData3, 'Hebbal')


matplotlib.rcParams["figure.figsize"] = (20, 10)
plt.hist(copyData3.price_per_sqft, rwidth = 0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()


print(copyData3[copyData3['bath'] > 10].head())

copyData3 = copyData3[~(copyData3['bath'] > 10)]
print(copyData3.shape)


plt.hist(copyData3.bath, rwidth = 0.8)
plt.xlabel("Numbers of bathrooms")
plt.ylabel("Count")
plt.show()


print(copyData3[copyData3['bath'] > copyData3['bhk']+2])

copyData3 = copyData3[copyData3['bath'] < copyData3['bhk'] + 2]
print(copyData3.shape)


copyData3 = copyData3.drop(['size', 'price_per_sqft'], axis='columns')
print(copyData3.head())

Dummies = pd.get_dummies(copyData3['location'])
print(Dummies.head())

Dummies = Dummies.drop('other', axis='columns')

copyData3 = copyData3.drop('location', axis='columns')

finalData = pd.concat([copyData3, Dummies], axis='columns')
print(finalData.head())

finalData = finalData.astype(int)
print(finalData.head())

print(finalData.shape)

x = finalData.drop('price', axis = 'columns')
print(x.head())
print(x.shape)

y = finalData['price']
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 10)

lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)

print(lr_clf.score(x_test, y_test))

print(lr_clf.score(x_train, y_train))


cv = ShuffleSplit(n_splits = 5, test_size = 0.25, random_state = 0)
print(cross_val_score(LinearRegression(), x, y, cv = cv))

def find_best_model_using_gridsearchcv(x, y):
    algos = {
        'linear_regression' : {
            'model' : LinearRegression(),
            'params' : {
                'fit_intercept' : [True, False],
                'positive' : [True, False]
            }
        },
        'lasso' : {
            'model' : Lasso(),
            'params' : {
                'alpha' : [1, 2],
                'selection' :['random', 'cyclic']
            }
        },
        'decision_tree' : {
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' : ['squared_error' , 'friedman_mse'],
                'splitter' : ['best' , 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(x, y)
        scores.append({
            'model' : algo_name,
            'best_score' : gs.best_score_,
            'best_params' : gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

print(find_best_model_using_gridsearchcv(x, y))


def predict_price(location, sqft, bath, bhk):
    # Start with a zero array for input
    x1 = np.zeros(len(x.columns))

    # Assign numerical features
    x1[0] = sqft
    x1[1] = bath
    x1[2] = bhk

    # Handle location column dynamically
    if location in x.columns:
        loc_index = x.columns.get_loc(location)
        x1[loc_index] = 1

    # Predict and make sure the price is not negative
    predicted_price = lr_clf.predict([x1])[0]
    return round(max(predicted_price, 0), 2)


print(predict_price('Indira Nagar', 1000, 2, 2))

print(predict_price('1st Phase JP Nagar',1000, 3, 3))

print(predict_price('1st Phase JP Nagar',1000, 3, 4))


# Extracting model in a pickle file
import pickle
with open('banglore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)

# Extracting locations into json file
import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))