import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_location_names():
    return __locations  # returns a list of location strings

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    # Create a zero input array the same length as the model's expected features
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if loc_index >= 0:
        x[loc_index] = 1

    # Predict and return price rounded to 2 decimal places
    return round(__model.predict([x])[0], 2)

def load_saved_artifacts():
    print("📦 Loading saved artifacts...")

    global __data_columns
    global __locations
    global __model

    # Load column names
    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # skip sqft, bath, bhk

    # Load trained model
    with open("./artifacts/banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)

    print("✅ Artifacts loaded successfully!")
