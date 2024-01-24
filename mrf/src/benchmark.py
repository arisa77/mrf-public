from dataset import Dataset, discretize_dataframe
from schema import Schema
import pandas as pd
import numpy as np
import json


def tennis_dataset():
    tennis= Dataset.load("../../data/tennis/tennis.csv", "../../data/tennis/tennis_schema.json")
    return tennis

def heart_dataset(drop=False):
    
    heart = Dataset.load("../../data/heart/heart_cleveland_upload.csv","../../data/heart/heart_schema.json")
    if drop:
        #heart = heart.drop(['fbs','restecg']) # drop some columns to reduce the dimensionality
        #heart = heart.drop(['trestbps','fbs','exang','oldpeak','slope','ca','thal','target'])
        heart = heart.project(['age','sex', 'cp', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'])
    return heart

def adult_dataset():

    df = pd.read_csv("../../data/adult/adult.csv")
    config = json.load(open("../../data/adult/adult_schema.json"))
    schema = Schema.load("../../data/adult/adult_schema.json")

    # some preprocessing before instantiating Dataset
    # drop rows with missing values
    df.replace("?", np.nan, inplace=True)
    df = df.dropna(axis=0)

    # drop "fnlwgt" and "educational-num"
    # education-num is duplicate of "education" 
    df = df.drop(columns=["fnlwgt","educational-num"])

    df = discretize_dataframe(df, config)
    dataset = Dataset(df, schema)

    # "relationship" can be derived from martial-status and gender
    # most observations have zero capital-gain and capital-loss.
    # most observations of native country are united-states
    dataset = dataset.drop(["relationship","capital-loss","capital-gain","native-country"])

    return dataset

def car_dataset():
    car= Dataset.load("../../data/car/car.csv", "../../data/car/car_schema.json")
    return car

def tictactoe_dataset():
    data= Dataset.load("../../data/tictactoe/tic-tac-toe.csv", "../../data/tictactoe/tictactoe_schema.json")
    return data

def iris_dataset():
    data= Dataset.load("../../data/iris/iris.csv", "../../data/iris/iris_schema.json")
    return data

def balancescale_dataset():
    data= Dataset.load("../../data/balance-scale/balance-scale.csv", "../../data/balance-scale/balancescale_schema.json")
    return data

def mushroom_dataset():
    data = Dataset.load("../../data/mushroom/mushroom.csv","../../data/mushroom/mushroom.json")
    return data