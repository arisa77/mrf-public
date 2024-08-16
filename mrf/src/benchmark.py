from dataset import Dataset, discretize_dataframe
from schema import Schema
import pandas as pd
import numpy as np
import json
def get_dataset(name):
    if name == 'tennis':
        return tennis_dataset()
    elif name == 'heart':
        return heart_dataset()
    elif name =='adult':
        return adult_dataset()
    elif name=='car':
        return car_dataset()
    elif name=='tictactoe':
        return tictactoe_dataset()
    elif name=='iris':
        return iris_dataset()
    elif name=='balance':
        return balancescale_dataset()
    elif name=='mushroom':
        return mushroom_dataset()
    elif name=='ads':
        return ads_dataset()
    elif name=='adult_binary':
        return adult_binary_dataset()

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

def adult_binary_dataset():
    df = pd.read_csv("../../data/adult-binary/adult-binary.csv")
    
    config = {"attributes":[]}
    # the first four attributes will be dropped
    # all 123 features are binary
    # the last column is the class
    for i in range(0,123):
        config['attributes'].append({
                "name": str(i),
                "type": "categorical",
                "categories": [0,1]
            })
    config['attributes'].append({
        "name": "class",
        "type": "categorical",
        "categories": [-1,1]
        })
    

    df = discretize_dataframe(df,config) # numerical encoding
    schema = Schema.load(config)
    dataset = Dataset(df, schema)
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

def ads_dataset():
    df = pd.read_csv("../../data/internet-ads/ads.csv",header=None)
    df=df.rename(columns={1558: 'class'}) # there are 1559 columns and the last column is the target class
    df = df.drop(columns=[0, 1, 2, 3]) # drop the first 4 columns as there are many missing values

    """
    create a schema:
        the first four attributes will be dropped
        5th-1558th columns are binary
        The last column is the class
    """
    config = {"attributes":[
        {
            "name": i,
            "type": "categorical",
            "categories": [0,1]
        } 
        for i in range(4,1558)
        ]
    }
    config['attributes'].append({
            "name": "class",
            "type": "categorical",
            "categories": ["nonad.","ad."]
        })

    df = discretize_dataframe(df,config) # numerical encoding
    schema = Schema.load(config)
    dataset = Dataset(df, schema)
    return dataset


