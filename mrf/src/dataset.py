import init
from schema import Schema
import pandas as pd
import json
from pandas.api.types import CategoricalDtype
import numpy as np
from sklearn.model_selection import train_test_split
from ektelo import workload,matrix

class Dataset:

    def __init__(self, df, schema):
        self.schema = schema
        self.df = df.loc[:,self.schema.attrnames]
        self.shape = df.shape
        self.size = df.shape[0]

    def copy(self):
        return Dataset(self.df, self.schema)
    
    def project(self, cols):
        """ project dataset onto a subset of columns """
        if type(cols) in [str, int]:
            cols = [cols]
        data = self.df.loc[:,cols]
        schema = self.schema.project(cols)
        return Dataset(data, schema)
    
    def drop(self, cols, indices=False):
        """drop columns"""
        if indices:
            cols = [self.schema.attrnames[c] for c in cols]
        proj = [c for c in self.schema.attrnames if c not in cols]
        return self.project(proj)
    
    
    def sample(self, size, replace=False):
        
        return Dataset(self.df.sample(n=size, replace=replace), self.schema)
    
    def split(self, size, shuffle=True):
        """patition dataset into n disjoint subsets"""
        df = self.df
        if shuffle:
            df = df.sample(frac=1.)
        dfs = np.array_split(df, size)
        return [Dataset(df, self.schema) for df in dfs]
    
    def train_test_split(self, test_size, random_state = None):
        """Split arrays or matrices into random train and test subsets."""
        dfs = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return Dataset(dfs[0], self.schema),Dataset(dfs[1], self.schema)
    
    @staticmethod
    def load(path, schema, discretize=True):
        """ Load data into a dataset object

        :param path: path to csv file
        :param schema: path to json file encoding the schema information
        """
        df = pd.read_csv(path)
        if discretize:
            config = json.load(open(schema))
            df = discretize_dataframe(df, config)
        schema = Schema.load(schema)
        
        return Dataset(df, schema)
    
    def datavector(self, flatten=True):
        """ return the database in vector-of-counts form """
        bins = [range(n+1) for n in self.schema.shape]
        ans = np.histogramdd(self.df.values, bins)[0]
        return ans.flatten() if flatten else ans
    
    def datamatrix(self):
        """ return the database in matrix-of-counts form - frequency table across feature domain and the target domain
        we assume that the target attribute appears last in the schema information
        """
        X = self.datavector()
        return X.reshape(-1, self.schema.shape[-1])

    def querymatrix(self):
        """return the set of queries in matrix form, each row corresponds to a query
        """
        domain = self.schema.shape
        queries = []
        for i in range(self.shape[0]):
            query = []
            for j in range(self.shape[1]):
                qvec = np.zeros((1,domain[j]), dtype=int)
                qvec[0][self.df.iloc[i][j]] = 1
                query.append(matrix.EkteloMatrix(qvec))
            queries.append(workload.Kronecker(query))
        return workload.VStack(queries)

def discretize_dataframe(df, config):
    """return a discretized version of the original data. 
    How to be discretized must be specified in config

    :param df: original data
    :param config: json file encoding the schema information
    """
    discretized_df = df.copy()  # Create a copy of the original DataFrame
    
    for attribute_info in config["attributes"]:
        attribute_name = attribute_info["name"]
        if attribute_name not in discretized_df.columns:
            continue
        is_numerical = attribute_info["type"] == "numerical"
        
        if is_numerical and "categorical" in attribute_info and attribute_info["categorical"]:
            categories = attribute_info["categories"]
            cutoffs = attribute_info["cutoffs"]
            
            # Apply discretization using cutoffs and numerical codes
            discretized_df[attribute_name] = pd.cut(
                discretized_df[attribute_name], bins=[float('-inf')]+ cutoffs + [float('inf')],
                labels=categories,
            ).cat.codes
        if not is_numerical: # categorical
            categories = attribute_info["categories"]
            if isinstance(categories, dict):
                mapping = {}
                for group, values in categories.items():
                    for value in values:
                        mapping[value] = group
                discretized_df[attribute_name] = discretized_df[attribute_name].replace(mapping)
                categories = categories.keys()
            cat_type = CategoricalDtype(categories=categories, ordered=False)
            discretized_df[attribute_name] = discretized_df[attribute_name].astype(cat_type).cat.codes
    
    # make sure there is no NaN
    assert discretized_df.isnull().values.any() == False
    
    return discretized_df

