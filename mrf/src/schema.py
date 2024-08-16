import json
import numpy as np
class Attribute:
    def __init__(self, name, values,type='nominal'):
        """
        :param name: attriubte name
        :param values: a list of all possilbe values of attribute
        """

        self.name = name # attribute name
        self.values = values # attribute values
        self.size = len(values) # attribute domain size
        self.codes = range(self.size) # numerical codes of the attribute values [0,1,...]
        self.config = dict(zip(self.codes,self.values)) # mapping from numerical codes to the actual attribute values
        self.type = type # nominal or categarical
        
    
    def get(self, key):
        return self.config[key] # return attribute value name
    
    
            
class Schema:
    def __init__(self, attrs):
        """
        :param attrs: a list of Attribute class attributes
        """
        self.attrs = attrs # attributes
        self.attrnames = tuple(attr.name for attr in self.attrs) # tuple of attribute names
        self.shape = tuple(attr.size for attr in self.attrs) # tuple of domain sizes for every attribute
        self.size = np.product(self.shape) # total domain size
        self.n_attrs = len(self.attrs) # number of attributes
        self.config = dict(zip(self.attrnames, self.attrs)) # mapping from attribute names to the corresponding Attribute class
    
    def get(self, key):
        return self.attrs[key] # return attribute info
    
    def project(self, attrs):
        """ project the domain onto a subset of attributes
        
        :param attrs: the attributes to project onto
        :return: the projected Schema object
        """
        # return the projected domain
        if type(attrs) is str:
            attrs = [attrs]
        return Schema([self.config[a] for a in attrs])
    
    def random_feature_subspace(self, size, n_sets=1,random_state=None):
        '''
        randomly sample features of size 'size' without replacement (no duplicate features in a set)
        if n_sets>1 then return a list of Schema objects
        '''

        if random_state is None:
            prng = np.random.mtrand._rand
        else:
            prng = np.random.RandomState(random_state)

        features = self.attrnames[:-1]
        targetname = self.attrnames[-1]
        sets = []
        for _ in range(n_sets):
            per_features = np.array(features)[prng.choice(len(features), size, replace=False)].tolist()
            per_features.append(targetname)
            schema = self.project(per_features)
            print(schema.attrnames, schema.shape)
            sets.append(schema)
        return sets
    
    @staticmethod
    def load(config):
        """ load the Schema object
        
        :param config: json path or dictionary
        :return: Schema object
        """
        if not isinstance(config, dict):
            config = json.load(open(config))
        attrs = [Attribute(attr_info['name'],attr_info['categories'],type='nominal' if attr_info['type']=='categorical' else 'ordinal') for attr_info in config['attributes']]
        return Schema(attrs)