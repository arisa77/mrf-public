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
    
    @staticmethod
    def load(path):

        config = json.load(open(path))
        attrs = [Attribute(attr_info['name'],attr_info['categories'],type='nominal' if attr_info['type']=='categorical' else 'ordinal') for attr_info in config['attributes']]
        return Schema(attrs)