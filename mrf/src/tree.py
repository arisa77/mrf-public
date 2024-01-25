import pandas as pd
import random
import copy
import numpy as np
import init
from dataset import Dataset
from hdmm import workload, matrix
from mem_matrix import MemMatrix
import itertools
class Tree:
    def __init__(self, schema, depth):
        
        self.schema = schema
        self.attrs = range(schema.n_attrs-1) # a list of attribute codes [0...]
        self.target = schema.n_attrs-1 # target code is n_attrs - 1
        self.targetname = self.schema.get(self.target).name # target name
        self.depth = depth
        self.tree = None
        self.leaf_counts = None # number of instance counts fit into each leaf node
        self.n_leaves = None # number of leaves in a tree

    def fit(self, dataset):
        """ fit the data on the tree 
        must be implemented in a child class
        """
        pass

    def apply(self, dataset):
        """apply the data on the tree to return leaf ids that the data reaches 
        must be implemented in a child class
        """
        # return leave ids
        pass

    def predict(self, dataset, voting='hard'):
        """return predicted results on the dataset
        :param voting: specify a voting mechanism
        """
        pass

    def print_tree(self):
        """print the tree 
        """
        pass


class ID3(Tree):
    def __init__(self, schema, depth):
        super().__init__(schema, depth)
        
    def fit(self, dataset):
        self.leaf_counts = []
        self.tree = self._train(dataset.df, self.attrs, self.depth)
        self.n_leaves = len(self.leaf_counts)
        return self
    
    def _train(self, df: pd.DataFrame, attributes, depth):
        """
        ID3 Decision Tree Algorithm

        
        :params df (pd.DataFrame): The input DataFrame containing the data.
        :params attributes (set): Set of attribute numerical codes to consider for splitting.
        :params depth (int): The maximum depth for the decision tree.

        :return dict: The decision tree structure as a nested dictionary.
        """
        
        if depth == 0 or len(attributes) == 0:
            # get counts for each class
            class_counts = [(df[self.targetname] == label).sum() for label in self.schema.get(self.target).codes]
            self.leaf_counts.append(class_counts)
            return len(self.leaf_counts)-1 # leaf id

        best_attribute = None
        best_score = -1
        for attribute in attributes:
            # Calculate a splitting criterion score (e.g., Information Gain)
            # based on attribute and target distribution
            score = self._calculate_splitting_criterion(df, attribute)
            if score > best_score:
                best_attribute = attribute
                best_score = score
         
        best_attrname = self.schema.get(best_attribute).name
        tree = {best_attrname: {}}
        attributes = list(set(attributes).difference({best_attribute}))
        for value in self.schema.get(best_attribute).codes:
            group_df = df[df[best_attrname] == value]
            subtree = self._train(group_df, attributes, depth - 1)
            tree[best_attrname][value] = subtree

        return tree

    def _calculate_splitting_criterion(self,df, attribute):
        """Implement your attribute selection measure here (e.g., Information Gain)
        and return the score for splitting on this attribute
        """
        
        # MAX OPERATOR
        if df.empty:
            return 0
        attrname = self.schema.get(attribute).name
        return pd.crosstab(df[attrname], df[self.targetname]).max(axis=1).sum()

    def apply(self, dataset):

        if not self.tree:
            raise ValueError('Model not trained yet')
        leaves = []
        for _, sample in dataset.df.iterrows():
            leaf = self._apply_single(self.tree, sample)
            leaves.append(leaf)
        return leaves
    
    def _apply_single(self, tree, sample):
        if not isinstance(tree, dict):
            return tree # return leaf id
        
        (attribute, subtrees), *rest = tree.items()
        
        # Find the subtree corresponding to the sample's attribute value
        sample_value = sample[attribute]
        return self._apply_single(subtrees[sample_value], sample)
    
    def predict(self, dataset, voting='hard'):
        """if weight voting is specified (voting='Weight') return instance counts at leaf nodes
        """
        
        if not self.tree:
            raise ValueError('Model not trained yet')
        
        leaves = self.apply(dataset)

        predicted_values = np.array(self.leaf_counts)[leaves] # shape=(#samples, #classes)
        return np.argmax(predicted_values,axis=1) if voting=='hard' else predicted_values


        
    def print_tree(self):
        self._print_tree(self.tree)
    
    def _print_tree(self, tree, indent=""):
        if not isinstance(tree, dict):
            print(indent+"["+" ".join(str(count) for count in self.leaf_counts[tree])+"]")
        else:
            for attribute, subtree in tree.items():
                print(indent + str(attribute) + ":")
                self._print_tree(subtree, indent + "|  ")

class Node:
    """
    This is meant to used in the following CART class
    """
    def __init__(self, attribute, attr, threshold):
        self.attr = attr
        self.name = attribute.name
        self.threshold = threshold
        self.type = attribute.type
        

class CART(Tree):
    def __init__(self, schema, depth):
        super().__init__(schema, depth)
        
    
    def fit(self, dataset):
        self.leaf_counts = []
        nominal_values = {}
        min_values={}
        max_values={}
        for attribute in self.schema.attrs:
            if attribute.type == 'nominal':
                nominal_values[attribute.name] = attribute.codes
            else:
                min_values[attribute.name] = 0
                max_values[attribute.name] = attribute.size-1
        self.tree = self._train(dataset.df, {key: self.schema.get(key) for key in self.attrs}, nominal_values, min_values, max_values, self.depth)
        assert np.array(self.leaf_counts).sum() == dataset.size
        self.n_leaves = len(self.leaf_counts)
        return self
    
    def _train(self, df: pd.DataFrame, attributes, nominal_values,min_values,max_values, depth):
        """
        CART Decision Tree Algorithm
        """
        
        if depth == 0 or len(attributes) == 0:
            # get counts for each class
            class_counts = [(df[self.targetname] == label).sum() for label in self.schema.get(self.target).codes]
            self.leaf_counts.append(class_counts)
            return len(self.leaf_counts)-1 # leaf id

        best_attribute = None
        best_attribute_key = None
        best_value = None
        best_score = -1
        for key, attribute in attributes.items():
            # Calculate a splitting criterion score (e.g., Information Gain)
            # based on attribute and target distribution
        
            score,value = self._calculate_splitting_criterion(df, attribute, nominal_values,min_values,max_values)
            
            if score > best_score:
                best_attribute = attribute
                best_attribute_key = key
                best_value = value
                best_score = score
        assert best_attribute != None
        if best_value == None:
            best_value = best_attribute.codes[0]
            
        node = Node(best_attribute, best_attribute_key, best_value)
        tree = {node: {}}
        
        left_df, right_df = self._update_dataframe(node, df, min_values, max_values)
        left_attributes, left_nominal_values, right_attributes, right_nominal_values = self._update_attributes(node, attributes, nominal_values, min_values, max_values)
        left_min_values, left_max_values, right_min_values, right_max_values = self._update_minmax_values(node, min_values,max_values)
        
        left_subtree = self._train(left_df,left_attributes, left_nominal_values, left_min_values, left_max_values, depth - 1)
        tree[node]['True'] = left_subtree

        right_subtree = self._train(right_df, right_attributes, right_nominal_values, right_min_values, right_max_values, depth - 1)
        tree[node]['False'] = right_subtree

        return tree

    def _update_dataframe(self, node, df, min_values=None, max_values=None):
        
        if node.type == 'nominal':
            left_df = df[df[node.name] == node.threshold]
            right_df = df[df[node.name] != node.threshold]
        else:
            values = list(range(min_values[node.name], node.threshold+1))
            left_df = df[df[node.name].isin(values)]
            right_df = df[df[node.name].isin(np.arange(node.threshold+1, max_values[node.name]+1))]
        
        return left_df, right_df
    
    def _update_attributes(self, node, attributes, nominal_values, min_values, max_values):
        left_attributes = copy.deepcopy(attributes)
        right_attributes = copy.deepcopy(attributes)

        left_nominal_values = copy.deepcopy(nominal_values)
        right_nominal_values = copy.deepcopy(nominal_values)
        if node.type == 'nominal':
            # for left child, remove attribute and make the corresponding nominal values empty
            del left_attributes[node.attr] 
            left_nominal_values[node.name] = [] 
            # for right child, remove the selected value from nominal values.
            # if the remaining values are only one then remove attribute from the candidates
            if len(nominal_values[node.name])==2:
                del right_attributes[node.attr] 
            right_nominal_values[node.name] = [value for value in nominal_values[node.name] if value!=node.threshold]
        else:
            # if there is only one candidate value assiged to a childe node, 
            # then remove the corresponding attriubte from the candidates
            if min_values[node.name] == node.threshold:
                del left_attributes[node.attr]
            if node.threshold+1 == max_values[node.name]:
                del right_attributes[node.attr]
            
        return left_attributes, left_nominal_values, right_attributes, right_nominal_values
        
    def _update_minmax_values(self, node, min_values, max_values):
        if node.type == 'nominal':
            return min_values, max_values, min_values, max_values
        
        right_min_values = copy.deepcopy(min_values)
        left_max_values = copy.deepcopy(max_values)

        
        left_max_values[node.name] = node.threshold
        right_min_values[node.name] = node.threshold+1

        return min_values, left_max_values, right_min_values, max_values
    
    def _calculate_splitting_criterion(self,df, attribute, nominal_values, min_values, max_values):
        # Implement your attribute selection measure here (e.g., Information Gain)
        # and return the score for splitting on this attribute
        
        # MAX OPERATOR
        if df.empty:
            return 0, None
        
        '''
        
            | C0 |  C1 |  MAX
        --------------------
        Yes |  3 |  2  |  3
        No  |  1 |  1  |  1
        '''
        
        if attribute.type == 'nominal':
            max_score = 0
            max_value = None
            for value in nominal_values[attribute.name]:
                left_df = df[df[attribute.name] == value] # is equal to
                right_df = df[df[attribute.name] != value]# is not equal to
                left_max = max([(left_df[self.targetname] == label).sum() for label in self.schema.get(self.target).codes])
                right_max = max([(right_df[self.targetname] == label).sum() for label in self.schema.get(self.target).codes])
                score = left_max + right_max
                if max_score < score:
                    max_score = score
                    max_value = value
            assert max_value != None

        else:
            max_score = 0
            min_value = min_values[attribute.name]
            max_value = max_values[attribute.name]
            for value in range(min_value,max_value):
                # check spliting point <= value
                # last value is not considered as split value as the right node becomes empty
                left_df = df[df[attribute.name].isin(range(min_value,value+1))]
                right_df = df[df[attribute.name].isin(range(value+1,max_value+1))]
                left_max = max([(left_df[self.targetname] == label).sum() for label in self.schema.get(self.target).codes])
                right_max = max([(right_df[self.targetname] == label).sum() for label in self.schema.get(self.target).codes])
                score = left_max + right_max
                if max_score < score:
                    max_score = score
                    max_value = value
            assert max_value != None

        return  max_score, max_value
    
    def apply(self, dataset):
        '''
        retrun leaf ids
        '''
        if not self.tree:
            raise ValueError('model not trained yet')
        leaves = []
        for _, sample in dataset.df.iterrows():
            leaf = self._apply(self.tree, sample)
            leaves.append(leaf)
        return leaves
    
    def _apply(self, tree, sample):
        if not isinstance(tree, dict):
            return tree # return leaf id        
        (attribute, subtrees), *rest = tree.items()
        
        # Find the subtree corresponding to the sample's attribute value
        sample_value = sample[attribute.name]
        if attribute.type=='nominal':
            if attribute.threshold == sample_value:
                return self._apply(subtrees['True'], sample)
            else:
                return self._apply(subtrees['False'], sample)
        else:
            if attribute.threshold >= sample_value:
                return self._apply(subtrees['True'], sample)
            else:
                return self._apply(subtrees['False'], sample) 
    def predict(self, dataset, voting='hard'):
        '''
        if weight=True then return instance counts at leaf nodes
        '''
        if not self.tree:
            raise ValueError('model not trained yet')
        leaves = self.apply(dataset)
        predicted_values = np.array(self.leaf_counts)[leaves] # shape=(#samples, #classes)
        
        return np.argmax(predicted_values,axis=1) if voting=='hard' else predicted_values
    

    def print_tree(self):
        self._print_tree(self.tree)
    
    def _print_tree(self, tree, indent=""):
        if not isinstance(tree, dict):
            print(indent+"["+" ".join(str(count) for count in self.leaf_counts[tree])+"]")
            #print(indent + tree)  # Leaf node, print the class counts
        else:
            attribute = list(tree.keys())[0]
            if attribute.type=='nominal':
                print(indent + str(attribute.name) + "==" + str(attribute.threshold) + ":")
            else:
                print(indent + str(attribute.name) + "<=" + str(attribute.threshold)+":")
            
            print(indent + "|  " + 'True:')   
            self._print_tree(tree[attribute]['True'], indent + "|  " + "|  ")
            print(indent + "|  " + 'False:')   
            self._print_tree(tree[attribute]['False'], indent + "|  "+ "|  ")


class ExtraTree(Tree):
    """ This is the generic ExtraTree class that supports variants of Extra Tree algorithms
    For now, we have ID3 and CART -like algorithms.
    """
    def __init__(self, schema, depth, random_state = None,alg='ID3'):
        super().__init__(schema, depth)
        if random_state is None:
            self.prng = np.random.mtrand._rand
        else:
            self.prng = np.random.RandomState(random_state)

        self.paths = None
        self.alg= alg
        if alg=='ID3':
            self.estimator = IExtraTree(schema, depth,random_state=random_state)
        elif alg=='CART':
            self.estimator = CExtraTree(schema, depth,random_state=random_state)
        else:
            raise ValueError('specified alg not supported')
    
    @staticmethod
    def get_workload(paths):
        """ Return a decision path matrix where each row encodes a root-to-leaf decision path
        """
        domains = tuple([len(d)for d in paths[0]]) #domain sizes
        # generating np.array for a small array is very time consuming
        # instead of callsing np.array for each path, call np.array once
        flattened_paths = [list(itertools.chain.from_iterable(path)) for path in paths]
        W = np.vstack(flattened_paths) # shape = (#paths, sum of domains)
        cutoffs = list(itertools.accumulate(domains))
        cutoffs.insert(0,0)
        queries = []
        for i in range(W.shape[0]):
            query  = matrix.Kronecker([matrix.EkteloMatrix(W[i:i+1, cutoffs[j]:cutoffs[j+1]]) for j in range(len(cutoffs)-1)])
            queries.append(query)
        return MemMatrix(matrix.VStack(queries))

        
    
    @staticmethod
    def decision_paths(schema, depth, random_state=None, workload=False, tree=False, alg='ID3'):
        """ Return a decision path matrix where each row encodes a root-to-leaf decision path
        """

        if alg=='ID3':
            extratree = IExtraTree(schema,depth,random_state=random_state)
        elif alg=='CART':
            extratree = CExtraTree(schema,depth,random_state=random_state)
        dummy = pd.DataFrame([[0]*schema.n_attrs], columns=schema.attrnames)
        extratree.fit(dummy)

        if workload:
            return ExtraTree.get_workload(extratree.paths)
        else:
            if tree:
                return {"path": extratree.paths, "tree": extratree.tree}
            else:
                return extratree.paths
    def fit(self, dataset):
        return self.estimator.fit(dataset)
    
    def fit_request(self, dataset, decision_tree):
        return self.estimator.fit_request(dataset,decision_tree)
    
    def apply(self, dataset):
        return self.estimator.apply(dataset) 

    def predict(self, dataset, voting='hard'):
        return self.estimator.predict(dataset,voting=voting)

    def print_tree(self):
        return self.estimator.print_tree()
      
class IExtraTree(ID3):
    def __init__(self, schema, depth, random_state = None):
        super().__init__(schema, depth)
        if random_state is None:
            self.prng = np.random.mtrand._rand
        else:
            self.prng = np.random.RandomState(random_state)

        self.paths = None

    def fit(self, dataset):

        self.leaf_counts = []
        self.paths = []
        path = [[1]*n for n in self.schema.shape[:-1]] # exclude the target domain
        df = dataset.df if isinstance(dataset, Dataset) else dataset
        self.tree = self._train(df, self.attrs, self.depth, path)
        self.n_leaves = len(self.leaf_counts)
        return self
    
    def _train(self, df: pd.DataFrame, attributes, depth,  path):
        """
        ID3-like Extra Tree Algorithm
        build a multi-way tree where each attribute node is determined randomly
        """
        
        if depth == 0 or len(attributes) == 0:
            # get counts for each class
            class_counts = [(df[self.targetname] == label).sum() for label in self.schema.get(self.target).codes]
            self.paths.append(copy.deepcopy(path))
            self.leaf_counts.append(class_counts)
            return len(self.leaf_counts)-1 # return the leaf id

        # randomly select attribute
        attr = self.prng.choice(attributes)
        attrname = self.schema.get(attr).name
        tree = {attrname: {}}
        attributes = list(set(attributes).difference({attr}))
        for value in self.schema.get(attr).codes:
            # for the selected attribute, set value i to one and other values to zeros
            path[attr] = [0]*self.schema.get(attr).size
            path[attr][value] = 1 
            group_df = df[df.iloc[:,attr] == value]
            subtree = self._train(group_df,attributes, depth-1, path)
            path[attr] = [1]*self.schema.get(attr).size # reset to all ones
            tree[attrname][value] = subtree
        return tree

    def fit_request(self, dataset, decision_tree):
        '''
        fit data on requested decision tree:
        '''
        self.paths = decision_tree['path']
        self.leaf_counts = []
        df = dataset.df if isinstance(dataset, Dataset) else dataset
        self.tree = self._fit_request(df, decision_tree['tree'])
        self.n_leaves = len(self.leaf_counts)
        return self
    
    def _fit_request(self, df, tree):

        if not isinstance(tree, dict):
            class_counts = [(df[self.targetname] == label).sum() for label in self.schema.get(self.target).codes]
            self.leaf_counts.append(class_counts)
            return tree  # Leaf node
        
        (attr, substructure), *rest = tree.items()
        tree = {attr: {}}
        for value in substructure.keys():
            group_df = df[df[attr] == value]
            subtree = self._fit_request(group_df, substructure[value])
            tree[attr][value] = subtree
        return tree



class CExtraTree(CART):
    def __init__(self, schema, depth, random_state = None):
        super().__init__(schema, depth)
        if random_state is None:
            self.prng = np.random.mtrand._rand
        else:
            self.prng = np.random.RandomState(random_state)

        self.paths = None
        
    
    def fit(self, dataset):
        
        self.leaf_counts = []
        nominal_values = {}
        min_values={}
        max_values={}
        for attribute in self.schema.attrs:
            if attribute.type == 'nominal':
                nominal_values[attribute.name] = attribute.codes
            else:
                min_values[attribute.name] = 0
                max_values[attribute.name] = attribute.size-1
        self.paths = []
        path = [[1]*n for n in self.schema.shape[:-1]] # exclude the target domain
        df = dataset.df if isinstance(dataset, Dataset) else dataset
        self.tree = self._train(df, {key: self.schema.get(key) for key in self.attrs}, nominal_values, min_values, max_values, self.depth, path)
        #assert np.array(self.leaf_counts).sum() == df.shape[0]
        self.n_leaves = len(self.leaf_counts)
        return self
    
    
    def _train(self, df: pd.DataFrame, attributes, nominal_values,min_values,max_values, depth, path):
        """
        CART-like Extra Tree Algorithm
        """
        
        if depth == 0 or len(attributes) == 0:
            # get counts for each class
            class_counts = [(df[self.targetname] == label).sum() for label in self.schema.get(self.target).codes]
            self.paths.append(copy.deepcopy(path))
            self.leaf_counts.append(class_counts)
            return len(self.leaf_counts)-1 # leaf id

        
        # randomly select attribute and threshold
        best_attribute_key = self.prng.choice(list(attributes.keys()))
        best_attribute = attributes[best_attribute_key]
        if best_attribute.type == "nominal":
            best_value = self.prng.choice(nominal_values[best_attribute.name])
        else:
            best_value = self.prng.randint(low=min_values[best_attribute.name], high=max_values[best_attribute.name]) # high is open ended [low, high)
    
        node = Node(best_attribute, best_attribute_key, best_value)
        tree = {node: {}}
        left_df, right_df = self._update_dataframe(node, df, min_values, max_values)
        left_attributes, left_nominal_values, right_attributes, right_nominal_values = self._update_attributes(node, attributes, nominal_values, min_values, max_values)
        left_min_values, left_max_values, right_min_values, right_max_values = self._update_minmax_values(node, min_values,max_values)
        left_path, right_path = self._update_path(node, path, nominal_values, min_values, max_values)
        
        left_subtree = self._train(left_df,left_attributes, left_nominal_values, left_min_values, left_max_values, depth - 1,left_path)
        tree[node]['True'] = left_subtree

        right_subtree = self._train(right_df, right_attributes, right_nominal_values, right_min_values, right_max_values, depth - 1,right_path)
        tree[node]['False'] = right_subtree

        return tree

    
    def _update_path(self, node, path, nominal_values, min_values, max_values):
        left_path = copy.deepcopy(path)
        right_path = copy.deepcopy(path)

        left_path[node.attr] = [0]*len(left_path[node.attr])
        right_path[node.attr] = [0]*len(right_path[node.attr])
        if node.type == 'nominal':
            # for the selected attribute, set value i to one and other values to zeros
            left_path[node.attr][node.threshold] = 1
            
            # set remaining values to 1
            for value in nominal_values[node.name]:
                if value != node.threshold:
                    right_path[node.attr][value] = 1
        else:
            for value in range(min_values[node.name], node.threshold+1):
                left_path[node.attr][value] = 1
            
            for value in range(node.threshold+1, max_values[node.name]+1):
                right_path[node.attr][value] = 1
        
        return left_path, right_path
    
    def fit_request(self, dataset, decision_tree):
        self.paths = decision_tree['path']
        self.leaf_counts = []
        df = dataset.df if isinstance(dataset, Dataset) else dataset
        min_values={attribute.name:0 for attribute in self.schema.attrs if attribute.type=='ordinal'}
        max_values={attribute.name:attribute.size-1 for attribute in self.schema.attrs if attribute.type=='ordinal'}
        self.tree = self._fit_request(df, decision_tree['tree'],min_values,max_values)
        self.n_leaves = len(self.leaf_counts)
        return self
    
    def _fit_request(self, df, tree, min_values, max_values):

        if not isinstance(tree, dict):
            class_counts = [(df[self.targetname] == label).sum() for label in self.schema.get(self.target).codes]
            self.leaf_counts.append(class_counts)
            return tree  # Leaf node
        
        (attribute, subtrees), *rest = tree.items()
        tree = {attribute: {}}
        

        left_df, right_df = self._update_dataframe(attribute, df, min_values, max_values)
        left_min_values, left_max_values, right_min_values, right_max_values = self._update_minmax_values(attribute, min_values,max_values)
        
        left_subtree = self._fit_request(left_df, subtrees['True'], left_min_values, left_max_values)
        tree[attribute]['True'] = left_subtree

        right_subtree = self._fit_request(right_df, subtrees['False'], right_min_values, right_max_values)
        tree[attribute]['False'] = right_subtree

        return tree