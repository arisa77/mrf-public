from tree import ExtraTree
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import init
from hdmm import matrix
from mem_matrix import MemMatrix
class EnsembleTrees:
    def __init__(self, schema, depth, n_estimators, bootstrap=False, disjoint=False, random_state=None, n_jobs=None):
        self.schema = schema
        self.depth = depth
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.disjoint = disjoint
        self.n_jobs = n_jobs
        
        if random_state is None:
            self.prng = np.random.mtrand._rand
        else:
            self.prng = np.random.RandomState(random_state)

    def _prepare_dataset(self, dataset):
        
        if self.disjoint:
            datasets = dataset.split(self.n_estimators, shuffle=True)
        else:
            if self.bootstrap:
                n = dataset.shape[0]
                datasets = [dataset.sample(n, replace=True) for _ in range(self.n_estimators)]
            else:
                datasets = [dataset.copy() for _ in range(self.n_estimators)]
        
        return datasets

    def predict(self, dataset, voting='hard', return_votes=False):
        '''
        weight=True if wegith voting
        return_votes=True if returning aggregated votes across all class labels
        '''
        if self.n_jobs:
            # parallel processing
            preds = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                delayed(estimator.predict)(dataset, voting=voting)
                for estimator in self.estimators
            )
        else:
            preds = [estimator.predict(dataset, voting=voting) for estimator in self.estimators]

        if voting == 'weight':
            preds = np.array(preds).transpose(1,0,2) # shape=(#samples, #estimators, #classes)
        else:
            preds = np.array(preds).T  # transpose row and column to make each row corresponds to each data sample
            

        # majority voting
        if voting == 'weight':
            if return_votes==False:
                return np.sum(preds, axis=1).argmax(axis=1) # aggregating across estimators and than return the majority label (#samples)
            else:
                return np.sum(preds,axis=1) # aggregating across estimators, returning a matrix of shape (#samples, #classes)
        else:
            target = self.estimators[0].target
            size = self.schema.get(target).size
            if return_votes==False:
                return np.array([np.bincount(pred,minlength=size).argmax() for pred in preds]) # return the majority label
            else:
                return np.array([np.bincount(pred,minlength=size) for pred in preds]) # (#samples, #classes)
    
    def apply(self, dataset):
        '''
        return leaf ids
        '''
        if self.n_jobs:
            # parallel processing
            preds = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                delayed(estimator.apply)(dataset)
                for estimator in self.estimators
            )
        else:
            preds = [estimator.apply(dataset) for estimator in self.estimators]

        return np.array(preds) # shape=(#estimators, #samples)
    
    def print_forest(self):
        for i, estimator in enumerate(self.estimators):
            print("%d estimator"%(i))
            estimator.print_tree()

                   
            

class ExtraTrees(EnsembleTrees):
    def __init__(self, schema, depth, n_estimators, bootstrap=False, disjoint=False, random_state= None,n_jobs=None,alg='ID3'):
        self.estimators = []
        for i in range(n_estimators):
            random_state = random_state+i if random_state else None
            self.estimators.append(ExtraTree(schema, depth, random_state=random_state,alg=alg))
        random_state = random_state+1 if random_state else None
        super().__init__(schema, depth, n_estimators, bootstrap=bootstrap, disjoint=disjoint,random_state=random_state,n_jobs=n_jobs)

    def fit(self, datasets):

        if not isinstance(datasets, list):
            datasets = self._prepare_dataset(datasets)
        if self.n_jobs:
            # parallel processing
            self.estimators = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                delayed(estimator.fit)(dataset) 
                for dataset, estimator in zip(datasets, self.estimators)
            )
        else:
            for dataset, estimator in zip(datasets, self.estimators):
                estimator.fit(dataset)
        

    def fit_request(self, datasets, decision_trees):
        assert len(decision_trees) == self.n_estimators

        if not isinstance(datasets, list):
            datasets = self._prepare_dataset(datasets)

        if self.n_jobs:
            self.estimators = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                delayed(self.estimators[i].fit_request)(datasets[i], decision_trees[i]) 
                for i in range(self.n_estimators)
            )
        else:
            for i in range(self.n_estimators):
                self.estimators[i]= self.estimators[i].fit_request(datasets[i], decision_trees[i])
        

    def get_forest(self):
        return [estimator.tree for estimator in self.estimators]
    
    def get_paths(self):
        paths = []
        for i in range(self.n_estimators):
            paths.append(self.estimators[i].paths)
        return paths
    
    def get_leaf_counts(self):
        return np.array([count for estimator in self.estimators
                 for count in estimator.leaf_counts])
    
    def update_leaf_counts(self, counts):
        start = 0
        for estimator in self.estimators:
            estimator.leaf_counts = counts[start:start+estimator.n_leaves]
            start+=estimator.n_leaves
    
    def update_forest(self, forest):
        if self.n_jobs:
            # parallel processing
            self.estimators = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                delayed(estimator.update_tree)(forest[i])
                for i, estimator in enumerate(self.estimators)
            )
        else:
            self.estimators = [estimator.update_tree(forest[i]) for i, estimator in enumerate(self.estimators)]


    def inference_workload(self, samples):
        
        leaves = self.apply(samples)
        
        n_leaves = [estimator.n_leaves for estimator in self.estimators]
        cum_leaves = n_leaves[0]
        
        for i in range(1,leaves.shape[0]):
            leaves[i]= leaves[i] + cum_leaves
            cum_leaves += n_leaves[i]
        leaves = leaves.T
        W = np.zeros((leaves.shape[0],sum(n_leaves)))
        W[np.arange(W.shape[0])[:,None],leaves] =1

        return MemMatrix(matrix.EkteloMatrix(W))
    
    @staticmethod
    def decision_paths(schema, depth, n_estimators, random_state = None, n_jobs = None, workload=False, tree=False,alg='ID3'):
        results = []
        
        if n_jobs:
            results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(ExtraTree.decision_paths)(schema, depth, random_state=random_state + i if random_state else None, tree=tree,alg=alg) 
                for i in range(n_estimators)
            )
            if workload:
                return ExtraTrees.get_workload(results)
            return results
        else:
            results = [ExtraTree.decision_paths(schema, depth, random_state=random_state + i if random_state else None, tree=tree,alg=alg) 
                for i in range(n_estimators)]
            if workload:
                return ExtraTrees.get_workload(results)
            return results  
    @staticmethod
    def get_workload(paths):
        return ExtraTree.get_workload([path for subpaths in paths for path in subpaths])
    


class VerticalExtraTrees:
    def __init__(self, schema, depth, n_estimators, n_ensembles=1, max_nfeatures=10, bootstrap=False, disjoint=False, random_state=None, n_jobs=None,alg='ID3'):
        if random_state is None:
            self.prng = np.random.mtrand._rand
        else:
            self.prng = np.random.RandomState(random_state)
        
        self.schema = schema
        self.depth = depth
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.disjoint = disjoint
        self.n_jobs = n_jobs

        self.n_ensembles=n_ensembles
        
        assert len(schema) == n_ensembles
        self.ensembles = []
        for i in range(n_ensembles):
            ensemble = ExtraTrees(schema[i], depth, n_estimators, bootstrap=bootstrap, disjoint=disjoint, random_state=random_state, n_jobs=n_jobs,alg=alg)
            self.ensembles.append(ensemble)


    def fit(self, datasets):
        if self.n_jobs:
            Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(ensemble.fit)(datasets.project(ensemble.schema.attrnames))
                    for ensemble in self.ensembles
                )
        else:
            for ensemble in self.ensembles:
                features = ensemble.schema.attrnames
                per_datasets = datasets.project(features)
                ensemble.fit(per_datasets)    
        

    def fit_request(self, datasets, decision_trees):
        assert len(decision_trees) == self.n_ensembles

        if self.n_jobs:
            Parallel(n_jobs=self.n_jobs, prefer='threads')(
                        delayed(ensemble.fit_request)(datasets.project(ensemble.schema.attrnames), decision_trees[i])
                        for i, ensemble in enumerate(self.ensembles)
                    )
        else:
            for decision_trees_per_ensemble, ensemble in zip(decision_trees, self.ensembles):
                features = ensemble.schema.attrnames
                per_datasets = datasets.project(features)
                ensemble.fit_request(per_datasets, decision_trees_per_ensemble) 

    def get_forest(self):
        if self.n_jobs:
            return Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(ensemble.get_forest)()
                    for ensemble in self.ensembles
                )
        else:
            return [ensemble.get_forest() for ensemble in self.ensembles]
    
    def get_paths(self):
        if self.n_jobs:
            return Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(ensemble.get_paths)()
                    for ensemble in self.ensembles
                )
        else:
            return [ensemble.get_paths() for ensemble in self.ensembles]
    
    def get_leaf_counts(self):
        if self.n_jobs:
            return Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(ensemble.get_leaf_counts)()
                    for ensemble in self.ensembles
                )
        else:
            return [ensemble.get_leaf_counts() for ensemble in self.ensembles]
    
    def update_leaf_counts(self, counts):
        assert len(counts) == self.n_ensembles
        if self.n_jobs:
            Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(ensemble.update_leaf_counts)(counts[i])
                    for i, ensemble in enumerate(self.ensembles)
                )
        else:
            for counts_per_ensemble, ensemble in zip(counts, self.ensembles):
                ensemble.update_leaf_counts(counts_per_ensemble)

    def update_forest(self, forest):
        if self.n_jobs:
            Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(ensemble.update_forest)(forest[i])
                    for i, ensemble in enumerate(self.ensembles)
                )
        else:
            for forest_per_ensemble, ensemble in zip(forest, self.ensembles):
                ensemble.update_forest(forest_per_ensemble
                                    )
    
    @staticmethod
    def decision_paths(schema, depth, n_estimators, n_ensembles, random_state = None, n_jobs = None, workload=False, tree=False,alg='ID3'):
    
        # schema should be a list of schema objects
        assert len(schema) == n_ensembles

        if n_jobs:
            return Parallel(n_jobs=n_jobs, prefer='threads')(
                    delayed(ExtraTrees.decision_paths)(per_schema, depth, n_estimators, random_state=random_state, n_jobs=n_jobs,workload=workload,tree=tree,alg=alg)
                    for per_schema in schema
                )
        else:
            results = []
            for per_schema in schema:
                result = ExtraTrees.decision_paths(per_schema, depth, n_estimators, random_state=random_state, n_jobs=n_jobs,workload=workload,tree=tree,alg=alg)
                results.append(result)
            return results


    @staticmethod
    def get_workload(paths, n_jobs=None):
        
        # len(paths) == n_ensembles
        if n_jobs:
            return Parallel(n_jobs=n_jobs, prefer='threads')(
                    delayed(ExtraTree.get_workload)([path for subpaths in paths_per_ensemble for path in subpaths])
                    for paths_per_ensemble in paths
                )
        else:
            Ws =[]
            for paths_per_ensemble in paths:
                W = ExtraTree.get_workload([path for subpaths in paths_per_ensemble for path in subpaths])
                Ws.append(W)
            return Ws
    

    def inference_workload(self, samples):
        if self.n_jobs:
            return Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(ensemble.inference_workload)(samples.project(ensemble.schema.attrnames))
                    for ensemble in self.ensembles
                )
        else:
            W = []
            for ensemble in self.ensembles:
                features = ensemble.schema.attrnames
                per_dataset = samples.project(features)
                W.append(ensemble.inference_workload(per_dataset))
            return W

    def predict(self, dataset, voting='hard', return_votes=False):
        '''
        weight=True if wegith voting
        return_votes=True if returning aggregated votes across all class labels
        '''
            
        if return_votes:
            if self.n_jobs:
            # parallel processing
                votes_ensembles = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(ensemble.predict)(dataset.project(ensemble.schema.attrnames),voting=voting,return_votes=True)
                    for ensemble in self.ensembles
                )
            else:
                votes_ensembles = []
                for ensemble in self.ensembles:
                    features = ensemble.schema.attrnames
                    per_dataset = dataset.project(features)
                    votes = ensemble.predict(per_dataset,voting=voting,return_votes=True)
                    votes_ensembles.append(votes)
            return votes_ensembles
        else:
            majority_votes = np.zeros((dataset.shape[0],self.schema[0].shape[-1]))
            if self.n_jobs:
                votes = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(ensemble.predict)(dataset.project(ensemble.schema.attrnames),voting=voting,return_votes=True)
                    for ensemble in self.ensembles
                )
                for votes_per_ensemble in votes:
                    majority_votes += votes_per_ensemble
            else:
                for ensemble in self.ensembles:
                    features = ensemble.schema.attrnames
                    per_dataset = dataset.project(features)
                    votes = ensemble.predict(per_dataset,voting=voting,return_votes=True)
                    
                    majority_votes+=votes
    
            return np.argmax(majority_votes,axis=1)

    
    def apply(self, dataset):
        '''
        return leaf ids
        '''
        if self.n_jobs:
            return Parallel(n_jobs=self.n_jobs, prefer='threads')(
                    delayed(ensemble.apply)(dataset.project(ensemble.schema.attrnames))
                    for ensemble in self.ensembles
                )
        else:
            leaf_ids = []
            for ensemble in self.ensembles:
                features = ensemble.schema.attrnames
                per_dataset = dataset.project(features)
                leaf_ids.append(ensemble.apply(per_dataset))
            return leaf_ids
    
    def print_forest(self):
        for i, ensemble in enumerate(self.ensembles):
            print("%d ensemble"%(i))
            ensemble.print_forest()