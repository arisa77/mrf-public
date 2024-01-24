
import init
from ektelo import workload, matrix
from hdmm import templates
import numpy as np
import sys
sys.path.append("../src")
from random_forest import ExtraTrees, VerticalExtraTrees
from mem_matrix import MemMatrix

class  PredictMechanismBase:
    """
    DP prediction methods
    """
    def __init__(self, schema, trained_model, voting=None, inference_query=None, strategy_query=None):
        
        self.schema = schema
        self.trained_model = trained_model
        self.n_estimators = trained_model.n_estimators
        self.voting =voting
        self.eps_per_query = None
        self.inference_query = inference_query # must be matrix
        self.strategy = strategy_query

    
    @staticmethod
    def optimize(W, p=1):
        # the optimization may not help and 
        # the performance might be the same as that of Identity
        if isinstance(W, list):
            A = []
            for Wi in W:
                if isinstance(Wi, MemMatrix):
                    Wi = matrix.EkteloMatrix(Wi.explicit_matrix().dot(np.eye(Wi.shape[1])))
                pid = templates.PIdentity(p, Wi.shape[1])
                pid.optimize(Wi)
                Ai = pid.strategy()
                A.append(MemMatrix(Ai))
        else:
            if isinstance(W, MemMatrix):
                W = W.explicit_matrix()
            pid = templates.PIdentity(p, W.shape[1])
            pid.optimize(W)
            A = pid.strategy()
            
            return MemMatrix(A)
    
    def predict(self, samples, eps, optimize=True, random_state=None):

        if random_state is None:
            prng = np.random.mtrand._rand
        else:
            prng = np.random.RandomState(random_state)

        self.eps_per_query = eps
        votes = self.trained_model.predict(samples, voting=self.voting, return_votes=True) #(samples, classes)
        if optimize==False:
            # LM
            # split the budget equally among test queries
            if isinstance(self.trained_model, VerticalExtraTrees):
                ## TODO
                raise ValueError('not supported yet')
            delta = votes.shape[0]
            noise = prng.laplace(loc=0.0, scale=delta/self.eps_per_query, size=(votes.shape[0],votes.shape[1]))
            Y = votes + noise
        else:
            W = self.inference_query
            A = self.strategy 
            if isinstance(self.trained_model, VerticalExtraTrees):
                """for each ensemble, aggregate votes across trees
                add noise to the resulting votes for every class
                return the majority class that has the largest noisy votes across the ensembles
                """
                Ys = []
                for W_i, A_i, votes_i in zip(W, A, votes):
                    A1_i = MemMatrix(A_i.explicit_matrix().pinv())
                    delta = A_i.explicit_matrix().sensitivity()
                    noise = MemMatrix.laplace_noise(prng=prng, scale=delta/self.eps_per_query, size=(A_i.shape[0],votes_i.shape[1]))
                    Y = votes_i + np.array(W_i.dot(A1_i).dot(noise).explicit_matrix())
                    Ys.append(Y)
                    
                #Ys = np.hstack(Ys) # horizontally stack the votes across the ensembles
                #n_class = votes[0].shape[1]
                #return np.argmax(Y,axis=1)%n_class, Y
            
                return np.stack(Ys).sum(axis=0).argmax(axis=1),Ys

            else:  
                A1 = MemMatrix(A.explicit_matrix().pinv())
                delta = A.explicit_matrix().sensitivity()
                noise = MemMatrix.laplace_noise(prng=prng, scale=delta/self.eps_per_query, size=(A.shape[0],votes.shape[1]))
                Y = votes + np.array(W.dot(A1).dot(noise).explicit_matrix())

                return np.argmax(Y,axis=1), Y 


class SubsampleAndAggregate(PredictMechanismBase):
    """ Subsample-and-aggregate framework
    """
    def __init__(self, schema, trained_model, voting='hard', inference_query=None,strategy_query=None):
        super().__init__(schema, trained_model, voting=voting, inference_query=inference_query,strategy_query=strategy_query)
        assert trained_model.disjoint ==True
        
                
        # class instance counts
        self.data = self.trained_model.get_leaf_counts()
        
        if voting == 'hard': # one-hot encoding of labels
            output_labels = np.argmax(self.data,axis=1)
            self.data = np.zeros(self.data.shape)
            self.data[np.arange(output_labels.size), output_labels] = 1

               
    @staticmethod
    def get_workload(classifier, samples):
        
        return classifier.inference_workload(samples)

    def predict(self, samples, eps, optimize=True, random_state=None):

        if self.voting == 'hard':
            self.eps_per_query = eps/2
        elif self.voting == 'weight':
            self.eps_per_query = eps
        else:
            raise ValueError("voting type of %s is not supported" %(self.voting) )
        if isinstance(self.trained_model, VerticalExtraTrees):
            self.eps_per_query /= self.trained_model.n_ensembles

        if optimize:
            if self.inference_query == None:
                self.set_workload(samples)
            if self.strategy == None:
                self.strategy = PredictMechanismBase.optimize(self.inference_query)

        return super().predict(samples, self.eps_per_query,optimize=optimize,random_state=random_state)

class BatchPredict(PredictMechanismBase):
    """ DP batch prediction technique
    """
    def __init__(self, schema, trained_model, inference_query=None, strategy_query=None):
        super().__init__(schema, trained_model, voting='weight', inference_query=inference_query, strategy_query=strategy_query)

        assert trained_model.disjoint == False
        
    @staticmethod
    def get_workload(classifier,samples):
        
        W = classifier.inference_workload(samples)
        if isinstance(classifier,VerticalExtraTrees):
            T = VerticalExtraTrees.get_workload(classifier.get_paths())
            assert len(W) == len(T)
            return  [Wi.dot(Ti) for Wi,Ti in zip(W,T)] # this is for VerticalExtraTrees
        else:
            T = ExtraTrees.get_workload(classifier.get_paths())
            return W.dot(T)
        

    
    def set_workload(self, samples):
        self.inference_query = BatchPredict.get_workload(self.trained_model, samples)
        #self.strategy = MemMatrix(workload.Identity(self.inference_query.shape[1]))

    def  predict(self, samples, eps, optimize=True,random_state=None):
        if optimize:
            self.eps_per_query = eps
        else:
            self.eps_per_query = eps/self.n_estimators

        if isinstance(self.trained_model, VerticalExtraTrees):
            self.eps_per_query /= self.trained_model.n_ensembles

        if optimize:
            if self.inference_query == None:
                self.set_workload(samples)
            if self.strategy == None:
                self.strategy = PredictMechanismBase.optimize(self.inference_query)

        return super().predict(samples, self.eps_per_query, optimize=optimize, random_state=random_state)



