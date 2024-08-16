from random_forest import ExtraTrees,VerticalExtraTrees
import init
import numpy as np
from hdmm import templates
from mem_matrix import MemMatrix
from joblib import Parallel, delayed


class DPExtraTrees(ExtraTrees):
    def __init__(self, schema, depth, n_estimators, bootstrap=False, disjoint=False,random_state=None,n_jobs=None,alg='ID3'):
        super().__init__(schema, depth, n_estimators, bootstrap=bootstrap, disjoint=disjoint,random_state=random_state,n_jobs=n_jobs,alg=alg)

    def fit(self, dataset, eps=1.0, optimize=False):
        
        super().fit(dataset) # train
        counts = self.get_leaf_counts()
        if eps is not None:
            if optimize:
                W = DPExtraTrees.get_workload(self.get_paths())
                A = DPExtraTrees.optimize(W, self.schema.shape[:-1])
                
                A1 = MemMatrix(A.explicit_matrix().pinv())
                WA1 = W.dot(A1)
                delta = A.explicit_matrix().sensitivity()
                print('sensitivity:',delta)
                #noise = self.prng.laplace(loc=0.0, scale=delta/eps, size=(A.shape[0],counts.shape[1]))
                noise = MemMatrix.laplace_noise(prng=self.prng, scale=delta/eps,size=(A.shape[0],counts.shape[1]))
                b = WA1.dot(noise).explicit_matrix()
                #dp_counts = counts + W.dot(A.pinv()).dot(noise)
                dp_counts = counts+np.array(b)
            else:
                delta = self.n_estimators if self.disjoint==False else 1.0
                print('sensitivity:',delta)
                noise = self.prng.laplace(loc=0.0, scale=delta/eps, size=counts.shape)
                dp_counts = counts+noise
            self.update_leaf_counts(dp_counts)
    
    
    @staticmethod
    def optimize(W, domain, ps=None):
        
        if isinstance(W, MemMatrix):
            W = W.explicit_matrix()

        if not ps:
            ps = [1]*len(domain) 
        assert len(ps) == len(domain)

        pid = templates.KronPIdentity(ps,domain)
        pid.optimize(W)
        A = pid.strategy()
        return MemMatrix(A)
    
    




class DPVerticalExtraTrees(VerticalExtraTrees):
    def __init__(self, schema, depth, n_estimators, n_ensembles=1,max_nfeatures=10, bootstrap=False, disjoint=False,random_state=None,n_jobs=None,alg='ID3'):
        super().__init__(schema, depth, n_estimators, n_ensembles=n_ensembles,max_nfeatures=max_nfeatures, bootstrap=bootstrap, disjoint=disjoint,random_state=random_state,n_jobs=n_jobs,alg=alg)
        
    def fit(self, dataset, eps=1.0, optimize=False):
        
        super().fit(dataset) # train
        counts = self.get_leaf_counts()
        if eps is not None:
            if optimize:
                W = DPVerticalExtraTrees.get_workload(self.get_paths())
                domain = [ensemble.schema.shape[:-1] for ensemble in self.ensembles ]
                print(domain)
                A = DPVerticalExtraTrees.optimize(W, domain)
                dp_counts = []
                for i in range(self.n_ensembles):
                    print(A[i].explicit_matrix())
                    A1 = A[i].explicit_matrix().pinv()
                    WA1 = W[i].explicit_matrix().dot(A1)
                    delta = A[i].explicit_matrix().sensitivity()*self.n_ensembles # composition theorem
                    print('sensitivity:',delta)
                    noise = self.prng.laplace(loc=0.0,scale=delta/eps,size=(A[i].shape[0],counts[i].shape[1]))
                    b = WA1.dot(noise)
                    dp_counts.append(counts[i]+np.array(b))
            else:
                delta = self.n_estimators*self.n_ensembles if self.disjoint==False else self.n_ensembles
                print('sensitivity:',delta)
                dp_counts = []
                for i in range(self.n_ensembles):
                    noise = self.prng.laplace(loc=0.0, scale=delta/eps, size=counts[i].shape)
                    dp_counts.append(counts[i]+noise)
            self.update_leaf_counts(dp_counts)
    
    
    @staticmethod
    def optimize(W, domain, ps=None):
        #  W and domain are lists
        assert len(W) == len(domain)
        n_ensembles = len(W)
        # A = []
        # for i in range(n_ensembles):
        #     ps_i = ps[i] if ps else None
        #     Ai = DPExtraTrees.optimize(W[i],domain[i],ps_i)
        #     A.append(Ai)
        
        A = Parallel(n_jobs=n_ensembles, prefer='threads')(
            delayed(DPExtraTrees.optimize)(W[i],domain[i],ps[i] if ps else None)
            for i in range(n_ensembles)
        )

        return A