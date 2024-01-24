import sys
sys.path.append("../src")
import init
import benchmark
from dp_random_forest import DPVerticalExtraTrees
from random_forest import VerticalExtraTrees
from dp_prediction import BatchPredict
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import itertools
import private_training
from ektelo import workload
from os import cpu_count
RANDOM_STATE = None
N_JOBS = cpu_count()
def get_accuracy(testdata, preds):
    
    df_test = testdata.df
    y_test = df_test[df_test.columns[-1]].values.tolist()
    results = []
    for y_pred in preds:
        acc = accuracy_score(y_test, y_pred[2])
        results.append([y_pred[0],y_pred[1],acc])
    return results

def run_batchtrain(params, schema, traindata, testdata, random_state=None):
    dpcls = DPVerticalExtraTrees(schema, params['depth'], params['n_estimators'], n_ensembles=params['n_ensembles'], random_state=random_state,n_jobs=params['n_jobs'],alg=params['alg'])
    dpcls.fit(traindata,eps=None) # train without DP
    counts = dpcls.get_leaf_counts()

    W = DPVerticalExtraTrees.get_workload(dpcls.get_paths())
    domain = [ensemble.schema.shape[:-1] for ensemble in dpcls.ensembles ]
                
    A = DPVerticalExtraTrees.optimize(W, domain)
    
    results = []
    for eps in params['eps']:
        for j in range(params['n_trials']):
            dp_counts = []
            for i in range(params['n_ensembles']):
                A1 = A[i].explicit_matrix().pinv()
                WA1 = W[i].explicit_matrix().dot(A1)
                delta = A[i].explicit_matrix().sensitivity()*params['n_ensembles']
                noise = np.random.laplace(loc=0.0,scale=delta/eps,size=(A[i].shape[0],counts[i].shape[1]))
                b = WA1.dot(noise)
                dp_counts.append(counts[i]+np.array(b))
            dpcls.update_leaf_counts(dp_counts)
            pred=dpcls.predict(testdata)
            results.append([eps,j,pred])
    return results

def run_batchpred(params, schema, traindata, testdata, random_state=None):
    trained_model =VerticalExtraTrees(schema, params['depth'], params['n_estimators'], n_ensembles=params['n_ensembles'], random_state=random_state,n_jobs=params['n_jobs'],alg=params['alg'])
    trained_model.fit(traindata) 
    votes = trained_model.predict(testdata, voting=params['voting'], return_votes=True) #(samples, classes)
        
    results=[]
    W = BatchPredict.get_workload(trained_model, testdata)
    for eps in params['eps']:
        for j in range(params['n_trials']):
            Ys = []
            for W_i, votes_i in zip(W, votes):
                W_i = W_i.explicit_matrix()
                A_i = workload.Identity(W_i.shape[1])
                A1_i = A_i.pinv()
                delta = A_i.sensitivity()*params['n_ensembles']
                noise = np.random.laplace(loc=0.0, scale=delta/eps, size=(A_i.shape[0],votes_i.shape[1]))
                Y = votes_i + np.array(W_i.dot(A1_i).dot(noise))
                Ys.append(Y)
            pred = np.stack(Ys).sum(axis=0).argmax(axis=1)
            results.append([eps,j,pred])
    return results

def experiment(params, dataset,type):

    traindata = dataset[0]
    testdata = dataset[1]
    result = []
    if type == 'train':
        pred = run_batchtrain(params, traindata.schema, traindata, testdata, random_state=params['random_state'])
    elif type == 'pred':
        pred = run_batchpred(params, traindata.schema, traindata, testdata, random_state=params['random_state'])
    
    acc = get_accuracy(testdata, pred)

    for var in acc:
        result.append([params['method'], params['depth'], params['n_estimators'],params['optimize'],params['voting'],testdata.size,params['alg'],
                    var[0],var[1],var[2]])
    df = pd.DataFrame(result, columns=['method','depth','n_estimators','optimize', 'voting','n_samples','alg','eps','run','acc'])
    return df

def main(main_config):
    dfs = []
    dataset = main_config['data']
    traindata, testdata = dataset.train_test_split(0.2, random_state=100)


    params = {'targetname':dataset.schema.attrnames[-1],
              'random_state':main_config['random_state'],
              'n_jobs':main_config['n_jobs'],
              'n_trials':main_config['n_trials'],
              'eps':main_config['eps'],
              'n_ensembles':main_config['n_ensembles'],
              'bootstrap':False,
              'disjoint':False
              }

    for type in main_config['type']:
        if type == 'training':
            for method in main_config['training method']:
                params['method'] = method
                if method == 'optimize':
                    params['optimize']=True
                elif method == 'original':
                    params['optimize']=False
                else:
                    break
                for depth,n_estimators, alg in itertools.product(main_config['depth'], main_config['n_estimators'],main_config['alg']):
                    params['depth'] = depth
                    params['alg'] = alg
                    params['voting']='hard'
                    print(params)
                    if params['optimize']:
                        # divide by the number of ensembles so that
                        # the total number of trees match with the baseline below
                        params['n_estimators'] = int(n_estimators/params['n_ensembles'])
                        dfs.append(experiment(params, [traindata, testdata],'train'))
                    else:
                        params['n_estimators'] = n_estimators
                        dfs.append(private_training.base_experiment(params, [traindata, testdata]))
        
        elif type == 'prediction':
            params['method'] = 'batch'    
            params['optimize']=True
                
            for depth,n_estimators, alg in itertools.product(main_config['depth'], main_config['n_estimators'],main_config['alg']):
                params['depth'] = depth
                params['alg'] = alg
                params['voting']='weight'
                print(params)
                
                # divide by the number of ensembles so that
                # the total number of trees match with the baseline below
                params['n_estimators'] = int(n_estimators/params['n_ensembles'])
                dfs.append(experiment(params, [traindata, testdata],'pred'))
                

        pd.concat(dfs).to_csv(main_config['output'], index=False)



def config_heart():
    config={}


    config['id'] = 2
    config['type'] = ['prediction','training']
    config['data'] = benchmark.heart_dataset()
    config['eps'] = [0.4,0.6,0.8,1.0,2.0]
    config['depth'] = [2]
    config['n_trials'] = 10
    config['n_estimators']=[102]
    config['n_ensembles']=3
    config['training method'] = ['optimize','original']
    config['alg']=['ID3']
    config['output']='temp/heart_nensembles3_2.csv'
    config['random_state'] = None
    config['n_jobs'] = 8

    return config

def config_mushroom():
    config={}

    config['id'] = 2
    config['type'] = ['prediction','training']
    config['data'] = benchmark.mushroom_dataset()
    config['eps'] = [0.01,0.03,0.05,0.1,0.2]
    config['depth'] = [3]
    config['n_trials'] = 5
    config['n_estimators']=[125]
    config['n_ensembles']=5
    config['training method'] = ['optimize','original']
    config['alg']=['ID3']
    config['output']='temp/mushroom_nensembles5_2.csv'
    config['random_state'] = None
    config['n_jobs'] = 8

    return config

if __name__ == "__main__":
    
    main_config=config_heart()
    #main_config=config_mushroom()
    main(main_config)

