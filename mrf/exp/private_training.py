import sys
sys.path.append("../src")
import init
import benchmark
from dp_random_forest import DPExtraTrees
from random_forest import ExtraTrees
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from os import cpu_count
RANDOM_STATE = None
N_JOBS = cpu_count()
def run(params, schema, traindata, testdata, random_state=None):
    
    print(params)
    n_trials = 1 if 'n_trials' not in params else params['n_trials']
    if not isinstance(params['eps'],list) and n_trials == 1:

        model = DPExtraTrees(schema, params['depth'], params['n_estimators'], 
                            params['bootstrap'],
                            params['disjoint'],
                            random_state=random_state,n_jobs=params['n_jobs'],alg=params['alg'])
        model.fit(traindata,eps=params['eps'],optimize=params['optimize'])
        pred = model.predict(testdata)
        return pred
    
    preds = []
    model = DPExtraTrees(schema, params['depth'], params['n_estimators'], 
                            params['bootstrap'],
                            params['disjoint'],
                            random_state=random_state,n_jobs=params['n_jobs'])
    model.fit(traindata, eps=None) # train withough dp
    
    ans = model.get_leaf_counts() # matrix
    
    if params['optimize']:
        W = ExtraTrees.get_workload(model.get_paths()).explicit_matrix()
        A = DPExtraTrees.optimize(W, schema.shape[:-1]).explicit_matrix()
        delta = A.sensitivity()
        WApinv = W.dot(A.pinv())
        for eps in params['eps']:
            noise = model.prng.laplace(loc=0.0, scale=delta/eps, size=(n_trials, A.shape[0],ans.shape[1]))
            for i in range(n_trials):
                dp_ans = ans + WApinv.dot(noise[i])
                model.update_leaf_counts(dp_ans)
                pred = model.predict(testdata,voting=params['voting'])
                preds.append([eps, i, pred])
    else:
        delta = params["n_estimators"] if params["disjoint"]==False else 1.0
        for eps in params['eps']:
            noise = model.prng.laplace(loc=0.0, scale=delta/eps, size=(n_trials, ans.shape[0],ans.shape[1]))
            for i in range(n_trials):
                dp_ans = ans + noise[i]
                model.update_leaf_counts(dp_ans)
                pred = model.predict(testdata,voting=params['voting'])
                preds.append([eps, i, pred])
            
    return preds

def get_accuracy(testdata, preds):
    # preds are the result retured from run method in private_inference
    df_test = testdata.df
    y_test = df_test[df_test.columns[-1]].values.tolist()
    results = []
    for y_pred in preds:
        acc = accuracy_score(y_test, y_pred[2])
        results.append([y_pred[0],y_pred[1],acc])
    return results


def base_experiment(params, dataset):

    traindata = dataset[0]
    testdata = dataset[1]#.train_test_split(0.2, random_state=100)
    result = []
    pred = run(params, traindata.schema, traindata, testdata, random_state=params['random_state'])
    acc = get_accuracy(testdata, pred)

    for var in acc:
        result.append([params['method'], params['depth'], params['n_estimators'],params['optimize'],params['voting'],testdata.size,params['alg'],
                    var[0],var[1],var[2]])
    df = pd.DataFrame(result, columns=['method','depth','n_estimators','optimize', 'voting','n_samples','alg','eps','run','acc'])
    return df

    

def experiment_1(dataset, name="temp/private_training_eps.csv", save=False):
    params = {'targetname':dataset.schema.attrnames[-1], 'bootstrap':False}
    params['depth'] = 4#4
    params['n_estimators'] = 30#80
    params['eps'] = [0.01,0.05,0.1,0.5,1.0]#[0.4,0.6,0.8,1.0,2.0]
    params['n_trials'] = 1#5#10
    params['random_state'] = RANDOM_STATE
    params['n_jobs'] = N_JOBS
    params['alg'] = 'ID3'
    params['voting'] = 'hard'

    traindata, testdata = dataset.train_test_split(0.2, random_state=100)

    dfs = []
    for method in ['optimize', 'disjoint', 'original']:
        params['method'] = method
        if method == 'optimize':
            params['optimize']=True
            params['disjoint']=False
        if method == 'disjoint':
            params['optimize']=False
            params['disjoint']=True
        if method == 'original':
            params['optimize']=False
            params['disjoint']=False
        dfs.append(base_experiment(params, [traindata, testdata]))
    if save:
        pd.concat(dfs).to_csv(name, index=False)

def experiment_2(dataset, name="temp/private_training_depth.csv",save=False):

    params = {'targetname':dataset.schema.attrnames[-1], 'bootstrap':False}
    params['n_estimators'] = 80
    params['eps'] = [2.0]
    params['n_trials'] = 10
    params['random_state'] = RANDOM_STATE
    params['n_jobs'] = N_JOBS
    params['alg'] = 'ID3'

    traindata, testdata = dataset.train_test_split(0.2, random_state=100)
    depths = [2,3,4,5]

    dfs = []
    for method in ['optimize', 'disjoint', 'original']:
        params['method'] = method
        if method == 'optimize':
            params['optimize']=True
            params['disjoint']=False
        if method == 'disjoint':
            params['optimize']=False
            params['disjoint']=True
        if method == 'original':
            params['optimize']=False
            params['disjoint']=False
        for depth in depths:
            params['depth'] = depth
            df = base_experiment(params, [traindata, testdata])
            dfs.append(df)
    if save:
        pd.concat(dfs).to_csv(name, index=False)

def experiment_3(dataset, name="temp/private_training_nestimators.csv",save=False):

    params = {'targetname':dataset.schema.attrnames[-1], 'bootstrap':False}
    params['depth'] = 4
    params['eps'] = [2.0]
    params['n_trials'] = 5
    params['random_state'] = RANDOM_STATE
    params['n_jobs'] = N_JOBS
    params['alg'] = 'ID3'

    traindata, testdata = dataset.train_test_split(0.2, random_state=100)
    n_estimators_list = [10,40,100]
    
    dfs = []
    for method in ['optimize', 'disjoint', 'original']:
        params['method'] = method
        if method == 'optimize':
            params['optimize']=True
            params['disjoint']=False
        if method == 'disjoint':
            params['optimize']=False
            params['disjoint']=True
        if method == 'original':
            params['optimize']=False
            params['disjoint']=False
        for n_estimators in n_estimators_list:
            params['n_estimators'] = n_estimators
            df = base_experiment(params, [traindata,testdata])
            dfs.append(df)
    if save:
        pd.concat(dfs).to_csv(name, index=False)

if __name__ == "__main__":
    
    #heart = benchmark.heart_dataset(drop=True)
    dataset = benchmark.car_dataset()
    #dataset = benchmark.adult_dataset()
    #dataset = benchmark.iris_dataset()
    experiment_1(dataset, name="temp/private_training_car_eps.csv", save=True)
    #experiment_2(dataset, name="temp/private_training_car_depth.csv", save=True)
    #experiment_3(dataset, name="temp/private_training_car_nestimators.csv", save=True)
    
    