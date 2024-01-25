import sys
sys.path.append("../src")
import init
import benchmark
from hdmm import workload
from random_forest import ExtraTrees
from mem_matrix import MemMatrix
from dp_prediction import SubsampleAndAggregate,BatchPredict
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from os import cpu_count


RANDOM_STATE_TRAIN = None
RANDOM_STATE_PREDICT = None
N_JOBS = cpu_count()


def run(params, testdata, trained_model,predicted_votes=None):
    print(params)
    W = None
    A = None
    testdata_notarget = testdata.drop([params['targetname']])
    
    if params['method'] == 'subsample':
        W = SubsampleAndAggregate.get_workload(trained_model, testdata_notarget)
        if params['optimize']:
            A = workload.Identity(W.shape[1])
    else:
        W = BatchPredict.get_workload(trained_model, testdata_notarget)
        if params['optimize']:
            A = workload.Identity(W.shape[1])
            
            '''
            For now we explicitly compute the matrices assume that the matrices are not huge

            For some reasons, MatMatrix class W does not provide proper ouput when explicit_matrix
            is performed. It seems something to do with VStack class in Ektelo.

            The following operation works properly but due to an experiment purpose,
            we want to avoide computing W.dot(A1) multiple times. 
            W.dot(A1).dot(noise).explicit_matrix() 
            '''
            
    preds = []
    
    if params["random_state"] is None:
        prng = np.random.mtrand._rand
    else:
        prng = np.random.RandomState(params["random_state"])

    n_trials = 1 if 'n_trials' not in params else params['n_trials']
    for eps in params['eps']:
        eps_per_query = eps/2 if params['voting'] == 'hard' else eps
        
        if params['optimize']==False:
            # LM
            # split the budget equally among test queries
            delta = predicted_votes.shape[0] if params['method']=='subsample' else predicted_votes.shape[0]*params['n_estimators']
            noise = prng.laplace(loc=0.0, scale=delta/eps_per_query, size=(n_trials, predicted_votes.shape[0],predicted_votes.shape[1]))
            for i in range(n_trials):
                Y = predicted_votes + noise[i]
                y_pred=np.argmax(Y,axis=1)
                preds.append([eps, i, y_pred])
        else:
            delta = A.sensitivity()
            noise = prng.laplace(loc=0.0, scale=delta/eps_per_query, size=(n_trials,A.shape[0],predicted_votes.shape[1]))
            for i in range(n_trials):
                #noise = MemMatrix.laplace_noise(prng=prng,scale=delta/eps_per_query,size=(A.shape[0],predicted_votes.shape[1]))
                #B = np.array(W.dot(noise).explicit_matrix())
                B = W.explicit_matrix().dot(noise[i])
                Y = predicted_votes + B
                y_pred=np.argmax(Y,axis=1)
                preds.append([eps, i, y_pred])
            
    return preds
    

def get_accuracy(testdata, preds):
    # preds are the result retured from run method in private_inference
    df_test = testdata.df
    y_test = df_test[df_test.columns[-1]].values.tolist()
    results = []
    for y_pred in preds:
        rocauc = accuracy_score(y_test, y_pred[2])
        results.append([y_pred[0],y_pred[1],rocauc])
    return results    

def get_forest_structure(schema, depth, n_estimators, alg='ID3'):

    decision_trees = ExtraTrees.decision_paths(schema, depth, n_estimators, random_state=RANDOM_STATE_TRAIN, n_jobs=N_JOBS, tree=True,alg=alg)
    return decision_trees

def experiment_subsample(params, dataset, random_forest):
    # run experiments for the subsample-and-aggregate framework
    # with variables: voting = [hard, weight] and optimize = [True, False]

    traindata = dataset[0]
    testdata = dataset[1]
    depth = params['depth']
    n_estimators = params['n_estimators']

    trained_model = ExtraTrees(traindata.schema, depth, n_estimators, disjoint=True, n_jobs=params['n_jobs'],alg=params['alg'])
    trained_model.fit_request(traindata, random_forest)
    
    del traindata

    result = []
    for voting in ['hard','weight']:
        params['voting']=voting
        votes = trained_model.predict(testdata, voting=params['voting'], return_votes=True)
        for optimize in [True, False]:
            params['optimize']=optimize
            pred = run(params, testdata, trained_model,votes)
            acc = get_accuracy(testdata, pred)

            for var in acc:
                result.append([params['method'],depth, n_estimators, params['optimize'],params['voting'],testdata.size,params['alg'],
                            var[0],var[1],var[2]])
    df = pd.DataFrame(result, columns=['method','depth','n_estimators','optimize','voting','n_samples','alg','eps','run','acc'])
    return df

def experiment_batch(params, dataset, random_forest):

    traindata = dataset[0]
    testdata = dataset[1]
    depth = params['depth']
    n_estimators = params['n_estimators']
    result = []
    
    trained_model = ExtraTrees(traindata.schema, depth, n_estimators, disjoint=False,n_jobs=params['n_jobs'],alg=params['alg'])
    trained_model.fit_request(traindata, random_forest)
    

    params['voting']='weight'
    votes = trained_model.predict(testdata, voting=params['voting'], return_votes=True)
    for optimize in [True, False]:
        params['optimize']=optimize
        pred = run(params, testdata, trained_model, votes)
        acc = get_accuracy(testdata, pred)

        for var in acc:
            result.append([params['method'],depth, n_estimators, params['optimize'],params['voting'], testdata.size, params['alg'],
                        var[0],var[1],var[2]])
    df = pd.DataFrame(result, columns=['method','depth','n_estimators','optimize','voting','n_samples','alg','eps','run','acc'])
    return df

def experiment_1(dataset, name="temp/private_inference_eps.csv",save=False):

    params = {'method':None,'optimize':None, 'voting':None,'targetname':dataset.schema.attrnames[-1]}
    params['eps'] = [0.4,0.6,0.8,1.0,2.0]
    params['n_trials'] = 5#10
    params['depth'] = 6#4
    params['n_estimators']=20
    params['random_state']=RANDOM_STATE_PREDICT
    params['n_jobs']=N_JOBS
    params['alg'] = 'CART' # ID3

    traindata, testdata = dataset.train_test_split(0.2, random_state=100)
    random_forest = get_forest_structure(dataset.schema,params['depth'],params['n_estimators'],alg=params['alg'])


    params['method'] = 'subsample'
    df_subsample = experiment_subsample(params, [traindata,testdata], random_forest)
 
    params['method'] = 'batch'
    df_batch = experiment_batch(params, [traindata,testdata], random_forest)
     
    if save:
        pd.concat([df_subsample,df_batch]).to_csv(name, index=False)

def experiment_2(dataset, name="temp/private_inference_nsamples.csv",save=False):
    # vary the number of samples

    params = {'method':None,'optimize':None, 'voting':None,'targetname':dataset.schema.attrnames[-1]}
    params['eps'] = [2.0]
    params['n_trials'] = 10
    params['depth'] = 4#3
    params['n_estimators']=50#10
    params['random_state']=RANDOM_STATE_PREDICT
    params['n_jobs']=N_JOBS
    test_sizes = [100,500,1000]#[100,150,200]

    traindata = dataset.copy()
    random_forest = get_forest_structure(dataset.schema,params['depth'],params['n_estimators'])

    dfs = []
    for test_size in test_sizes:
        testdata = dataset.sample(test_size)

        params['method'] = 'subsample'
        df_subsample = experiment_subsample(params, [traindata,testdata], random_forest)

        params['method'] = 'batch'
        df_batch = experiment_batch(params, [traindata,testdata], random_forest)

        dfs.append(df_subsample)
        dfs.append(df_batch)
    if save:
        pd.concat(dfs).to_csv(name, index=False)

def experiment_3(dataset, name="temp/private_inference_depth.csv",save=False):
    # vary the depth

    params = {'method':None,'optimize':None, 'voting':None,'targetname':dataset.schema.attrnames[-1]}
    params['eps'] = [2.0]
    params['n_trials'] = 10
    params['n_estimators']=80
    params['random_state']=RANDOM_STATE_PREDICT
    params['n_jobs']=N_JOBS
    depths = [2,3,4,5]#[2,3,4]

   
    traindata, testdata = dataset.train_test_split(0.2, random_state=100)
    
    dfs = []
    for depth in depths:
        params['depth'] = depth
        random_forest = get_forest_structure(dataset.schema,params['depth'],params['n_estimators'])

        params['method'] = 'subsample'
        df_subsample = experiment_subsample(params, [traindata,testdata], random_forest)

        params['method'] = 'batch'
        df_batch = experiment_batch(params, [traindata,testdata], random_forest)

        dfs.append(df_subsample)
        dfs.append(df_batch)
    if save:
        pd.concat(dfs).to_csv(name, index=False) 

def experiment_4(dataset, name="temp/private_inference_nestimators.csv",save=False):
    # vary the number of estimators

    params = {'method':None,'optimize':None, 'voting':None,'targetname':dataset.schema.attrnames[-1]}
    params['eps'] = [2.0]
    params['n_trials'] = 10
    params['depth'] = 4#3
    params['random_state']=RANDOM_STATE_PREDICT
    params['n_jobs']=N_JOBS
    nestimators_list = [10,50,100]#[5,10,20,40]

    traindata, testdata = dataset.train_test_split(0.2, random_state=100)


    dfs = []
    for n_estimators in nestimators_list:
        params['n_estimators'] = n_estimators
        random_forest = get_forest_structure(dataset.schema,params['depth'],params['n_estimators'])

        params['method'] = 'subsample'
        df_subsample = experiment_subsample(params, [traindata,testdata], random_forest)

        params['method'] = 'batch'
        df_batch = experiment_batch(params, [traindata,testdata], random_forest)

        dfs.append(df_subsample)
        dfs.append(df_batch)
    if save:
        pd.concat(dfs).to_csv(name, index=False) 
if __name__ == "__main__":
   
    car = benchmark.car_dataset()
    experiment_1(car, name='temp/private_inference_car_eps.csv',save=False)
    #experiment_2(car, name='temp/private_inference_car_nsamples.csv',save=True)
    #experiment_3(car, name='temp/private_inference_car_depth.csv',save=True)
    #experiment_4(car, name='temp/private_inference_car_nestimators.csv',save=True)
