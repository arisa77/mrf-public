import sys
sys.path.append("../src")
import init
import benchmark
from joblib import Parallel, delayed
from dp_prediction import SubsampleAndAggregate,BatchPredict
from random_forest import ExtraTrees,VerticalExtraTrees
from dp_random_forest import DPExtraTrees
from sklearn.metrics import roc_auc_score,accuracy_score
from ektelo import matrix
import numpy as np
import pandas as pd
import time
from os import cpu_count
EPS = 1.0
RANDOM_STATE = None #12345 #None
N_JOBS = -1#-1 #-1 #cpu_count()
BUILDTREE_LABEL = 'Tree Build.'
OPT_LABEL = 'OPT'
NOISE_LABEL = 'Noise Gen.'
FIT_LABEL = 'Fitting'
PREDICT_LABEL = 'Predict'
DPMEASURE_LABEL = 'DP Measure' 
TOTAL_LABEL = 'Total w/o Tree Build.'
ACCURACY = 'Accuracy'

def get_accuracy(testdata, preds):
    df_test = testdata.df
    y_test = df_test[df_test.columns[-1]].values.tolist()
    acc = accuracy_score(y_test, preds)
    return acc
    
def build_trees(schema, depth, n_estimators, alg='ID3', n_ensembles=None):
    start = time.time()
    if not n_ensembles:
        trees = ExtraTrees.decision_paths(schema, depth, n_estimators, random_state=RANDOM_STATE, n_jobs=N_JOBS, tree=True,alg=alg)
    else:
        trees = VerticalExtraTrees.decision_paths(schema, depth, n_estimators, n_ensembles, random_state=RANDOM_STATE, n_jobs=N_JOBS, tree=True,alg=alg)
    end = time.time()
    print('Building Tree:',end-start)
    return trees

def nonprivate(trees, schema, traindata, testdata, depth, n_estimators):
    out = {}
    start = time.time()
    model = ExtraTrees(schema, depth, n_estimators, 
                                False,
                                False,
                                random_state=RANDOM_STATE,n_jobs=N_JOBS)
    

    '''
    1. Optimization (pre-processing)
        not applicable
    '''
    out[OPT_LABEL] = 0

    '''
    2. Data-independent preprocessing
        not applicable
    '''
    out[NOISE_LABEL] = 0

    '''
    3. Fitting training data
    '''
    s3 = time.time()
    model.fit_request(traindata, trees)
    out[FIT_LABEL] = time.time()-s3
    print('Fitting training data:',time.time()-s3)

    '''
    4. DP Measurement
        not applicable
    '''
    out[DPMEASURE_LABEL] = 0

    
    start = time.time()
    y = model.predict(testdata)
    out[PREDICT_LABEL] = time.time()-start
    print('Prediction',time.time()-start)

    acc =  get_accuracy(testdata,y)
    out[ACCURACY] = acc
    print('Accuracy', acc)
    return out

def DPBatchTrain(trees, schema, traindata, testdata, depth, n_estimators):
    out = {}
    start = time.time()
    model = ExtraTrees(schema, depth, n_estimators, 
                                False,
                                False,
                                random_state=RANDOM_STATE,n_jobs=N_JOBS)
    

    '''
    1. Optimization (pre-processing)
        : can be invoked ahead of time without accessing the actucal data
    '''
    s1 = time.time()
    W = ExtraTrees.get_workload([tree['path'] for tree in trees]).explicit_matrix()
    #s1 = time.time()
    A = DPExtraTrees.optimize(W, schema.shape[:-1]).explicit_matrix()
    out[OPT_LABEL] = time.time()-s1
    print('Optimization:',time.time()-s1)
    delta = A.sensitivity()

    '''
    2. Data-independent preprocessing
    '''
    s2 = time.time()
    WApinv = W.dot(A.pinv())
    noise = model.prng.laplace(loc=0.0, scale=delta/EPS, size=(A.shape[0],schema.shape[-1]))
    noise = WApinv.dot(noise) # refined noise
    out[NOISE_LABEL] = time.time()-s2
    print('Data-independent preprocessing:',time.time()-s2)

    '''
    3. Fitting training data
    '''
    s3 = time.time()
    model.fit_request(traindata, trees)
    ans = model.get_leaf_counts() # matrix
    out[FIT_LABEL] = time.time()-s3
    print('Fitting training data:',time.time()-s3)

    '''
    4. DP Measurement
    '''
    s4 = time.time()
    dp_ans = ans + noise
    model.update_leaf_counts(dp_ans)
    out[DPMEASURE_LABEL] = time.time()-s4
    print('DP measurement:',time.time()-s4)

    # out[TOTALTRAIN_LABEL] = time.time()-start
    # print('Total training :',time.time()-start)

    start = time.time()
    y = model.predict(testdata)
    out[PREDICT_LABEL] = time.time()-start
    print('Prediction',time.time()-start)

    acc =  get_accuracy(testdata,y)
    out[ACCURACY] = acc
    print('Accuracy', acc)
    
    return out

    
def HDDPBatchTrain(trees, schema, traindata, testdata, depth, n_estimators, n_ensembles):
    out = {}
    # Vertical Partitioning

    start = time.time()
    model = VerticalExtraTrees(schema, depth, n_estimators, n_ensembles,random_state=RANDOM_STATE,n_jobs=N_JOBS,alg='ID3')
    
    '''
    1. Optimization (pre-processing)
        : can be invoked ahead of time without accessing the actucal data
    '''
    s1 = time.time()
    W = VerticalExtraTrees.get_workload([[tree['path'] for tree in trees_per_ensemble] for trees_per_ensemble in trees])
    domains = [ensemble.schema.shape[:-1] for ensemble in model.ensembles]
    
    #s1 = time.time()
    A = Parallel(n_jobs=N_JOBS, prefer='threads')(
            delayed(DPExtraTrees.optimize)(W[i],domains[i],None)
            for i in range(n_ensembles)
    )
    out[OPT_LABEL] = time.time()-s1
    print('Optimization:',time.time()-s1)

    '''
    2. Data-independent preprocessing
    '''
    
    def _preprocess(Wi, Ai,schemai):
        Wi = Wi.explicit_matrix()
        Ai = Ai.explicit_matrix()
        WApinv = Wi.dot(Ai.pinv())
        delta = Ai.sensitivity()
        noise = np.random.laplace(loc=0.0, scale=delta/(EPS/n_ensembles), size=(Ai.shape[0],schemai.shape[-1]))
        noise = WApinv.dot(noise) # refined noise
        return noise
    
    s2 = time.time()
    noise = Parallel(n_jobs=1, prefer='threads')(
            delayed(_preprocess)(W[i],A[i],model.ensembles[i].schema)
            for i in range(n_ensembles)
    )
    out[NOISE_LABEL] = time.time()-s2
    print('Data-independent preprocessing:',time.time()-s2)


    '''
    3. Fitting training data
    '''
    s3 = time.time()
    model.fit_request(traindata, trees)
    ans = model.get_leaf_counts() # matrix
    out[FIT_LABEL]=time.time()-s3
    print('Fitting training data:',time.time()-s3)

    '''
    4. DP Measurement
        - threading may cause slower performance
    '''
    def _dp_measure(ansi, noisei):
        return ansi + noisei
    s4 = time.time()
    dp_ans = Parallel(n_jobs=1, prefer='threads')(
            delayed(_dp_measure)(ans[i],noise[i])
            for i in range(n_ensembles)
    )
    #dp_ans = [ans[i] + noise[i] for i in range(n_ensembles)]
    model.update_leaf_counts(dp_ans)
    out[DPMEASURE_LABEL]=time.time()-s4
    print('DP measurement:',time.time()-s4)

    print('Total training :',time.time()-start)

    start = time.time()
    y = model.predict(testdata)
    out[PREDICT_LABEL] = time.time()-start
    print('Prediction',time.time()-start)

    acc =  get_accuracy(testdata,y)
    out[ACCURACY] = acc
    print('Accuracy', acc)

    return out


def DPBatchPred(trees, schema, traindata, testdata, depth, n_estimators):
    out = {}
    start = time.time()
    trained_model = ExtraTrees(schema, depth, n_estimators, disjoint=False, random_state = RANDOM_STATE,n_jobs=N_JOBS,alg='ID3')
    trained_model.update_forest(trees)
    
    '''
    1. Optimization 
        - this is prediction at a time
        - can be processed without accessing the training data
    '''
    s1 = time.time()
    #W = BatchPredict.get_workload(trained_model, testdata.drop([-1], indices=True)).explicit_matrix()
    W = trained_model.inference_workload(testdata.drop([-1], indices=True)).explicit_matrix()
    #s1 = time.time()
    T = ExtraTrees.get_workload([tree['path'] for tree in trees]).explicit_matrix()
    W = matrix.EkteloMatrix(W.dense_matrix().dot(T.dense_matrix()))
    #s1 = time.time()
    #A = MemMatrix(workload.Identity(W.shape[1]))
    A = BatchPredict.optimize(W,p=1).explicit_matrix()
    out[OPT_LABEL] = time.time()-s1
    print('Optimization:',time.time()-s1)
    delta = A.sensitivity()

    '''
    2. Data-independent preprocessing
    '''
    s2 = time.time()
    A1 = A.pinv()
    noise = trained_model.prng.laplace(loc=0.0, scale=delta/EPS, size=(A.shape[0],schema.shape[-1]))
    noise = W.dot(A1).dot(noise)
    out[NOISE_LABEL]=time.time()-s2
    print('Data-independent preprocessing:',time.time()-s2)

    '''
    3. Fitting Training data and get predictions 
        - this is training time
        - can be processed without accessing the training data
    '''
    s3 = time.time()
    trained_model.fit_request(traindata, trees)
    out[FIT_LABEL]=time.time()-s3
    print('Fitting training data:',time.time()-s3)

    s3 = time.time()
    votes = trained_model.predict(testdata, voting='weight', return_votes=True) #(samples, classes)
    out[PREDICT_LABEL]=time.time()-s3
    print('Prediction:',time.time()-s3)

    '''
    4. DP Measurement for prediction
    '''
    s4 = time.time()
    Y = votes + noise
    y = np.argmax(Y,axis=1)
    out[DPMEASURE_LABEL]=time.time()-s4
    print('DP measurement:', time.time()-s4)

    print('Total:',time.time()-start)

    acc =  get_accuracy(testdata,y)
    out[ACCURACY] = acc
    print('Accuracy', acc)

    return out

def HDDPBatchPred(trees, schema, traindata, testdata, depth, n_estimators, n_ensembles):
    out = {}
    start = time.time()
    trained_model = VerticalExtraTrees(schema, depth, n_estimators, n_ensembles,disjoint=False, random_state = RANDOM_STATE,n_jobs=N_JOBS,alg='ID3')
    trained_model.update_forest(trees)
    
    '''
    1. Optimization 
        - this is prediction at a time
        - can be processed without accessing the training data
    '''
       
    
    def _optimize(Wi, Ti):
        Wi = Wi.explicit_matrix()
        Ti = Ti.explicit_matrix()
        Wi =  matrix.EkteloMatrix(Wi.dense_matrix().dot(Ti.dense_matrix()))
        Ai = BatchPredict.optimize(Wi,p=1).explicit_matrix()
        return [Wi, Ai]

    s1 = time.time()
    W = trained_model.inference_workload(testdata)
    T = VerticalExtraTrees.get_workload([[tree['path'] for tree in trees_per_ensemble] for trees_per_ensemble in trees])
 
    #s1 = time.time()
    #A = MemMatrix(workload.Identity(W.shape[1]))
    M = Parallel(n_jobs=N_JOBS, prefer='threads')(
            delayed(_optimize)(W[i], T[i])
            for i in range(n_ensembles)
    )
    out[OPT_LABEL] = time.time() - s1
    print('Optimization:',time.time()-s1)
    

    '''
    2. Data-independent preprocessing
    '''
    def _preprocess(Mi,schemai):
        Wi = Mi[0]
        Ai = Mi[1]
        WApinv = Wi.dot(Ai.pinv())
        delta = Ai.sensitivity()
        noise = np.random.laplace(loc=0.0, scale=delta/(EPS/n_ensembles), size=(Ai.shape[0],schemai.shape[-1]))
        noise = WApinv.dot(noise) # refined noise
        return noise
    s2 = time.time()
    noise = Parallel(n_jobs=N_JOBS, prefer='threads')(
            delayed(_preprocess)(M[i],trained_model.ensembles[i].schema)
            for i in range(n_ensembles)
    )
    out[NOISE_LABEL]=time.time()-s2
    print('Data-independent preprocessing:',time.time()-s2)

    '''
    3. Fitting Training data and get predictions 
        - this is training time
        - can be processed without accessing the training data
    '''
    s3 = time.time()
    trained_model.fit_request(traindata, trees)
    out[FIT_LABEL] = time.time() - s3

    s3 = time.time()
    votes = trained_model.predict(testdata, voting='weight', return_votes=True) #(samples, classes)
    out[PREDICT_LABEL] = time.time() - s3
    
    print('Fitting training data:',time.time()-s3)

    '''
    4. DP Measurement for prediction
    '''
    def _dp_measure(votei, noisei):
        return votei + noisei
    s4 = time.time()
    #y = [np.argmax(votes[i] + noise[i],axis=1) for i in range(n_ensembles)]
    Y = Parallel(n_jobs=N_JOBS, prefer='threads')(
            delayed(_dp_measure)(votes[i],noise[i])
            for i in range(n_ensembles)
    )
    out[DPMEASURE_LABEL] = time.time() - s4
    print('DP measurement:', time.time()-s4)

    print('Total:',time.time()-start)
    
    y = sum(Y).argmax(axis=1)
    acc =  get_accuracy(testdata,y)
    out[ACCURACY] = acc
    print('Accuracy', acc)

    return out
        
def SubsampleAggregate(trees, schema, traindata, testdata, depth, n_estimators):
    out = {}
    start = time.time()
    trained_model = ExtraTrees(schema, depth, n_estimators, disjoint=True, random_state = RANDOM_STATE,n_jobs=N_JOBS,alg='ID3')
    trained_model.update_forest(trees)

    '''
    1. Optimization 
        - this is prediction at a time
        - can be processed without accessing the training data
    '''
    s1 = time.time()
    W = trained_model.inference_workload(testdata.drop([-1], indices=True)).explicit_matrix()
    #s1 = time.time()
    #A = MemMatrix(workload.Identity(W.shape[1]))
    A = SubsampleAndAggregate.optimize(W,p=1).explicit_matrix()
    out[OPT_LABEL] = time.time() - s1
    print('Optimization:',time.time()-s1)
    delta = A.sensitivity()

    '''
    2. Data-independent preprocessing
    '''
    s2 = time.time()
    A1 = A.pinv()
    noise = trained_model.prng.laplace(loc=0.0, scale=delta/EPS, size=(A.shape[0],schema.shape[-1]))
    noise = W.dot(A1).dot(noise)
    out[NOISE_LABEL] = time.time() - s2
    print('Data-independent preprocessing:',time.time()-s2)

    '''
    3. Fitting Training data and get predictions 
        - this is training time
        - can be processed without accessing the training data
    '''
    s3 = time.time()
    trained_model.fit_request(traindata, trees)
    out[FIT_LABEL] = time.time() - s3

    s3 = time.time()
    votes = trained_model.predict(testdata, voting='weight', return_votes=True) #(samples, classes)
    out[PREDICT_LABEL] = time.time() - s3
    print('Fitting training data:',time.time()-s3)

    '''
    4. DP Measurement for prediction
    '''
    s4 = time.time()
    Y = votes + noise
    y = np.argmax(Y,axis=1) 
    out[DPMEASURE_LABEL] = time.time() - s4
    print('DP measurement:', time.time()-s4)
    
    print('Total:',time.time()-start)

    acc =  get_accuracy(testdata,y)
    out[ACCURACY] = acc
    print('Accuracy', acc)
    
    return out

def HDDSubsampleAggregate(trees, schema, traindata, testdata, depth, n_estimators, n_ensembles):
    out = {}
    start = time.time()
    trained_model = VerticalExtraTrees(schema, depth, n_estimators, n_ensembles,disjoint=True, random_state = RANDOM_STATE,n_jobs=N_JOBS,alg='ID3')
    trained_model.update_forest(trees)
    
    '''
    1. Optimization 
        - this is prediction at a time
        - can be processed without accessing the training data
    '''
   
    
    
    
    def _optimize(Wi):
        Wi = Wi.explicit_matrix()
        Ai = SubsampleAndAggregate.optimize(Wi,p=1).explicit_matrix()
        return [Wi, Ai]

    s1 = time.time()
    W = trained_model.inference_workload(testdata)
    #A = MemMatrix(workload.Identity(W.shape[1]))
    M = Parallel(n_jobs=N_JOBS, prefer='threads')(
            delayed(_optimize)(W[i])
            for i in range(n_ensembles)
    )
    out[OPT_LABEL] = time.time() - s1
    print('Optimization:',time.time()-s1)
    

    '''
    2. Data-independent preprocessing
    '''
    def _preprocess(Mi,schemai):
        Wi = Mi[0]
        Ai = Mi[1]
        WApinv = Wi.dot(Ai.pinv())
        delta = Ai.sensitivity()
        noise = np.random.laplace(loc=0.0, scale=delta/EPS, size=(Ai.shape[0],schemai.shape[-1]))
        noise = WApinv.dot(noise) # refined noise
        return noise
    s2 = time.time()
    noise = Parallel(n_jobs=N_JOBS, prefer='threads')(
            delayed(_preprocess)(M[i],trained_model.ensembles[i].schema)
            for i in range(n_ensembles)
    )
    out[NOISE_LABEL]=time.time()-s2
    print('Data-independent preprocessing:',time.time()-s2)

    '''
    3. Fitting Training data and get predictions 
        - this is training time
        - can be processed without accessing the training data
    '''
    s3 = time.time()
    trained_model.fit_request(traindata, trees)
    out[FIT_LABEL] = time.time() - s3

    s3 = time.time()
    votes = trained_model.predict(testdata, voting='weight', return_votes=True) #(samples, classes)
    out[PREDICT_LABEL] = time.time() - s3
    
    print('Fitting training data:',time.time()-s3)

    '''
    4. DP Measurement for prediction
    '''
    def _dp_measure(votei, noisei):
        return votei + noisei
    s4 = time.time()
    #y = [np.argmax(votes[i] + noise[i],axis=1) for i in range(n_ensembles)]
    Y = Parallel(n_jobs=N_JOBS, prefer='threads')(
            delayed(_dp_measure)(votes[i],noise[i])
            for i in range(n_ensembles)
    )
    out[DPMEASURE_LABEL] = time.time() - s4
    print('DP measurement:', time.time()-s4)

    print('Total:',time.time()-start)

    y = sum(Y).argmax(axis=1)
    acc =  get_accuracy(testdata,y)
    out[ACCURACY] = acc
    print('Accuracy', acc)

    return out

def main(params,methods,outputfile=None):
    result = []
    dataset = benchmark.get_dataset(params['dataset'])#.sample(size=1000)
    traindata, testdata = dataset.train_test_split(0.2, random_state=100)
    print(traindata.df.shape, testdata.df.shape)
    for _ in range(params['n_trials']):
        if 'vbatchtrain' in methods or  'vbatchpred' in methods or 'vsubsample' in methods:
            # randomly sample features
            vschema = dataset.schema.random_feature_subspace(params['max_nfeatures'],n_sets=params['n_ensembles'],random_state=RANDOM_STATE)
            n_vestimators = int(params['n_estimators']/params['n_ensembles'])
            print(n_vestimators)
            start = time.time()
            vtrees = build_trees(vschema, params['depth'],n_vestimators,n_ensembles=params['n_ensembles'])
            vtree_building_time = time.time()-start
        if 'nonprivate' in methods or 'batchtrain' in methods or 'batchpred' in methods or 'subsample' in methods:
            start = time.time()
            schema = dataset.schema
            trees = build_trees(schema,params['depth'],params['n_estimators'])
            tree_building_time = time.time() - start

    
        for method in methods:
            out = None
            start = time.time()
            if method == 'vbatchtrain' or method == 'vbatchpred' or method =='vsubsample':
                if method == 'vbatchtrain':
                    out = HDDPBatchTrain(vtrees, vschema, traindata, testdata, params['depth'],n_vestimators,params['n_ensembles'])
                elif method == 'vbatchpred':
                    out = HDDPBatchPred(vtrees, vschema, traindata, testdata, params['depth'],n_vestimators,params['n_ensembles'])
                elif method =='vsubsample':
                    out = HDDSubsampleAggregate(vtrees, vschema, traindata, testdata, params['depth'],n_vestimators,params['n_ensembles'])
                out[BUILDTREE_LABEL] = vtree_building_time
            else:
                if method == 'nonprivate':
                    out = nonprivate(trees, schema, traindata, testdata, params['depth'],params['n_estimators'])
                elif method == 'batchtrain':
                    out = DPBatchTrain(trees, schema, traindata, testdata, params['depth'],params['n_estimators'])
                elif method =='batchpred':
                    out = DPBatchPred(trees, schema, traindata, testdata, params['depth'],params['n_estimators'])
                elif method=='subsample':
                    out = SubsampleAggregate(trees, dataset.schema, traindata, testdata, params['depth'],params['n_estimators']) 
                out[BUILDTREE_LABEL] = tree_building_time
            out[TOTAL_LABEL] = time.time() - start
            out['Method'] = method
            result.append(out)
    
    
    df = pd.DataFrame(result)
    df = df[['Method',BUILDTREE_LABEL,OPT_LABEL,NOISE_LABEL,FIT_LABEL,DPMEASURE_LABEL,PREDICT_LABEL,TOTAL_LABEL,ACCURACY]]
    df['dataset'] = params['dataset']
    df['depth'] = params['depth']
    df['n_estimators'] = params['n_estimators']
    df = df.assign(n_ensembles=lambda x: x['Method'].apply(lambda y: params['n_ensembles'] if y in ['vbatchtrain', 'vbatchpred'] else np.nan))
    df = df.assign(max_nfeatures=lambda x: x['Method'].apply(lambda y: params['max_nfeatures'] if y in ['vbatchtrain', 'vbatchpred'] else np.nan))

    print(df[['Method',BUILDTREE_LABEL,OPT_LABEL,NOISE_LABEL,FIT_LABEL,DPMEASURE_LABEL,PREDICT_LABEL,TOTAL_LABEL, ACCURACY]])
    if outputfile:
        df.to_csv(outputfile, index=False) # saving the result into the outputfile
    

def get_params(dataset):
    params={}
    params['dataset'] = dataset
    params['n_trials'] = 5
    save_dir ="temp/"
    if dataset == 'car':
        params['outputfile'] = save_dir+'runtime_car_subsample.csv'
        params['depth'] = 3#4
        params['n_estimators'] = 16#100
        params['n_ensembles'] = 2
        params['max_nfeatures'] = 4
        params['methods'] = ['nonprivate','subsample','vsubsample']

        # params['outputfile'] = save_dir+'runtime_car.csv'
        # params['depth'] = 4
        # params['n_estimators'] = 128
        # params['n_ensembles'] = 2
        # params['max_nfeatures'] = 4
        # params['methods'] = ['nonprivate','batchtrain','batchpred','vbatchtrain','vbatchpred']

    if dataset == 'adult_binary':
        params['outputfile'] = save_dir+'runtime_adult_binary_subsample.csv'
        params['depth'] = 6
        params['n_estimators'] = 20
        params['n_ensembles'] = 4
        params['max_nfeatures'] = 10
        params['methods'] = ['nonprivate','vsubsample']#['nonprivate','vbatchtrain','vbatchpred']  # subsample taking too long time

        # params['outputfile'] = save_dir+'runtime_adult_binary_3.csv'
        # params['depth'] = 8
        # params['n_estimators'] = 60
        # params['n_ensembles'] = 3
        # params['max_nfeatures'] = 10
        # params['methods'] = ['nonprivate','vbatchtrain','vbatchpred']  # subsample taking too long time


    if dataset == 'mushroom':
        params['outputfile'] = save_dir+'runtime_mushroom.csv'
        params['depth'] = 3
        params['n_estimators'] = 125
        params['n_ensembles'] = 5
        params['max_nfeatures'] = 4
        params['methods'] = ['nonprivate','vbatchtrain','vbatchpred'] # subsample taking too long time if n_estimators is large

    if dataset == 'adult':
        params['outputfile'] = 'temp/runtime_adult_subsample.csv'
        params['depth'] = 6
        params['n_estimators'] = 16
        params['n_ensembles'] = 2
        params['max_nfeatures'] = 8
        params['methods'] = ['vsubsample']#['nonprivate','batchtrain','batchpred','vbatchtrain','vbatchpred','subsample'] # batchpred taking for too long time.

    if dataset =='heart':
        params['outputfile'] = save_dir+'runtime_heart.csv'
        params['depth'] = 2
        params['n_estimators'] = 128
        params['n_ensembles'] = 2
        params['max_nfeatures'] = 4
        params['methods'] = ['nonprivate','vbatchtrain','vbatchpred'] 

    if dataset =='iris':
        params['outputfile'] = save_dir+'runtime_iris.csv'
        params['depth'] = 2
        params['n_estimators'] = 64
        params['n_ensembles'] = 1
        params['max_nfeatures'] = None
        params['methods'] = ['nonprivate','batchtrain','batchpred'] 

    if dataset =='balance':
        params['outputfile'] = save_dir+'runtime_balance.csv'
        params['depth'] = 2
        params['n_estimators'] = 128
        params['n_ensembles'] = 1
        params['max_nfeatures'] = None
        params['methods'] = ['nonprivate','batchtrain','batchpred']
    return params




if __name__ == "__main__":
    
    #params = get_params('car')
    #params = get_params('mushroom')
    #params = get_params('adult')
    #params = get_params('adult_binary')
    params = get_params('heart')
    #params = get_params('iris')
    #params = get_params('balance')
    

    #main(params,params['methods'],outputfile=params['outputfile'])
    main(params,params['methods'])
    