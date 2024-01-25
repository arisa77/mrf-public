import config
import private_training
import private_inference
import itertools
import pandas as pd
def main(main_config):
    dfs = []
    dataset = main_config['data']
    traindata, testdata = dataset.train_test_split(0.2, random_state=100)


    params = {'targetname':dataset.schema.attrnames[-1],
              'random_state':main_config['random_state'],
              'n_jobs':main_config['n_jobs'],
              'n_trials':main_config['n_trials'],
              'eps':main_config['eps'],
              'bootstrap':False,
              }
    
    for type in main_config['type']:
        if type == 'training':
            for method in main_config['training method']:
                params['method'] = method
                if method == 'optimize':
                    params['optimize']=True
                    params['disjoint']=False
                elif method == 'disjoint':
                    params['optimize']=False
                    params['disjoint']=True
                elif method == 'original':
                    params['optimize']=False
                    params['disjoint']=False
                else:
                    break
                for depth,n_estimators,alg in itertools.product(main_config['depth'], main_config['n_estimators'],main_config['alg']):
                    params['depth'] = depth
                    params['n_estimators'] = n_estimators
                    params['alg'] = alg
                    params['voting']='hard'
                    dfs.append(private_training.base_experiment(params, [traindata, testdata]))
        elif type=='prediction':
            for depth,n_estimators,alg in itertools.product(main_config['depth'], main_config['n_estimators'],main_config['alg']):
                params['depth'] = depth
                params['n_estimators'] = n_estimators
                params['alg'] = alg 
                random_forest = private_inference.get_forest_structure(dataset.schema,params['depth'],params['n_estimators'],alg=params['alg'])

                for method in main_config['prediction method']:
                    params['method'] = method
                    if method == 'subsample':
                        df = private_inference.experiment_subsample(params, [traindata,testdata], random_forest)
                    if method=='batch':
                        df = private_inference.experiment_batch(params, [traindata,testdata], random_forest)
                    dfs.append(df)
    pd.concat(dfs).to_csv(main_config['output'], index=False)


def sample_size(main_config):
    '''measuring accuracy with varying test sample size for DP prediction'''
    dfs = []
    dataset = main_config['data']
    traindata, testdata_ = dataset.train_test_split(0.2, random_state=100)


    params = {'targetname':dataset.schema.attrnames[-1],
              'random_state':main_config['random_state'],
              'n_jobs':main_config['n_jobs'],
              'n_trials':main_config['n_trials'],
              'eps':main_config['eps'],
              'bootstrap':False,
              }
    
    for sample_size in main_config['sample size']:
        testdata = testdata_.sample(sample_size)
        
        
        for depth,n_estimators,alg in itertools.product(main_config['depth'], main_config['n_estimators'],main_config['alg']):
            params['depth'] = depth
            params['n_estimators'] = n_estimators
            params['alg'] = alg 
            random_forest = private_inference.get_forest_structure(dataset.schema,params['depth'],params['n_estimators'],alg=params['alg'])

            for method in main_config['prediction method']:
                params['method'] = method
                if method == 'subsample':
                    df = private_inference.experiment_subsample(params, [traindata,testdata], random_forest)
                if method=='batch':
                    df = private_inference.experiment_batch(params, [traindata,testdata], random_forest)
                dfs.append(df)
    pd.concat(dfs).to_csv(main_config['output'], index=False)


def run_default():
    config={}
    config['id'] = 0
    config['type'] = ['training','prediction']
    config['data'] = benchmark.car_dataset()
    config['eps'] = [0.4,0.6,0.8,1.0,2.0]
    config['depth'] = [2,3,4,5]
    config['n_trials'] = 5
    config['n_estimators']=[1,4,8,16,32,64,128]
    config['training method'] = ['optimize', 'disjoint', 'original']
    config['prediction method'] = ['subsample', 'batch']
    config['sample size'] = [10,50,100,500,1000]
    config['alg']=['ID3']
    config['output']='temp/car_id_0.csv'
    config['random_state'] = None
    config['n_jobs'] = 8
    
    main(config)
if __name__ == "__main__":
    main_config = config.default()
    #main_config=config.config_car()
    #main_config=config.config_iris()
    #main_config=config.config_adult()
    #main_config=config.config_tictactoe()
    #main_config = config.config_balancescale()

    main(main_config)
    #sample_size(main_config)
    