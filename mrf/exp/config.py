import sys
sys.path.append("../src")
import benchmark

def config_adult():
    config={}

    config['id'] = 2
    config['type'] = ['prediction','training']
    config['data'] = benchmark.adult_dataset()
    config['eps'] = [0.01,0.05,0.1,0.5,1.0]
    config['depth'] = [4]
    config['n_trials'] = 5
    config['n_estimators']=[32]
    config['training method'] = ['optimize', 'disjoint', 'original']
    config['prediction method'] = ['subsample', 'batch']
    config['sample size'] = []
    config['alg']=['ID3']
    config['output']='temp/adult_id2.csv'
    config['random_state'] = None
    config['n_jobs'] = 8
    return config

    # config['id'] = 4
    # config['type'] = ['prediction']
    # config['data'] = benchmark.adult_dataset()
    # config['eps'] = [0.01,0.05,0.1,0.5,1.0]
    # config['depth'] = [6]
    # config['n_trials'] = 5
    # config['n_estimators']=[16]
    # config['prediction method'] = ['subsample']
    # config['sample size'] = []
    # config['alg']=['ID3']
    # config['output']='temp/adult_id4.csv'
    # config['random_state'] = None
    # config['n_jobs'] = 8


    # return config

def config_car():
    config={}
    config['id'] = 1
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
    config['output']='temp/car_id1.csv'
    config['random_state'] = None
    config['n_jobs'] = 8
    return config

    # config['id'] = 2
    # config['type'] = ['prediction']
    # config['data'] = benchmark.car_dataset()
    # config['eps'] = [0.4,0.6,0.8,1.0,2.0]
    # config['depth'] = [2,3,4,5]
    # config['n_trials'] = 5
    # config['n_estimators']=[1,4,8,16,32,64,128]
    # config['prediction method'] = ['subsample']
    # config['alg']=['ID3']
    # config['output']='temp/car_id2.csv'
    # config['random_state'] = None
    # config['n_jobs'] = 8
    #return config

    # config['id'] = 10
    # config['type'] = ['prediction']
    # config['data'] = benchmark.car_dataset()
    # config['eps'] = [2.0]
    # config['depth'] = [4]
    # config['n_trials'] = 5
    # config['n_estimators']=[16]
    # config['prediction method'] = ['subsample', 'batch']
    # config['sample size'] = [5,10,50,100,500,1000]
    # config['alg']=['ID3']
    # config['output']='temp/car_id10.csv'
    # config['random_state'] = None
    # config['n_jobs'] = 8

    #return config


def config_iris():
    config={}
    config['id'] = 1
    config['type'] = ['prediction','training']
    config['data'] = benchmark.iris_dataset()
    config['eps'] = [0.4,0.6,0.8,1.0,2.0]
    config['depth'] = [2,3,4]
    config['n_trials'] = 10
    config['n_estimators']=[1,4,8,16,32,64]
    config['training method'] = ['optimize', 'disjoint', 'original']
    config['prediction method'] = ['subsample', 'batch']
    config['sample size'] = [10,50,100]
    config['alg']=['ID3']
    config['output']='temp/iris_id1.csv'
    config['random_state'] = None
    config['n_jobs'] = 8
    return config

    # config['id'] = 2
    # config['type'] = ['prediction']
    # config['data'] = benchmark.iris_dataset()
    # config['eps'] = [0.4,0.6,0.8,1.0,2.0]
    # config['depth'] = [2,3,4]
    # config['n_trials'] = 10
    # config['n_estimators']=[1,4,8,16,32,64]
    # config['prediction method'] = ['subsample']
    # config['alg']=['ID3']
    # config['output']='temp/iris_id2.csv'
    # config['random_state'] = None
    # config['n_jobs'] = 8
    #return config
    

def config_balancescale():
    config={}

    config['id'] = 4
    config['type'] = ['training','prediction']
    config['data'] = benchmark.balancescale_dataset()
    config['eps'] = [0.4,0.6,0.8,1.0,2.0]
    config['depth'] = [2]
    config['n_trials'] = 5
    config['n_estimators']=[128]
    config['training method'] = ['optimize', 'disjoint', 'original']
    config['prediction method'] = ['subsample', 'batch']
    config['sample size'] = [10,50,100,500]
    config['alg']=['ID3']
    config['output']='temp/balancescale_id4.csv'
    config['random_state'] = None
    config['n_jobs'] = 8
    return config


    # config['id'] = 6
    # config['type'] = ['prediction']
    # config['data'] = benchmark.balancescale_dataset()
    # config['eps'] = [0.4,0.6,0.8,1.0,2.0]
    # config['depth'] = [2]
    # config['n_trials'] = 5
    # config['n_estimators']=[16]
    # config['prediction method'] = ['subsample']
    # config['alg']=['ID3']
    # config['output']='temp/balancescale_id6.csv'
    # config['random_state'] = None
    # config['n_jobs'] = 8

    # return config


def main():
    pass
    
if __name__ == "__main__":
    main()