
import sklearn
from sklearn import *
import argparse
import pandas as pd
import os
import numpy as np
import time

parser = argparse.ArgumentParser(description="script gerad_experiments")
parser.add_argument('-d_p', '--data_path',    type=str, required=False, help='Path containing the data')
parser.add_argument('-r_p', '--results_path', type=str, required=False, help='Path of the folder where to save results')
parser.add_argument('-p', '--preprocess',     type=str, required=False, help='Preprocess to choose: "normalize" only available')
parser.add_argument('-a','--alphas',          nargs='+', help='List of regularization alphas to test', required=True)
args = parser.parse_args()

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

if __name__ == "__main__":
    t0 = time.time()
    name = "gerard_script"

    ##### Argparsing ###############################
    if args.data_path :
        data_path    = args.data_path 
    else:
        data_path    = "./data/boston_house_prices.csv"

    if args.results_path:
        results_path = args.results_path
    else:
        results_path = "./results"

    if args.preprocess:
        if args.preprocess == "normalize":
            # Do not normalize like this! (it's just an example)
            preprocess =  lambda x: (x-x.mean())/x.std()
        else:
            raise Exception('Preprocess {} not known!'.format(args.preprocess))
    else:
        preprocess = None

    alphas       = args.alphas
    alphas       = [float(x) for x in alphas]
    ###############################################

    df = pd.read_csv(data_path, header=1)

    df_Y = df[["MEDV"]]
    df_X = df[set(df.columns) - set("MEDV")]

    df_X_train, df_X_test, df_Y_train, df_Y_test = sklearn.model_selection.train_test_split(df_X, df_Y)
        
    # stefan likes linear regressors so he tests one
    y_train = df_Y_train.values.flatten()
    y_test = df_Y_test.values.flatten()

    if preprocess:
        df_X_train =preprocess(df_X_train)
        df_X_test = preprocess(df_X_test) 

    results_train = {}
    results_test = {}

    for c in alphas:
        l = sklearn.linear_model.Lasso(alpha=c, tol=0.001)
        l.fit(df_X_train, y_train)

        y_hat_train =  l.predict(df_X_train)
        mse_train = sklearn.metrics.mean_squared_error(y_train, y_hat_train)
        print(" alpha={}, mse train : {}".format(c,mse_train))
        results_train[c] = mse_train

        y_hat_test =  l.predict(df_X_test)
        mse_test = sklearn.metrics.mean_squared_error(y_test, y_hat_test)
        print(" alpha={}, mse test: {}".format(c, mse_test))  
        results_test[c] = mse_test


    df_results = pd.DataFrame({"train":results_train, "test": results_test})
    
    if os.path.exists(results_path)==False:
        os.mkdir(results_path)

    results_file = os.path.join(results_path, name+".csv")
    df_results["alpha"] = df_results.index
    df_results.to_csv(results_file, index=False)
    print("\n\tScript terminated, results written to {} file".format(results_file))
    print("\tTime needed:  {} minutes".format((time.time()- t0)/60))


