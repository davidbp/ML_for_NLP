
import sklearn
from sklearn import *
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description="script stefan_model")
parser.add_argument('-d_p', '--data_path',    type=str, required=True, help='Path containing the data')
parser.add_argument('-r_p', '--results_path', type=str, required=True, help='Path of the folder where to save results')
args = parser.parse_args()

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

if __name__ == "__main__":


    ##### Argparsing ###############################
    data_path            = args.data_path 
    results_path = args.results_path
    ###############################################
    
    df = pd.read_csv(data_path, header=1)

    df_Y = df[["MEDV"]]
    df_X = df[set(df.columns) - set("MEDV")]

    df_X_train, df_X_test, df_Y_train, df_Y_test = sklearn.model_selection.train_test_split(df_X, df_Y)
        
    # stefan likes linear regressors so he tests one
    y_train = df_Y_train.values.flatten()
    y_test = df_Y_test.values.flatten()
        
    C = [0.1, 0.2, 0.3]
    results_train = {}
    results_test = {}
    for c in C:
        l = sklearn.neural_network.MLPRegressor(alpha=c, tol=0.001)
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

    df_results["alpha"] = df_results.index
    results_file = os.path.join(results_path, "stefan.csv")
    df_results.to_csv(results_file, index=False)
    print("\n\tScript terminated, results written to {} file".format(results_file))
