"""In this module, we ask you to define your pricing model, in Python."""

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
# NOTE THAT ANY TENSORFLOW VERSION HAS TO BE LOWER THAN 2.4. So 2.3XXX would work.

# TODO: import your modules here.
# Don't forget to add them to requirements.txt before submitting.



# Feel free to create any number of other functions, constants and classes to use
# in your model (e.g., a preprocessing function).

def label_encoding(merged, col):
    ls = list(set(merged[col].unique()))
    item_cnt = [0] + [i for i in range(1,len(ls))]
    dc = dict(zip(ls, item_cnt))
    merged[col] = merged[col].map(dc)
    print("Label Encoding of '" + str(col) + "' column succeeded.")


def fit_model(X_raw, y_raw):
    """Model training function: given training data (X_raw, y_raw), train this pricing model.

    Parameters
    ----------
    X_raw : Pandas dataframe, with the columns described in the data dictionary.
        Each row is a different contract. This data has not been processed.
    y_raw : a Numpy array, with the value of the claims, in the same order as contracts in X_raw.
        A one dimensional array, with values either 0 (most entries) or >0.

    Returns
    -------
    self: this instance of the fitted model. This can be anything, as long as it is compatible
        with your prediction methods.

    """
    X_raw.drop("id_policy",axis = 1,inplace = True)
    label_encoding_cols = ['pol_coverage','pol_pay_freq','pol_payd','pol_usage',
                      'drv_sex1','drv_drv2','drv_sex2','vh_make_model',
                      'vh_fuel','vh_type']
    for col in label_encoding_cols:
        label_encoding(X_raw, col)
    print("================")
    print("finished!")
    
    x_trn = X_raw[X_raw["year"]<=3]
    y_trn = y_raw[X_raw["year"]<=3]
    x_val = X_raw[X_raw["year"]>=4]
    y_val = y_raw[X_raw["year"]>=4]
    
    lgb_train = lgb.Dataset(x_trn, y_trn)
    lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)
    
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression', # 目的 : 回帰  
        'metric': {'rmse'}, # 評価指標 : rsme(平均二乗誤差の平方根) 
    
        'learning_rate': 0.01,
        'num_leaves': 100,
        'random_state':123,
        #'max_dapth':5,
        'bagging_fraction':0.9,
        'feature_fraction':1.0,
        'min_data_in_leaf': 300,
        'lambda_l1':1,
        'lambda_l2':2.5,
        'num_iteration': 1000, #1000回学習
    }

    # モデルの学習
    model = lgb.train(params, # パラメータ
            train_set=lgb_train, # トレーニングデータの指定
            valid_sets=lgb_eval, # 検証データの指定
            early_stopping_rounds=30,# 100回ごとに検証精度の改善を検討　→ 精度が改善しないなら学習を終了(過学習に陥るのを防ぐ)
            verbose_eval = 10
                 )
    # TODO: train your model here.

    return model  # By default, training a model that returns a mean value (a mean model).



def predict_expected_claim(model, X_raw):
    """Model prediction function: predicts the expected claim based on the pricing model.

    This functions estimates the expected claim made by a contract (typically, as the product
    of the probability of having a claim multiplied by the expected cost of a claim if it occurs),
    for each contract in the dataset X_raw.

    This is the function used in the RMSE leaderboard, and hence the output should be as close
    as possible to the expected cost of a contract.

    Parameters
    ----------
    model: a Python object that describes your model. This can be anything, as long
        as it is consistent with what `fit` outpurs.
    X_raw : Pandas dataframe, with the columns described in the data dictionary.
        Each row is a different contract. This data has not been processed.

    Returns
    -------
    avg_claims: a one-dimensional Numpy array of the same length as X_raw, with one
        expected claim per contract (in same order). These expected claims must be POSITIVE (>0).
    """

    # TODO: estimate the expected claim of every contract.
    
    X_raw.drop("id_policy",axis = 1,inplace = True)
    label_encoding_cols = ['pol_coverage','pol_pay_freq','pol_payd','pol_usage',
                      'drv_sex1','drv_drv2','drv_sex2','vh_make_model',
                      'vh_fuel','vh_type']
    for col in label_encoding_cols:
        label_encoding(X_raw, col)
    print("================")
    print("finished!")
    
    
    
    
    
    return model.predict(X_raw, num_iteration=model.best_iteration)  # Estimate that each contract will cost 114 (this is the naive mean model). You should change this!



def predict_premium(model, X_raw):
    X_raw.drop("id_policy",axis = 1,inplace = True)
    label_encoding_cols = ['pol_coverage','pol_pay_freq','pol_payd','pol_usage',
                      'drv_sex1','drv_drv2','drv_sex2','vh_make_model',
                      'vh_fuel','vh_type']
    for col in label_encoding_cols:
        label_encoding(X_raw, col)
    print("================")
    print("finished!")

    return model.predict(X_raw, num_iteration=model.best_iteration) * 1.89



def save_model(model):
	"""Saves this trained model to a file.

	This is used to save the model after training, so that it can be used for prediction later.

	Do not touch this unless necessary (if you need specific features). If you do, do not
	 forget to update the load_model method to be compatible.

	Parameters
	----------
	model: a Python object that describes your model. This can be anything, as long
	    as it is consistent with what `fit` outpurs."""

	with open('trained_model.pickle', 'wb') as target:
		pickle.dump(model, target)




def load_model():
	"""Load a saved trained model from the file.

	   This is called by the server to evaluate your submission on hidden data.
	   Only modify this *if* you modified save_model."""

	with open('trained_model.pickle', 'rb') as target:
		trained_model = pickle.load(target)
	return trained_model
