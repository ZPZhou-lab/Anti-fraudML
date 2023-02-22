import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def merge_feats(static_feats : pd.DataFrame, dynamic_feats : pd.DataFrame):
    """
    Parameters
    ----------
    static_feats : pd.DataFrame
        The static feats constructed. THe index should be set as `user_id`
    dynamic_feats : pd.DataFrame
        The dynamic trade feats construnced. The index should be set with `user_id`
    
    Return
    ----------
    (X_train, y_train, X_test) : tuple
        
    """
    # 训练集
    train_df = pd.read_csv("../data/训练集标签.csv",index_col=0)
    train_df = train_df.merge(right=static_feats,left_index=True,right_index=True)
    train_df = train_df.merge(right=dynamic_feats,left_index=True,right_index=True)
    y_train = train_df.pop("black_flag")
    X_train = train_df

    # 测试集
    test_df = pd.read_csv("../data/test_dataset.csv",index_col=0)
    test_df = test_df.merge(right=static_feats,left_index=True,right_index=True)
    test_df = test_df.merge(right=dynamic_feats,left_index=True,right_index=True)
    X_test = test_df

    return X_train, y_train, X_test

def evaluation_model(model, X_train, y_train, X_valid, y_valid, cols, verbose : bool=False):
    """
    Return
    ----------
    train_f1, valid_f1 : float
        F1 Score on training and validation set.
    feats_imp : pd.Series
        Feature importance.
    """

    train_f1 = f1_score(y_train,model.predict(X_train[cols]))
    valid_f1 = f1_score(y_valid,model.predict(X_valid[cols]))
    if verbose:
        print("f1 score on train: %.4f"%(train_f1))
        print("f1 score on valid: %.4f"%(valid_f1))
    try:
        feats_imp = pd.Series(index=model.feature_names_in_,data=model.feature_importances_)
        feats_imp.sort_values(ascending=False,inplace=True)
    except:
        feats_imp = None
        print("No feature importance mehtod in model.")
    
    return train_f1, valid_f1, feats_imp

def build_pipeline():
    ...