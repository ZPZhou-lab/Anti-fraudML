import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

def aggregate_trade(data : pd.DataFrame, feats : pd.DataFrame):
    df = data.copy()
    # 交易金额占账户余额的比例
    # 绝对交易金额
    df["money_abs"] = df["jyje"].abs()
    df["balance_ratio"] = df["money_abs"] / df["zhye"]
    df["balance_ratio"] = df["balance_ratio"].replace(np.inf,-1)

    # 交易总次数
    feats["trade_cnts"] = df.groupby("zhdh")["jdbj"].count()
    # 入账比例
    feats["in_ratio"] = df.groupby("zhdh")["jdbj"].sum() / feats["trade_cnts"]

    # 聚合函数
    agg_func = {
        "jyje": ["min","max","mean","std",np.ptp],
        "money_abs": ["min","max","mean","std",np.ptp],
        "balance_ratio": ["min","max","mean","std",np.ptp],
        "zhye": ["min","max","mean","std",np.ptp],
        "dfmccd": ["min","max","mean","std",np.ptp],
        "dfzh": ["nunique"],
        "dfhh": ["nunique"],
        "jyqd": ["nunique"],
        "zydh": ["nunique"]
    }

    agg_df_0 = df[df["jdbj"] == 0].groupby("zhdh").agg(agg_func).reset_index()
    agg_df_0.set_index("zhdh",inplace=True)
    agg_df_0.columns = ["trade_out_" + f[0] + "_" + f[1] for f in agg_df_0.columns]
    feats = feats.merge(right=agg_df_0,left_index=True,right_index=True)

    agg_df_1 = df[df["jdbj"] == 1].groupby("zhdh").agg(agg_func).reset_index()
    agg_df_1.set_index("zhdh",inplace=True)
    agg_df_1.columns = ["trade_in_" + f[0] + "_" + f[1] for f in agg_df_1.columns]
    feats = feats.merge(right=agg_df_1,left_index=True,right_index=True)

    return feats

def aggregate_trade_people(data : pd.DataFrame, feats : pd.DataFrame):
    # 每个用户与不同账户的交易次数
    trade_people_cnts = data.groupby(by=["zhdh","dfzh"])["jdbj"].count()
    # 与多少人发生交易
    feats["trade_people_cnts"] = trade_people_cnts.groupby("zhdh").count()
    # 与同一用户交易的次数最大值，平均值和波动性
    feats["trade_pelple_cnts_max"] = trade_people_cnts.groupby("zhdh").max()
    feats["trade_pelple_cnts_avg"] = trade_people_cnts.groupby("zhdh").mean()
    feats["trade_pelple_cnts_std"] = trade_people_cnts.groupby("zhdh").std()

    return feats

def aggregate_trade_bank(data : pd.DataFrame, feats : pd.DataFrame):
    # 每个用户在各个银行交易次数
    trade_bank_cnts = data.groupby(by=["zhdh","dfhh"])["jdbj"].count()
    # 与多少银行发生交易
    feats["trade_bank_cnts"] = trade_bank_cnts.groupby("zhdh").count()
    # 与同一银行交易的次数最大值，平均值和波动性
    feats["trade_bank_cnts_max"] = trade_bank_cnts.groupby("zhdh").max()
    feats["trade_bank_cnts_avg"] = trade_bank_cnts.groupby("zhdh").mean()
    feats["trade_bank_cnts_std"] = trade_bank_cnts.groupby("zhdh").std()

    return feats

def aggregate_trade_days(data : pd.DataFrame, feats : pd.DataFrame):
    # 每个用户每日交易次数
    trade_cnts_days = data.groupby(by=["zhdh","jyrq"])["jdbj"].count()
    # 交易天数
    feats["trade_day_cnts"] = trade_cnts_days.groupby("zhdh").count()
    # 单日最大交易次数，平均次数，和波动性
    feats["trade_day_cnts_max"] = trade_cnts_days.groupby("zhdh").max()
    feats["trade_day_cnts_avg"] = trade_cnts_days.groupby("zhdh").mean()
    feats["trade_day_cnts_std"] = trade_cnts_days.groupby("zhdh").std()

    return feats

def aggregate_trade_channel(data : pd.DataFrame, feats : pd.DataFrame):
    # 每个用户每个渠道交易次数
    trade_cnts_channels = data.groupby(by=["zhdh","jyqd"])["jdbj"].count()
    # 交易渠道数
    feats["trade_channel_cnts"] = trade_cnts_channels.groupby("zhdh").count()
    # 渠道最大交易次数，平均交易次数和波动性
    feats["trade_channel_cnts_max"] = trade_cnts_channels.groupby("zhdh").max()
    feats["trade_channel_cnts_avg"] = trade_cnts_channels.groupby("zhdh").mean()
    feats["trade_channel_cnts_std"] = trade_cnts_channels.groupby("zhdh").std()

    return feats

def aggregate_trade_summary(data : pd.DataFrame, feats : pd.DataFrame):
    # 每个用户每种摘要交易次数
    trade_cnts_summary = data.groupby(by=["zhdh","zydh"])["jdbj"].count()
    # 交易摘要数
    feats["trade_summary_cnts"] = trade_cnts_summary.groupby("zhdh").count()
    # 不同摘要最大交易次数，平均交易次数和波动性
    feats["trade_summary_cnts_max"] = trade_cnts_summary.groupby("zhdh").max()
    feats["trade_summary_cnts_avg"] = trade_cnts_summary.groupby("zhdh").mean()
    feats["trade_summary_cnts_std"] = trade_cnts_summary.groupby("zhdh").std()

    return feats

def aggregate_trade_label_money(data : pd.DataFrame, feats : pd.DataFrame):
    df = data.copy()
    df["money_abs"] = df["jyje"].abs()
    df = df.groupby(by=["zhdh","jdbj"])

    df_temp = pd.pivot(df["money_abs"].max().reset_index(),index="zhdh",columns="jdbj")
    df_temp.columns = ["trade_money_abs_max_0","trade_money_abs_max_1"]
    df_temp.fillna(0,inplace=True)
    feats = feats.merge(right=df_temp,left_index=True,right_index=True)

    df_temp = pd.pivot(df["money_abs"].mean().reset_index(),index="zhdh",columns="jdbj")
    df_temp.columns = ["trade_money_abs_mean_0","trade_money_abs_mean_1"]
    df_temp.fillna(0,inplace=True)
    feats = feats.merge(right=df_temp,left_index=True,right_index=True)

    df_temp = pd.pivot(df["money_abs"].std().reset_index(),index="zhdh",columns="jdbj")
    df_temp.columns = ["trade_money_abs_std_0","trade_money_abs_std_1"]
    df_temp.fillna(0,inplace=True)
    feats = feats.merge(right=df_temp,left_index=True,right_index=True)

    return feats

def aggregate_trade_period(data : pd.DataFrame, feats : pd.DataFrame, 
                           add_each_period_cnts : bool=True, freq : int=3600):
    """
    Parameters
    ----------
    add_each_period_cnts : bool, default = `True`
        Whether to count number of trades in each period.
    freq : int, default = 3600
        The threshold (seconds) to cut-off the time period in a day.
    """
    df = data.copy()
    df["period"] = data["jysj"].apply(lambda x : pd.to_timedelta(x))
    df["period"] = df["period"].apply(lambda x : int(x.seconds / freq))
    
    trade_cnts_period = df.groupby(by=["zhdh","period"])["jdbj"].count()
    # 交易时间段次数
    feats["trade_period_cnts"] = trade_cnts_period.groupby("zhdh").count()
    # 各时间段最大交易次数，平均交易次数和波动性
    feats["trade_period_cnts_max"] = trade_cnts_period.groupby("zhdh").max()
    feats["trade_period_cnts_avg"] = trade_cnts_period.groupby("zhdh").mean()
    feats["trade_period_cnts_std"] = trade_cnts_period.groupby("zhdh").std()
    
    # 统计每个用户各时间段的交易次数
    if add_each_period_cnts:
        trade_cnts_each_period =\
            pd.pivot(df.groupby(by=["zhdh","period"])["jdbj"].count().reset_index(),
                    index="zhdh",columns="period")
        num_periods = df["period"].nunique()
        trade_cnts_each_period.columns = ["trade_period_%d_cnts"%(i) for i in range(num_periods)]
        trade_cnts_each_period.fillna(0,inplace=True)
        feats = feats.merge(right=trade_cnts_each_period,left_index=True,right_index=True)
    
    return feats

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
    train_df = pd.read_csv("../../data/训练集标签.csv",index_col=0)
    train_df = train_df.merge(right=static_feats,left_index=True,right_index=True)
    train_df = train_df.merge(right=dynamic_feats,left_index=True,right_index=True)
    y_train = train_df.pop("black_flag")
    X_train = train_df

    # 测试集
    test_df = pd.read_csv("../../data/test_dataset.csv",index_col=0)
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