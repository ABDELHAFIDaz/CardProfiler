import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

COLS = ["BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
        "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]


def load_raw(path):
    return pd.read_csv(path)


def clean(df):
    return df.dropna(subset=["CREDIT_LIMIT"]).drop_duplicates()


def prepare_clustering(df_clean):
    df_prep = df_clean.copy()
    for col in COLS:
        df_prep[col] = np.log1p(df_prep[col])
    df_prep[COLS] = StandardScaler().fit_transform(df_prep[COLS])
    return df_prep