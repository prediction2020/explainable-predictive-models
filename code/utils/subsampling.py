import numpy as np
import pandas as pd



def subsample(X,y,subsampling_type):
    df = pd.concat([X,y],axis=1)
    label = list(df)[-1]

    if subsampling_type=='random':
        df_bad = df[(df[label] == 1)]
        df_good = df[(df[label] == 0)]

        df_sub = pd.concat([df_good.sample(len(df_bad.index)),df_bad])
        X_new, y_new = df_sub.drop(label,axis=1), df_sub[[label]]

    elif subsampling_type=='none':
        X_new, y_new = X,y

    return X_new, y_new
