import numpy as np
import pandas as pd
import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes
import cvxpy as cvx


def subsample(X,y,sub_type):
    df = pd.concat([X,y],axis=1)
    label = list(df)[-1]

    if sub_type=='random':
        df_bad = df[(df[label] == 1)]
        df_good = df[(df[label] == 0)]

        df_sub = pd.concat([df_good.sample(len(df_bad.index)),df_bad])
        X_new, y_new = df_sub.drop(label,axis=1), df_sub[[label]]

    elif sub_type=='optimized':
        df_new = df.copy()
        cats = df.columns[df.dtypes=='category']

        for i, col in enumerate(cats):
            df_new[col] = df[col].astype('int64')
        df_m = df_new[cats].values

        n = df_m.shape[0]
        k = df[label].value_counts()[1]*2
        v = df_m.shape[1]
        p_t = np.full([v], 0.5)

        a = cvx.Variable(n)
        constraints = [
            a <= 1,
            a >= 0,
            cvx.sum(a) == k,
        ]
        emperical_distribution = (df_m.T @ a) / k
        least_squares = cvx.Minimize(cvx.sum_squares(emperical_distribution - p_t) / v)
        prob = cvx.Problem(least_squares, constraints)
        prob.solve()

        df_sub = df.iloc[np.argsort(np.array(a.value).ravel())[-k:]]
        X_new, y_new = df_sub.drop(label,axis=1), df_sub[[label]]
    else:
        X_new, y_new = X,y

    return X_new, y_new
