import numpy as np
import pandas as pd
from sqlalchemy import Column, Integer, Float, String
import itertools


def make_model_table_glm(model, feature_names=None, isLogistic=False,
                         isStatsmodels=False):
    family = "LOGISTIC" if isLogistic else "GAUSSIAN"
    if isStatsmodels:
        coefs = model.params.values
        features = model.params.index.values
    else:
        if feature_names is None:
            raise(ValueError,
                  "for sklearn model, feature_names is necessary.")
        coefs = np.hstack([model.intercept_, model.coef_.ravel()])
        features = np.hstack(["(Intercept)", feature_names])
    model_dict = [{"attribute": i, "predictor": feature,
                   "category": None, "estimate": coef,
                   "family": family}
                  for i, (coef, feature) in enumerate(zip(coefs, features))]
    return pd.DataFrame(model_dict)


def load_model_glm(df_model, engine, database_name, table_name):
    dtype_glm_model = {"attribute": Integer, "predictor": String(length=30),
                       "category": String(length=30), "estimate": Float,
                       "family": String(length=30)}
    table_list = pd.read_sql_query("""
        select * from dbc.Tables
        where TableKind = 'T'
            and TableName = '{table_name}'
            and DatabaseName = '{database_name}'
            """.format(table_name=table_name, database_name=database_name),
        engine)
    if len(table_list) > 0:
        engine.execute(f"drop table {table_name}")
    df_model.to_sql(name=table_name, con=engine,
                    index=False,
                    dtype=dtype_glm_model)


def make_model_table_gnb(model, feature_names):
    model_dict = [[{"class_nb": class_nb, "variable": feature,
                    "type_nb": "NUMERIC", "category": None,
                    "cnt": int(cnt), "sum_nb": cnt * theta,
                    "sumSq": sigma**2 * cnt, "totalCnt": int(cnt)}
                   for class_nb, cnt, theta, sigma
                   in zip(model.classes_, model.class_count_, model.theta_[:, i],
                          model.sigma_[:, i])]
                  for i, feature in enumerate(feature_names)]
    model_dict = list(itertools.chain.from_iterable(model_dict))
    return pd.DataFrame(model_dict)


def make_model_table_cnb(model, category_dict):
    model_dict = [[[{"class_nb": class_nb, "variable": feature,
                     "type_nb": "CATEGORICAL",
                     "category": category,
                     "cnt": int(cnt_category), "sum_nb": None,
                     "sumSq": None, "totalCnt": int(cnt_class)}
                    for category, cnt_category
                    in zip(category_dict[feature], cnt_categories)]
                   for class_nb, cnt_class, cnt_categories
                   in zip(model.classes_, model.class_count_, cnt_category_class)]
                  for feature, cnt_category_class
                  in zip(category_dict.keys(), model.category_count_)]
    model_dict = list(itertools.chain.from_iterable(model_dict))
    model_dict = list(itertools.chain.from_iterable(model_dict))
    return pd.DataFrame(model_dict)


def load_model_nb(df_model, engine, database_name, table_name):
    dtype_glm_model = {"class_nb": String(length=30), "variable": String(length=30),
                       "type_nb": String(length=30), "category": String(length=30),
                       "cnt": Integer, "sum_nb": Float,
                       "sumSq": Float, "totalCnt": Integer}
    table_list = pd.read_sql_query("""
        select * from dbc.Tables
        where TableKind = 'T'
            and TableName = '{table_name}'
            and DatabaseName = '{database_name}'
            """.format(table_name=table_name, database_name=database_name),
        engine)
    if len(table_list) > 0:
        engine.execute(f"drop table {table_name}")
    df_model.to_sql(name=table_name, con=engine,
                    index=False,
                    dtype=dtype_glm_model)
