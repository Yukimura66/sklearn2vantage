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


def load_model_glm(df_model, engine, table_name):
    dtype_glm_model = {"attribute": Integer, "predictor": String(length=30),
                       "category": String(length=30), "estimate": Float,
                       "family": String(length=30)}
    df_model.to_sql(name=table_name, con=engine,
                    index=False, if_exists="replace",
                    dtype=dtype_glm_model)


def make_model_table_gnb(model, feature_names):
    model_dict = [[{"class_nb": str(class_nb), "variable": feature,
                    "type_nb": "NUMERIC", "category": None,
                    "cnt": int(cnt), "sum_nb": cnt * theta,
                    "sumSq": sigma**2 * cnt, "totalCnt": int(cnt)}
                   for class_nb, cnt, theta, sigma
                   in zip(model.classes_, model.class_count_,
                          model.theta_[:, i], model.sigma_[:, i])]
                  for i, feature in enumerate(feature_names)]
    model_dict = list(itertools.chain.from_iterable(model_dict))
    return pd.DataFrame(model_dict)


def make_model_table_cnb(model, category_dict):
    model_dict = [[[{"class_nb": str(class_nb), "variable": feature,
                     "type_nb": "CATEGORICAL",
                     "category": category,
                     "cnt": int(cnt_category), "sum_nb": None,
                     "sumSq": None, "totalCnt": int(cnt_class)}
                    for category, cnt_category
                    in zip(category_dict[feature], cnt_categories)]
                   for class_nb, cnt_class, cnt_categories
                   in zip(model.classes_, model.class_count_,
                          cnt_category_class)]
                  for feature, cnt_category_class
                  in zip(category_dict.keys(), model.category_count_)]
    model_dict = list(itertools.chain.from_iterable(model_dict))
    model_dict = list(itertools.chain.from_iterable(model_dict))
    return pd.DataFrame(model_dict)


def load_model_nb(df_model, engine, table_name):
    def max_len(x):
        return int(np.nan_to_num(df_model[x].str.len().max(), nan=5))
    dtype_glm_model = {"class_nb": String(length=max_len("class_nb")),
                       "variable": String(length=max_len("variable")),
                       "type_nb": String(length=30),
                       "category": String(length=max_len("category")),
                       "cnt": Integer, "sum_nb": Float,
                       "sumSq": Float, "totalCnt": Integer}
    df_model.to_sql(name=table_name, con=engine,
                    index=False, if_exists="replace",
                    dtype=dtype_glm_model)


def make_model_table_tree(model, feature_names):
    is_split_node = model.tree_.children_left != -1
    n_split_node = is_split_node.sum()
    null_col = np.repeat(None, n_split_node)
    blank_col = np.repeat("", n_split_node)
    children_left = model.tree_.children_left[is_split_node]
    children_right = model.tree_.children_right[is_split_node]
    def func(x): return model.classes_[np.argmax(x)]
    votes = model.tree_.value.squeeze()
    labels = np.apply_along_axis(func, 1, votes)
    majorvotes = votes.max(axis=1)
    size = model.tree_.n_node_samples
    features = feature_names[model.tree_.feature]
    probs = votes/votes.sum(axis=1).reshape(-1, 1)

    tree_table = pd.DataFrame({"node_id": np.where(is_split_node)[0],
                               "node_size": size[is_split_node],
                               "node_gini": (model.tree_
                                                  .impurity[is_split_node]),
                               "node_entropy": null_col,
                               "node_chisq_pv": null_col,
                               "node_label": labels[is_split_node],
                               "node_majorvotes": majorvotes[is_split_node],
                               "split_value": (model.tree_
                                                    .threshold[is_split_node]),
                               "split_gini": null_col,
                               "split_entropy": null_col,
                               "split_chisq_pv": null_col,
                               "left_id": children_left,
                               "left_size": size[children_left],
                               "left_label": labels[children_left],
                               "left_majorvotes": majorvotes[children_left],
                               "right_id": children_right,
                               "right_size": size[children_right],
                               "right_label": labels[children_right],
                               "right_majorvotes": null_col,
                               "left_bucket": blank_col,
                               "right_bucket": blank_col,
                               "left_label_probdist":
                                   [",".join([f"{val:.5f}" for val in val])
                                       for val in probs[children_left]],
                               "right_label_probdist":
                                   [",".join([f"{val:.5f}" for val in val])
                                       for val in probs[children_right]],
                               "prob_label_order":
                                   [",".join(list(model.classes_.astype(str)))]
                               * n_split_node,
                               "attribute": list(features[is_split_node])})

    split_nodes = list(np.where(is_split_node)[0])
    children_left_new = children_left.copy()
    children_right_new = children_right.copy()
    replace_dict = {}

    for i in range(len(split_nodes)):
        split_node = split_nodes.pop(0)
        replace_dict[children_left[i]] = split_node * 2 + 1
        replace_dict[children_right[i]] = split_node * 2 + 2
        split_nodes = [replace_dict[val] if val in replace_dict else val
                       for val in split_nodes]

    tree_table[["node_id", "left_id", "right_id"]] = \
        tree_table[["node_id", "left_id", "right_id"]].replace(replace_dict)

    return tree_table


def load_model_tree(df_model, engine, table_name):
    def max_len(x): return df_model[x].str.len().max()
    dtype_tree_model = {"node_id": Integer, "node_size": Integer,
                        "node_gini": Float, "node_entropy": Float,
                        "node_chisq_pv": Float,
                        "node_label": String(length=30),
                        "node_majorvotes": Integer, "split_value": Float,
                        "split_gini": Float, "split_entropy": Float,
                        "split_chisq_pv": Float, "left_id": Integer,
                        "left_size": Integer, "left_label": String(length=30),
                        "left_majorvotes": Integer, "right_id": Integer,
                        "right_size": Integer,
                        "right_label": String(length=30),
                        "right_majorvotes": Integer,
                        "left_bucket": String(length=30),
                        "right_bucket": String(length=30),
                        "left_label_probdist":
                            String(length=max_len("left_label_probdist")),
                        "right_label_probdist":
                            String(length=max_len("right_label_probdist")),
                        "prob_label_order":
                            String(length=max_len("prob_label_order")),
                        "attribute": String(length=max_len("attribute"))}
    df_model.to_sql(name=table_name, con=engine,
                    index=False, if_exists="replace",
                    dtype=dtype_tree_model)
