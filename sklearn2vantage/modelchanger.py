import numpy as np
import pandas as pd
from sqlalchemy import Column, Integer, Float, String
import itertools
import sklearn.tree
from sqlalchemy_teradata.types import CLOB
from . import dataloader


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
    dataloader.dropIfExists(table_name, engine.url.database, engine)
    dtype_glm_model = {"attribute": Integer, "predictor": String(length=30),
                       "category": String(length=30), "estimate": Float,
                       "family": String(length=30)}
    df_model.to_sql(name=table_name, con=engine,
                    index=False,
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
    dataloader.dropIfExists(table_name, engine.url.database, engine)
    df_model.to_sql(name=table_name, con=engine,
                    index=False,
                    dtype=dtype_glm_model)


def make_model_table_tree(model, feature_names, isRegression=False):
    id_trees = calcIdTrees(model)
    n_node = len(model.tree_.children_left)
    is_split_node = model.tree_.children_left != -1
    n_split_node = sum(is_split_node)
    children_left = model.tree_.children_left[is_split_node]
    children_right = model.tree_.children_right[is_split_node]
    size = model.tree_.n_node_samples
    features = feature_names[model.tree_.feature]

    if isRegression:
        labels = np.array(
            [f"{val:.10f}" for val in model.tree_.value.squeeze()])
        majorvotes = probs = np.repeat(None, n_node)
        left_label_prob = right_label_prob = prob_label_order \
            = [None] * n_split_node
    else:
        votes = model.tree_.value.squeeze()
        labels = np.apply_along_axis(lambda x: model.classes_[np.argmax(x)],
                                     1, votes)
        majorvotes = votes.max(axis=1)
        probs = votes/votes.sum(axis=1).reshape(-1, 1)
        left_label_prob = [",".join([f"{val:.5f}" for val in val])
                           for val in probs[children_left]]
        right_label_prob = [",".join([f"{val:.5f}" for val in val])
                            for val in probs[children_right]]
        prob_label_order = [",".join(list(model.classes_.astype(str)))] \
            * n_split_node

    tree_table = pd.DataFrame(
        {"node_id": id_trees[is_split_node],
         "node_size": size[is_split_node],
         "node_gini": model.tree_.impurity[is_split_node],
         "node_entropy": np.repeat(None, n_split_node),
         "node_chisq_pv": np.repeat(None, n_split_node),
         "node_label": labels[is_split_node],
         "node_majorvotes": majorvotes[is_split_node],
         "split_value": model.tree_.threshold[is_split_node],
         "split_gini": np.repeat(None, n_split_node),
         "split_entropy": np.repeat(None, n_split_node),
         "split_chisq_pv": np.repeat(None, n_split_node),
         "left_id": id_trees[children_left],
         "left_size": size[children_left],
         "left_label": labels[children_left],
         "left_majorvotes": majorvotes[children_left],
         "right_id": id_trees[children_right],
         "right_size": size[children_right],
         "right_label": labels[children_right],
         "right_majorvotes": np.repeat(None, n_split_node),
         "left_bucket": np.repeat("", n_split_node),
         "right_bucket": np.repeat("", n_split_node),
         "left_label_probdist": left_label_prob,
         "right_label_probdist": right_label_prob,
         "prob_label_order": prob_label_order,
         "attribute": list(features[is_split_node])})

    return tree_table


def load_model_tree(df_model, engine, table_name):
    def max_len(x):
        return int(np.nan_to_num(df_model[x].str.len().max(), nan=1))
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
    dataloader.dropIfExists(table_name, engine.url.database, engine)
    df_model.to_sql(name=table_name, con=engine,
                    index=False,
                    dtype=dtype_tree_model)


def dict2json(obj: dict) -> str:
    json_str = (str(obj).replace("\\", "")
                .replace("\'", '\"').replace('"{', "{")
                .replace('}"', "}"))
    return json_str


def makeTreeJson(tree: sklearn.tree._tree.Tree, idx_node: int,
                 features: list, isRegression: bool = False,
                 classes: list = None, depth: int = 0,
                 coef: float = 1.0, const: float = 0.0) -> str:
    if isRegression and "mse" not in tree.criterion:
        raise ValueError("criterion must be 'mse'")

    id_trees = calcIdTrees(tree)
    type_str = "REGRESSION" if isRegression else "CLASSIFICATION"

    tree_dict = {"size_": tree.tree_.n_node_samples[idx_node],
                 "id_": id_trees[idx_node]}
    if isRegression:
        tree_dict.update({
            "sum_": (tree.tree_.value.squeeze()[idx_node] * coef + const)
            * tree.tree_.n_node_samples[idx_node],
            "sumSq_": (tree.tree_.impurity[idx_node] * coef**2
                       + (tree.tree_.value.squeeze()[idx_node] * coef
                          + const)**2)
            * tree.tree_.n_node_samples[idx_node]
        })

    if tree.tree_.children_left[idx_node] != -1:  # split node
        tree_dict.update(
            {"maxDepth_": tree.tree_.max_depth - depth,
             "split_": {
                 "splitValue_": tree.tree_.threshold[idx_node],
                 "attr_": features[tree.tree_.feature[idx_node]],
                 "score_": tree.tree_.impurity[idx_node],
                 "type_": f"{type_str}_NUMERIC_SPLIT",
                 "leftNodeSize_":
                     tree.tree_.n_node_samples[
                     tree.tree_.children_left[idx_node]
                 ],
                 "rightNodeSize_":
                     tree.tree_.n_node_samples[
                     tree.tree_.children_right[idx_node]
                 ],
                 "scoreImprove_": calcScoreImprove(tree, idx_node)},
             "leftChild_": makeTreeJson(
                 tree, tree.tree_.children_left[idx_node], features,
                 isRegression, classes, depth+1, coef, const),
             "rightChild_": makeTreeJson(
                 tree, tree.tree_.children_right[idx_node], features,
                 isRegression, classes, depth+1, coef, const),
             "nodeType_": f"{type_str}_NODE"
             }
        )
        if not isRegression:
            tree_dict.update({
                "responseCounts_":
                    {str(class_name): int(n_class_samp)
                        for class_name, n_class_samp
                        in zip(classes, tree.tree_.value[idx_node].squeeze())}
            })

    else:  # leaf node
        tree_dict.update(
            {"maxDepth_": 0,
             "nodeType_": f"{type_str}_LEAF"}
        )
        if not isRegression:
            tree_dict.update({
                "label_": str(classes[
                    int(tree.classes_[
                        tree.tree_.value[idx_node].argmax()])
                ])
            })

    return dict2json(tree_dict)


def getNodeDict(tree: sklearn.tree._tree.Tree,
                idx_node: int, new_idx_node: int) -> dict:
    nodeDict = {}
    left_idx_node = tree.tree_.children_left[idx_node]
    right_idx_node = tree.tree_.children_right[idx_node]
    nodeDict[left_idx_node] = new_idx_node * 2 + 1
    nodeDict[right_idx_node] = new_idx_node * 2 + 2
    split_nodes = np.where(tree.tree_.children_left != -1)[0]
    if left_idx_node in split_nodes:
        nodeDict.update(
            getNodeDict(tree, left_idx_node, new_idx_node * 2 + 1)
        )
    if right_idx_node in split_nodes:
        nodeDict.update(
            getNodeDict(tree, right_idx_node, new_idx_node * 2 + 2)
        )
    return nodeDict


def calcIdTrees(tree: sklearn.tree._tree.Tree) -> np.ndarray:
    nodeDict = getNodeDict(tree, 0, 0)
    nodeDict.update({0: 0})
    nodeDict = sorted(nodeDict.items())
    return np.array([x[1] for x in nodeDict])


def calcScoreImprove(tree, idx_node) -> float:
    n_sample = tree.tree_.n_node_samples[idx_node]
    idx_left = tree.tree_.children_left[idx_node]
    idx_right = tree.tree_.children_right[idx_node]
    n_sample_left = tree.tree_.n_node_samples[idx_left]
    n_sample_right = tree.tree_.n_node_samples[idx_right]
    impurity_left = tree.tree_.impurity[idx_left]
    impurity_right = tree.tree_.impurity[idx_right]
    impurity_after = (n_sample_left / n_sample * impurity_left
                      + n_sample_right / n_sample * impurity_right)
    return tree.tree_.impurity[idx_node] - impurity_after


def make_model_table_forest(model, feature_names, classes: list = None,
                            isRegression=False):
    # Gradient Boost or not( = Random Forest)
    if isinstance(model, sklearn.ensemble._gb.GradientBoostingRegressor):
        estimators = model.estimators_.squeeze().tolist()
        coef = model.learning_rate * model.n_estimators
        const = model.init_.constant_.item()
    else:
        estimators = model.estimators_
        coef = 1.0
        const = 0.0

    forest_table = pd.DataFrame(
        {"worker_ip": ["100.0.0.0"] * model.n_estimators,
         "task_index": [0] * model.n_estimators,
         "tree_num": np.arange(model.n_estimators),
         "tree": [makeTreeJson(a_tree, 0, feature_names, isRegression,
                               classes, 0, coef, const)
                  for a_tree in estimators]
         }
    )
    return forest_table


def load_model_forest(df_model, engine, table_name):
    def max_len(x): return df_model[x].str.len().max()
    dtype_tree_model = {"worker_ip": String(max_len("worker_ip")),
                        "task_index": Integer,
                        "tree_num": Integer,
                        "tree": CLOB}
    dataloader.dropIfExists(table_name, engine.url.database, engine)
    df_model.to_sql(name=table_name, con=engine,
                    index=False,
                    dtype=dtype_tree_model)
