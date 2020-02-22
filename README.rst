sklearn2vantage
==================

sklearn2vantage is a Python module for converting sklearn model to Teradata
Vantage model table.

This module has 2 feature. One is converting scikit-learn model to Teradata
Vantage model and another is uploading pandas dataframe to Teradata.

Installation
----------------

Dependencies
~~~~~~~~~~~~~~~~

sklearn2vantage requires:

- Python
- NumPy
- pandas
- SQLAlchemy
- scikit-learn
- paramiko
- scp
- teradata
- sqlalchemy-teradata
- teradatasql
- teradatasqlalchemy

Supported model
~~~~~~~~~~~~~~~~~
Following models are supported.

====================== =====================
scikit-learn           Teradata Vantage
====================== =====================
RandomForestClassifier DecisionForestPredict
RandomForestRegressor  DecisionForestPredict
GradientBoostRegressor DecisionForestPredict
LinearRegression       GLMPredict
Lasso                  GLMPredict
Ridge                  GLMPredict
Linear                 GLMPredict
LogisticRegression     GLMPredict
GaussianNB             NaiveBayesPredict
CategoricalNB          NaiveBayesPredict
DecisionTreeClassifier DecisionTreePredict
DecusionTreeRegressor  DecisionTreePredict
====================== =====================

Some models in statsmodels are also supported.

====================== =====================
statsmodels            Teradata Vantage
====================== =====================
Logit                  GLMPredict
OLS                    GLMPredict
====================== =====================

User installation
~~~~~~~~~~~~~~~~~
::

  pip install sklearn2vantage

or ::

  conda install sklearn2vantage -c temporary-recipes

Example: conveting model
~~~~~~~~~~~~~~~~~~~~~~~~
::

  import sklearn2vantage as s2v
  import pandas as pd
  from sqlalchemy import create_engine
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier

  engine = create_engine("teradata://dbc:dbc@173.168.56.128:1025/tdwork")

  df = pd.read_sql_query("select * from some_data sample 50000", engine)
  X = df.drop("target", axis=1)
  y = df.target
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

  rf_clf = RandomForestClassifier()
  rf_clf.fit(X_train, y_train)

  rf_clf_table = \
    s2v.make_model_table_forest(rf_clf, X_train.columns,
                                ['setosa', 'versicolor', 'virginica'])

  s2v.load_model_forest(rf_clf_table, engine, "rf_clf_table")
  pd.read_sql_query("""
    select * from DecisionForestPredict (
      on iris partition by any
      on rf_clf_table as ModelTable DIMENSION
      USING
      NumerixInputs ('sepal_length', 'sepal_width',
                    'petal_length', 'petal_width')
      IdColumn ('id')
      Accumulate ('species')
      Detailed ('false')
  ) as dt""", engine)

For further usage, please see `HowToUse.ipynb <https://github.com/Yukimura66/sklearn2vantage/blob/master/HowToUse.ipynb>`_.

Example: data loading
~~~~~~~~~~~~~~~~~~~~~
::

  import pandas as pd
  import sklearn2vantage as s2v
  from sqlalchemy import create_engine
  engine = create_engine("teradata://dbc:dbc@173.168.56.128:1025/tdwork")
  df_titanic = pd.read_csv("titanic/train.csv").set_index("PassengerId")
  s2v.tdload_df(df_titanic, engine, tablename="titanic_train",
                ifExists="replace", ssh_ip="173.168.56.128",
                ssh_username="root", ssh_password="root")

For further usage, please see `HowToUseDataloader.ipynb <https://github.com/Yukimura66/sklearn2vantage/blob/master/HowToUseDataloader.ipynb>`_.
