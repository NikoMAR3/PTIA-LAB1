import pandas as pd

from DesicionTree.DesicionTree import DecisionTree

columnas = ["age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

df_train = pd.read_csv("../Datos/adult.data", header=None, names=columnas)
df_test = pd.read_csv("../Datos/adult.test", header=None, names=columnas, skiprows=1)

df_train = pd.read_csv("../Datos/adult.data", header=None, names=columnas)
df_test = pd.read_csv("../Datos/adult.test", header=None, names=columnas, skiprows=1)

# Quedarse solo con categóricas
categoricas = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]

X_train = df_train[categoricas]
Y_train = df_train["income"].apply(lambda x: 1 if ">50K" in x else 0)

X_test = df_test[categoricas]
Y_test = df_test["income"].apply(lambda x: 1 if ">50K" in x else 0)

X_train = X_train.replace(" ?", None).dropna()
Y_train = Y_train[X_train.index]

X_test = X_test.replace(" ?", None).dropna()
Y_test = Y_test[X_test.index]

tree = DecisionTree(max_depth=5, min_categories=1)
tree.train(X_train, Y_train,True,True)

Yp = tree.predict(X_test)
print("Métrica:", tree.metric(Y_test, Yp))
print("Profundidad:", tree.depth())
print("Reglas:", tree.rules())


# Caso AND
X = pd.DataFrame({
    "A": [0, 0, 1, 1],
    "B": [0, 1, 0, 1]
})

# Caso AND
Y = pd.Series([0, 0, 0, 1])

# Caso OR
Y = pd.Series([0, 1, 1, 1])

# Caso XOR
Y = pd.Series([0, 1, 1, 0])

tree = DecisionTree(max_depth=3, min_categories=1)
tree.train(X, Y,True,True)

Yp = tree.predict(X)
print("Métrica:", tree.metric(Y, Yp))
print("Estructura:", tree.to_string())
print("Reglas:", tree.rules())
print("Profundidad:", tree.depth())