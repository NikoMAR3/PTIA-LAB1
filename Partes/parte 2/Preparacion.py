import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

columnas = ["age","workclass","fnlwgt","education","education-num",
            "marital-status","occupation","relationship","race","sex",
            "capital-gain","capital-loss","hours-per-week","native-country","income"]

base = os.path.dirname(os.path.abspath(__file__))
ruta_train = os.path.join(base, "..", "..", "Datos", "adult.data")
ruta_test  = os.path.join(base, "..", "..", "Datos", "adult.test")

df_train = pd.read_csv(ruta_train, header=None, names=columnas, na_values=" ?")
df_test  = pd.read_csv(ruta_test,  header=None, names=columnas, skiprows=1, na_values=" ?")

df_train["split"] = "train"
df_test["split"]  = "test"
df_all = pd.concat([df_train, df_test])

# Limpiar income
df_all["income"] = df_all["income"].str.strip().str.replace(".", "", regex=False)
df_all["Y"] = (df_all["income"] == ">50K").astype(int)

# Eliminar nulos, columnas redundantes
df_all = df_all.dropna()
df_all = df_all.drop(columns=["education", "fnlwgt", "income"])

# Codificar categóricas
categoricas = ["workclass","marital-status","occupation","relationship","race","sex","native-country"]
le = LabelEncoder()
for col in categoricas:
    df_all[col] = le.fit_transform(df_all[col])

# Separar train y test
df_tr = df_all[df_all["split"] == "train"].drop(columns=["split"])
df_te = df_all[df_all["split"] == "test"].drop(columns=["split"])

# Split train/dev
X = df_tr.drop(columns=["Y"])
Y = df_tr["Y"]
X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

X_test = df_te.drop(columns=["Y"])
Y_test = df_te["Y"]

print(f"Train: {X_train.shape}, Dev: {X_dev.shape}, Test: {X_test.shape}")