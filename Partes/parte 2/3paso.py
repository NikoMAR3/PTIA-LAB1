import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score

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

df_all["income"] = df_all["income"].str.strip().str.replace(".", "", regex=False)
df_all["Y"] = (df_all["income"] == ">50K").astype(int)
df_all = df_all.dropna()
df_all = df_all.drop(columns=["education", "fnlwgt", "income"])

categoricas = ["workclass","marital-status","occupation","relationship","race","sex","native-country"]
le = LabelEncoder()
for col in categoricas:
    df_all[col] = le.fit_transform(df_all[col])

df_tr = df_all[df_all["split"] == "train"].drop(columns=["split"])
df_te = df_all[df_all["split"] == "test"].drop(columns=["split"])

X = df_tr.drop(columns=["Y"])
Y = df_tr["Y"]
X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
X_test = df_te.drop(columns=["Y"])
Y_test = df_te["Y"]


dt = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt.fit(X_train, Y_train)

print("=== DecisionTree ===")
print(f"F1 train: {f1_score(Y_train, dt.predict(X_train)):.3f}")
print(f"F1 dev:   {f1_score(Y_dev,   dt.predict(X_dev)):.3f}")
print(f"Profundidad: {dt.get_depth()}")
print(f"Hojas:       {dt.get_n_leaves()}")

rf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
rf.fit(X_train, Y_train)

print("\n=== RandomForest ===")
print(f"F1 train: {f1_score(Y_train, rf.predict(X_train)):.3f}")
print(f"F1 dev:   {f1_score(Y_dev,   rf.predict(X_dev)):.3f}")


gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, Y_train)

print("\n=== GradientBoosting ===")
print(f"F1 train: {f1_score(Y_train, gb.predict(X_train)):.3f}")
print(f"F1 dev:   {f1_score(Y_dev,   gb.predict(X_dev)):.3f}")


resultados = {
    "DecisionTree":      f1_score(Y_dev, dt.predict(X_dev)),
    "RandomForest":      f1_score(Y_dev, rf.predict(X_dev)),
    "GradientBoosting":  f1_score(Y_dev, gb.predict(X_dev)),
}
mejor = max(resultados, key=resultados.get)
print(f"\nMejor modelo (dev): {mejor} con F1={resultados[mejor]:.3f}")

modelos = {"DecisionTree": dt, "RandomForest": rf, "GradientBoosting": gb}
Yp_test = modelos[mejor].predict(X_test)
print(f"\n{mejor} - F1 test: {f1_score(Y_test, Yp_test):.3f}")