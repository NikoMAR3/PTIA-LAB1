import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

columnas = ["age","workclass","fnlwgt","education","education-num",
            "marital-status","occupation","relationship","race","sex",
            "capital-gain","capital-loss","hours-per-week","native-country","income"]

base = os.path.dirname(os.path.abspath(__file__))
ruta_train = os.path.join(base, "..", "..", "Datos", "adult.data")

df_train = pd.read_csv(ruta_train, header=None, names=columnas, na_values=" ?")


# Vista general
print(df_train.shape)
print(df_train.dtypes)
print(df_train.isnull().sum())

# Distribución de clases
df_train["income"].value_counts(normalize=True).plot(kind="bar", title="Distribución de clases")
plt.show()

# Histogramas numéricos
df_train[["age","capital-gain","capital-loss","hours-per-week","education-num"]].hist(figsize=(12,6))
plt.tight_layout()
plt.show()

# Correlación
sns.heatmap(df_train[["age","capital-gain","capital-loss","hours-per-week","education-num"]].corr(), annot=True)
plt.show()