import pandas as pd

from DesicionTree.DesicionTree import DecisionTree

# Caso AND
X = pd.DataFrame({
    "A": [0, 0, 1, 1],
    "B": [0, 1, 0, 1]
})
Y = pd.Series([0, 0, 0, 1])

tree = DecisionTree(max_depth=3, min_categories=1)
tree.train(X, Y,True,True)

Yp = tree.predict(X)
print("Métrica:", tree.metric(Y, Yp))
print("Estructura:", tree.to_string())
print("Reglas:", tree.rules())
print("Profundidad:", tree.depth())