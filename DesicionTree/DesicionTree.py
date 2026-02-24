import pandas as pd
from pandas import Series

from BaseFunctions.Criterium import Criterium
from BaseFunctions.Metric import Metric


class DecisionTree:
  """ Representa un árbol de decisión
  """

  def __init__(self, max_depth: int, min_categories: int):
      self.max_depth = max_depth
      self.min_categories = min_categories
      self.tree = None
      self.criterium = Criterium.use("entropy")
      self.metric_f1 = Metric.use("F1Score")()

  def metric(self, Y: pd.DataFrame, Yp: pd.DataFrame) -> float:
    """ computa la metrica del modelo a partir de los ejemplares comparando las salidas
    Args:
      Y  : valores de salidas esperadas (etiquetadas)
      Yp  : valores de salidas obtenidas
    Return:
       desempeño del modelo para ejemplares
    """
    return self.metric_f1.value(Yp, Y)

  def predict(self, X: pd.DataFrame) -> Series:
    """ computa una serie de entradas a traves del arbol generando una predicción
    Args:
      X    : valores de características (entradas)
    Return:
     valores de salidas obtenidas
    """
    resultados = []
    for _, fila in X.iterrows():
        nodo = self.tree
        while "hoja" not in nodo:
            atributo = nodo["atributo"]
            valor = fila[atributo]
            if valor not in nodo["hijos"]:
                nodo = list(nodo["hijos"].values())[0]
            else:
                nodo = nodo["hijos"][valor]
        resultados.append(nodo["hoja"])
    return pd.Series(resultados)



  def train(self, X: pd.DataFrame, Y: pd.DataFrame, print_impurity: bool, do_graphic: bool):
    """ construye y entrena el árbol de decisión a partir de unos ejemplares.
    Args:
      X  : valores de características - conjunto de entrenamiento
      Y  : valores de salidas esperadas - conjunto de entrenamiento
      print_impurity : mostrar la impureza del arbol por iteración
      do_graphic: graficar las impurezas por iteración
    """
    self.tree = {"atributo": None, "hijos": {}, "X": X, "Y": Y}
    pila = [(self.tree, X, Y, 0)]

    while pila:
        nodo, X_n, Y_n, depth = pila.pop()

        if depth == self.max_depth or len(Y_n.unique()) <= self.min_categories:
            nodo["hoja"] = Y_n.mode()[0]
            continue

        mejor = max(X_n.columns, key=lambda a: self.criterium.gain(a, X_n, Y_n))
        nodo["atributo"] = mejor

        for valor in X_n[mejor].unique():
            sub_X = X_n[X_n[mejor] == valor]
            sub_Y = Y_n[X_n[mejor] == valor]
            hijo = {"atributo": None, "hijos": {}}
            nodo["hijos"][valor] = hijo
            pila.append((hijo, sub_X, sub_Y, depth + 1))
    pass


  def depth(self)->int:
    """ consulta la profundidad del árbol
    Return:
      profundidad del árbol
    """
    if self.tree is None:
        return 0

    pila = [(self.tree, 0)]
    max_depth = 0

    while pila:
        nodo, nivel = pila.pop()
        max_depth = max(max_depth, nivel)
        for hijo in nodo["hijos"].values():
            if "hoja" not in hijo:
                pila.append((hijo, nivel + 1))

    return max_depth



  def rules(self) -> [str]:
    """ consultar las reglas del árbol
    Return:
      reglas del árbol de decisión
    """
    reglas = []
    pila = [(self.tree, "")]

    while pila:
        nodo, condicion = pila.pop()

        if "hoja" in nodo:
            reglas.append(f"{condicion} -> {nodo['hoja']}")
            continue

        for valor, hijo in nodo["hijos"].items():
            nueva_condicion = f"{condicion} AND {nodo['atributo']}={valor}"
            pila.append((hijo, nueva_condicion))

    return reglas

  def to_string(self) -> str:
    """ consultar la estructura del arbol
    Return:
      estructura del árbol
    """
    resultado: str = ""
    pila = [(self.tree, 0)]

    while pila:
        nodo, nivel = pila.pop()
        indent = "  " * nivel

        if "hoja" in nodo:
            resultado += f"{indent}-> {nodo['hoja']}\n"
            continue

        resultado += f"{indent}{nodo['atributo']}\n"
        for valor, hijo in nodo["hijos"].items():
            resultado += f"{indent}  {valor}\n"
            pila.append((hijo, nivel + 1))

    return resultado

