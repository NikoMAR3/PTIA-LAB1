from abc import ABC, abstractmethod

import numpy as np
import pandas as pd



class Criterium(ABC):
  """ Abstracta: Define el criterio para seleccionar y dar prioridad a los atributos
  Representa la impureza
  """
  @classmethod
  def use(cls, name: str):
    name = name.lower()
    match name:
      case "entropy":
        from BaseFunctions.Entropy import Entropy
        return Entropy()

  @abstractmethod
  def impurity(self, V: pd.DataFrame) -> float:
    pass

  def gain(self, a: str, X: pd.DataFrame, Y: pd.Series) -> float:
    h_padre = self.impurity(Y)
    h_hijos = 0
    for valor in X[a].unique():
      subset = Y[X[a] == valor]
      h_hijos += (len(subset) / len(Y)) * self.impurity(subset)
    return h_padre - h_hijos

  def treeImpurity(self, nodes: [pd.DataFrame]) -> float:
    """ computa la impureza de todo un arbol
    Args:
      nodes    : datos de cada uno de los nodos del arbol
    Returns:
      valor de la impureza del arbol
    """
    impurity = 0
    total = sum(len(n) for n in nodes)
    for n in nodes:
      impurity += self.impurity(n) * (len(n) / total)
    return impurity

