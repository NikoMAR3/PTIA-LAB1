# Documentar los métodos implementados
import pandas as pd

from BaseFunctions.Criterium import Criterium


class Entropy(Criterium):
  """ Criterio de selección - impureza por entropia
  """
  def __new__(cls):
    pass

  def impurity(self, V: pd.DataFrame) -> float:
    pass

  def gain(self, a: str, X: pd.DataFrame, Y: pd.DataFrame) -> float:
    pass

  def treeImpurity(self, nodes: pd.DataFrame) -> float:
    pass