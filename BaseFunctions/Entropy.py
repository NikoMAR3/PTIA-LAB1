# Documentar los métodos implementados
import numpy as np
import pandas as pd

from BaseFunctions.Criterium import Criterium


class Entropy(Criterium):
  """ Criterio de selección - impureza por entropia
  """
  def impurity(self, V: pd.DataFrame) -> float:
    proporciones = V.value_counts(normalize=True)
    h = 0
    for p in proporciones:
        h -= p * np.log2(p)
    return h
