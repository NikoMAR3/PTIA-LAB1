# Documentar los métodos implementados
import pandas as pd

from BaseFunctions.Metric import Metric


class Precision(Metric):
  """ Metrica de precision). Implementa Metric
  """


  def value(self, Y: pd.DataFrame, Yp: pd.DataFrame)->float:
    return ((Y == 1) & (Yp == 1)).sum() / (((Y == 1) & (Yp == 1)).sum() + ((Y == 1) & (Yp == 0)).sum())