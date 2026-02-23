# Documentar los métodos implementados
import pandas as pd

from BaseFunctions.Metric import Metric


class Accuracy(Metric):
  """ Metrica de exactitud (acertados / totales). Implementa Metric
  """

  def value(self, Y: pd.DataFrame, Yp: pd.DataFrame)->float:
      return (((Y == 1) & (Yp == 1)).sum() + ((Y == 0) & (Yp == 0)).sum())/ len(Y)

