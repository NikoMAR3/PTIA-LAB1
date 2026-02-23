# Documentar los métodos implementados
import pandas as pd

from BaseFunctions.Metric import Metric


class F1Score(Metric):
  """ Metrica de media armónica entre Precisión y Exhaustividad. Implementa Metric
  """

  def value(self, Y: pd.DataFrame, Yp: pd.DataFrame)->float:
    TP = ((Y == 1) & (Yp == 1)).sum()
    FP = ((Y == 0) & (Yp == 1)).sum()
    FN = ((Y == 1) & (Yp == 0)).sum()

    if (TP + FP) == 0 or (TP + FN) == 0:
      return 0.0

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    if (precision + recall) == 0:
      return 0.0

    return (2 * precision * recall) / (precision + recall)

