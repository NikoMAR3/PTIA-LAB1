# Documentar los métodos implementados
from BaseFunctions.Metric import Metric


class Precision(Metric):
  """ Metrica de precision). Implementa Metric
  """
  def __new__(cls):
    pass

  def value(self, Y: pd.DataFrame, Yp: pd.DataFrame)->float:
    pass