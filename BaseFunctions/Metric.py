from abc import ABC, abstractmethod

import pandas as pd





class Metric(ABC):
  """ Abstracta: define entradas, salidas y el comportamiento inicial de los métodos clave para cualquier metrica
  Representa una metrica de desempeño P para evaluar una tarea T
  """
  @classmethod
  def use(cls, name: str):
    from BaseFunctions.F1Score import F1Score
    from BaseFunctions.Presicion import Precision
    from BaseFunctions.Recall import Recall
    from BaseFunctions.Accuracy import Accuracy
    """ obtiene metrica (clase) a partir del nombre
    Args:
        name: nombre esperado de la metrica
    Returns:
        clase métrica correspondiente
    """
    match name:
      case "F1Score":
        return F1Score
      case "Precision":
        return Precision
      case "Recall":
        return Recall
      case "Accuracy":
        return Accuracy
      case _:
        raise ValueError(f"Métrica '{name}' no reconocida")


  @abstractmethod
  def value(self, Y: pd.DataFrame, Yp: pd.DataFrame)->float:
    """ computa el desempeño P
    Args:
      Y   s de salidas esperadas (etiquetadas)
      Yp  : valores de salidas obtenidas
    Return:
      valor del desempeño
    """
    pass