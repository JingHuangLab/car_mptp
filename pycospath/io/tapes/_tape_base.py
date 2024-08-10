#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The base class of DataTapes."""

from typing import Self

from abc import ABC, abstractmethod

class Tape(ABC):
  """The DataTape for recording various simulation data."""

  @classmethod
  @abstractmethod
  def erase(cls, instance: Self) -> Self:
    """Replicate a new instance of the class with an empty data pool from an existing instance."""

  @abstractmethod
  def write(self, cargo: tuple) -> None:
    """Append the cargo to the temporal DataTape."""

  @abstractmethod
  def serialize(self, to_file: str) -> None:
    """Serialize all taped data to file."""


