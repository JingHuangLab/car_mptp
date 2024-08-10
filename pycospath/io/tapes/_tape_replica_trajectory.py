#
# pyCoSPath: A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""Tape for recording the Replica region coordinates."""

from typing import Self

import numpy as np

from pycospath.io.tapes import Tape

class ReplicaTrajectoryTape(Tape):
  """Tape for recording the Replica region coordinates."""

  @classmethod
  def erase(cls, instance: Self = None) -> Self:
    """Replicate a new instance of the class with an empty data pool from an existing instance.
    
      Args:
        instance (ReplicaTrajectoryTape):
          This keyword only presents for API consistency: it has no effect.
    """
    assert isinstance(instance, ReplicaTrajectoryTape), 'Illegal instance type.'
    return cls()

  def __init__(self) -> None:
    """Initialize a Tape for recording the Replica region coordinates."""
    # Data pool.
    self._data_replica_coordinates: np.ndarray = None

  def write(self, cargo: tuple[np.ndarray, ]) -> None:
    """Write cargo to the temporal ReplicaTrajectoryTape. 
    
      Args:
        cargo (tuple[np.ndarray]):
          A tuple of:
            1. The Replica region coordinates (replica_coordinates).
    """
    # Unpack cargo and to legal type.
    replica_coordinates = cargo[0]
    replica_coordinates = np.expand_dims(replica_coordinates, axis=0)
    
    # Append values. 
    self._data_replica_coordinates = replica_coordinates if self._data_replica_coordinates is None \
                  else np.concatenate((self._data_replica_coordinates, replica_coordinates), axis=0)
  
  def serialize(self, to_file: str) -> None:
    """Serialize all taped data to file."""
    np.savez(to_file, 
             replica_coords = self._data_replica_coordinates, ) # TODO, here.


