#
# pyCoSPath: A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""DataTape for recording the Voronoi boundary collision data."""

from typing import Self

import numpy as np

from pycospath.io.tapes import Tape

class VoronoiReflectionTape(Tape):
  """DataTape for recording the Voronoi boundary collision data."""

  @classmethod
  def erase(cls, instance: Self = None) -> Self:
    """Replicate a new instance of the class with an empty data pool from an existing instance.
    
      Args:
        instance (VoroCollisionDataTape):
          This keyword only presents for API consistency: it has no effect.
    """
    assert isinstance(instance, VoronoiReflectionTape), f'Illegal instance type.'
    return cls()
    
  def __init__(self) -> None:
    """Inintialize a DataTape for recording the Voronoi boundary collision data."""
    # Data pool.
    self._data_step_index = None
    self._data_cell_index = None
    self._data_replica_coordinates = None

  def write(self, cargo: tuple[int, int, np.ndarray]) -> None:
    """Write cargo to the temporal DataTape.

      Args:
        cargo (tuple[int, int, np.ndarray]):
          A tuple of (sequentially in the tuple):
          1. the step count at the time of collision to the Voronoi boundaries; 
          2. the box ID to which the Replica collides; 
          3. the Replica region coordinates after when the collision is detected.
    """
    # Unpack cargo.
    i_step, j_cell, replica_coordinates = cargo

    # To legal type.
    i_step = np.asarray([i_step], dtype=int)
    j_cell = np.asarray([j_cell], dtype=int)
    replica_coordinates = np.expand_dims( replica_coordinates , axis=0)
    
    # Append values.
    self._data_step_index = i_step if self._data_step_index is None \
                                        else np.concatenate((self._data_step_index, i_step), axis=0)
    self._data_cell_index = j_cell if self._data_cell_index is None \
                                        else np.concatenate((self._data_cell_index, j_cell), axis=0)
    self._data_replica_coordinates = replica_coordinates if self._data_replica_coordinates is None \
                  else np.concatenate((self._data_replica_coordinates, replica_coordinates), axis=0)

  def serialize(self, to_file: str) -> None:
    """Serialize all taped data to file."""
    np.savez(to_file, 
             step_index     = self._data_step_index, 
             cell_index     = self._data_cell_index, 
             replica_coords = self._data_replica_coordinates, ) # TODO: here.


