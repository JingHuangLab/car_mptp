#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The 2D Systems."""

from typing import Type

from abc import ABC

from functools import partial

import numpy as np

from pycospath.comm.twod import TwoDForce, TwoDMullerBrownForce, TwoDWolfeQuappForce

class TwoDSystem(ABC):
  """The base class of 2D System."""

  def __init__(self, 
               particle_mass: float = 1.,
               ) -> None:
    """The base class of 2D system.
    
      Args:
        particle_mass (float, optional):
          The mass tied to the 2D particle.
          Default: 1.
    """
    # Sanity check.
    assert isinstance(particle_mass, float), "Illegal particle_mass type."
    assert particle_mass > 0.,               "Illegal particle_mass spec."
    self._particle_mass = particle_mass

    # List of forces.
    self._forces: list[TwoDForce] = []
  
  def get_particle_mass(self) -> float:
    """Get the mass of the 2D particle."""
    return self._particle_mass
  
  def append_force(self, twod_force: TwoDForce) -> int:
    """Append a TwoDForce object to the System, returns the ID of the Force in the System."""
    assert isinstance(twod_force, TwoDForce), "Illegal force type."
    self._forces.append(twod_force)
    return len(self._forces)-1
  
  def remove_force(self, twod_force_id: int) -> None:
    """Remove the TwoDForce object in the System using the ID of the Force."""
    self._forces.pop(twod_force_id)
  
  def get_force(self, twod_force_id: int) -> TwoDForce:
    """Get the TwoDForce object in the System using the ID of the Force."""
    return self._forces[twod_force_id]

  def get_potential_energy(self, coordinates: np.ndarray) -> float:
    """Get the potential energy at the coordinates. 
      
      Args:
        coordinates (np.ndarray):
          The coordinates, shape (2, ).

      Returns:
        potential_energy (float):
          The total potential energy.
    """
    potential_energy = 0
    for force in self._forces:
      potential_energy += force.get_potential_energy(coordinates=coordinates)

    return potential_energy
  
  def get_potential_gradients(self, coordinates: np.ndarray) -> np.ndarray:
    """Get the potential gradients at the coordinates.

      Args:
        coordinates (np.ndarray):
          The coordinates, shape (2, ).
      
      Returns:
        potential_gradients (np.ndarray):
          The potential gradients, shape (2, ).
    """
    potential_gradients = np.zeros((2, ))
    for force in self._forces:
      potential_gradients += force.get_potential_gradients(coordinates=coordinates)
    
    return potential_gradients

  def get_pes(self, 
              xmin: float, xmax: float, 
              ymin: float, ymax: float, 
              grid: int,   ecut: float,
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the potential energy surface of the 2D potential function in region.
        
      Args:
        xmin / xmax (floats):
          The minimal / maximal of x axis.
        ymin / ymax (floats):
          The minimal / maximal of y axis.
        grid (int):
          The number of grid points on each dimension. Must be positive.
        ecut (float):
          Energy values higher than ecut is set to ecut for ploting purpose.
        
      Returns:
        x_cor (np.ndarray), y_cor (np.ndarray), v (np.ndarray): 
          A tuple that could be plotted directly with matplotlib:
          ```
          ax.contourf(x_cor, y_cor, v)
          ```
    """
    # Build the x-y grid dimensions
    x_cor = np.linspace(xmin, xmax, num=grid)
    y_cor = np.linspace(ymin, ymax, num=grid)

    # Get coordinates grid.
    xx, yy = np.mgrid[xmin:xmax:grid*1j, ymin:ymax:grid*1j]
    cor = np.vstack([xx.ravel(), yy.ravel()]).T  # cor.shape=(grid^2, 2)

    # Compute potential at each coordinates grid point. 
    v = np.zeros((cor.shape[0], ))
    for pt in range(cor.shape[0]):
      v[pt] = self.get_potential_energy(cor[pt])

    # Set the upper bound of V
    v = np.where(v<=ecut, v, ecut)
    v = v.reshape((grid, grid)).T

    return x_cor, y_cor, v



# Aliases. 

def _twod_muller_brown_system(particle_mass: float = 1.) -> TwoDSystem:
  """Returns a TwoDSystem with a TwoDMullerBrownForce."""
  twod_system = TwoDSystem(particle_mass=particle_mass)
  twod_system.append_force(TwoDMullerBrownForce())
  twod_system.get_pes = partial(twod_system.get_pes, xmin=-1.5, xmax=1.3, 
                                                     ymin=-0.5, ymax=2.3, 
                                                     grid= 101, ecut=200, )
  return twod_system


def _twod_wolfe_quapp_system(particle_mass: float = 1.) -> TwoDSystem:
  """Returns a TwoDSystem with a TwoDWolfeQuappForce."""
  twod_system = TwoDSystem(particle_mass=particle_mass)
  twod_system.append_force(TwoDWolfeQuappForce())
  twod_system.get_pes = partial(twod_system.get_pes, xmin=-2.5, xmax= 2.5, 
                                                     ymin=-2.5, ymax= 2.5, 
                                                     grid= 101, ecut=10. , )
  return twod_system


TwoDMullerBrownSystem: Type[TwoDSystem] = _twod_muller_brown_system
"""Creates a TwoDSystem() with a TwoDMullerBrownForce()."""

TwoDWolfeQuappSystem:  Type[TwoDSystem] = _twod_wolfe_quapp_system
"""Creates a TwoDSystem() with a TwoDWolfeQuappForce()."""
