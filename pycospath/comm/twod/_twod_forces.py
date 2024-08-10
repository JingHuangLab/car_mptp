#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The 2D potentials."""

from abc import ABC, abstractmethod

import numpy as np

class TwoDForce(ABC):
  """The abstract class of 2D Forces."""

  @abstractmethod
  def get_potential_energy(self, coordinates: np.ndarray) -> float:
    """Get the potential energy at the coordinates."""

  @abstractmethod
  def get_potential_gradients(self, coordinates: np.ndarray) -> np.ndarray:
    """Get the potential gradients at the coordinates."""



class TwoDRmsdCVForce(TwoDForce):
  """The TwoD RMSD potential."""

  def __init__(self, 
               force_constant:        float,
               force_cutoff_distance: float, 
               reference_coordinates: np.ndarray, 
               ) -> None:
    """Create a RMSD bias potential.
    
      Args:
        force_constant (float):
          The force constant.
        force_cutoff_d (float):
          The distance beyond which the RMSD bias force is active.
        ref_coordinates (np.ndarray):
          The reference coordinates.
    """
    # Sanity checks.
    assert isinstance(force_constant, float), "Illegal force_constant type."
    assert force_constant > 0.,               "Illegal force_constant spec."
    self._force_constant = force_constant

    assert isinstance(force_cutoff_distance, float), "Illegal force_cutoff_distance type."
    assert force_cutoff_distance >= 0.,              "Illegal force_cutoff_distance spec."
    self._force_cutoff_distance = force_cutoff_distance

    assert isinstance(reference_coordinates, np.ndarray), "Illegal reference_coordinates type."
    assert reference_coordinates.shape == (2, ),          "Illegal reference_coordinates spec."
    self._reference_coordinates = np.copy(reference_coordinates)

  def get_potential_energy(self, coordinates: np.ndarray) -> float:
    """Get the potential energy at the coordinates."""
    # Note that the number of particles is always 1.
    rmsd = np.sqrt(np.sum((coordinates - self._reference_coordinates)**2))
    ener = self._force_constant * (np.amax([0, rmsd-self._force_cutoff_distance])**2)
    return ener

  def get_potential_gradients(self, coordinates: np.ndarray) -> np.ndarray:
    """Get the potential gradients at the coordinates."""
    # Note that the number of particles is always 1.
    diff = coordinates - self._reference_coordinates
    rmsd = np.sqrt(np.sum(diff**2))
    grad = 2.*self._force_constant*np.amax([0., 1-self._force_cutoff_distance/rmsd])*diff
    return grad



class TwoDMullerBrownForce(TwoDForce):
  """The Müller-Brown potential.

    Location of Saddle Points and Minimum Energy Paths by a Constrained Simplex Optimization 
    Procedure
    K. Müller, L.D. Brown, Theor. Chim. Acta, 1979, 53, 75-93. DOI: 10.1007/BF00547608
  """
  
  def __init__(self) -> None:
    """Create a 2D Müller-Brown potential."""
    # M-B pot constants.
    self._A = [-200., -100. , -170. , 15. , ]
    self._a = [-  1., -  1. , -  6.5,   .7, ]
    self._b = [   0.,    0. ,   11. ,   .6, ]
    self._c = [- 10., - 10. , -  6.5,   .7, ] 
    self._x = [   1.,    0. , -  0.5, -1. , ]
    self._y = [   0.,     .5,    1.5,  1. , ]

  def get_potential_energy(self, coordinates: np.ndarray) -> float:
    """Get the potential energy at the coordinates."""
    ener = 0
    for i in range(4):
      ener += self._A[i] * np.exp(  self._a[i]*(coordinates[0]-self._x[i])**2
                                  + self._b[i]*(coordinates[0]-self._x[i]) \
                                              *(coordinates[1]-self._y[i])
                                  + self._c[i]*(coordinates[1]-self._y[i])**2 )
    return ener
  
  def get_potential_gradients(self, coordinates: np.ndarray) -> np.ndarray:
    """Get the potential gradients at the coordinates."""
    grad = np.zeros(coordinates.shape)
    for i in range(4):
      u = self._A[i] * np.exp(  self._a[i]*(coordinates[0]-self._x[i])**2
                              + self._b[i]*(coordinates[0]-self._x[i])*(coordinates[1]-self._y[i])
                              + self._c[i]*(coordinates[1]-self._y[i])**2 )
      grad[0] += u * (  2.*self._a[i]*(coordinates[0]-self._x[i])\
                      +    self._b[i]*(coordinates[1]-self._y[i]) )
      grad[1] += u * (  2.*self._c[i]*(coordinates[1]-self._y[i])\
                      +    self._b[i]*(coordinates[0]-self._x[i]) )
    return grad



class TwoDWolfeQuappForce(TwoDForce):
  """The Wolfe-Quapp potential.

    The Chemical Dynamics of Symmetric and Asymmetric Reaction Coordinates
    S. Wolfe, H.B. Schlegel, I.G. Csizmadia, F. Bernardi, J. Am. Chem. Soc., 1975, 97, 2020-2024.
    DOI: 10.1021/ja00841a005

    A Growing String Method for the Reaction Pathway Defined by a Newton Trajectory
    W. Quapp, J. Chem. Phys., 2005, 122, 174106. DOI: 10.1063/1.1885467
  """
  def __init__(self) -> None:
    """Create a 2D Wolfe-Quapp force.
    
      Args:
        particle_mass (float, optional):
          The mass tied to the 2D particle.
          Default: 1.
    """
    # W-Q pot constants.
    self._a = -2.  # x^2
    self._b = -4.  # y^2
    self._c =   .3 # x
    self._d =   .1 # y

  def get_potential_energy(self, coordinates: np.ndarray) -> float:
    """Get the potential energy at the coordinates."""
    ener =           coordinates[0]**4 +              coordinates[1]**4 + \
           self._a * coordinates[0]**2 +    self._b * coordinates[1]**2 + \
                                     coordinates[0] * coordinates[1]    + \
           self._c * coordinates[0]    +    self._d * coordinates[1]
    return ener
  
  def get_potential_gradients(self, coordinates: np.ndarray) -> np.ndarray:
    """Get the potential gradients at the coordinates."""
    grad = np.zeros(coordinates.shape)
    grad[0] = 4.*coordinates[0]**3 - 2.*self._a*coordinates[0] + coordinates[1] + self._c
    grad[1] = 4.*coordinates[1]**3 - 2.*self._b*coordinates[1] + coordinates[0] + self._d
    return grad


