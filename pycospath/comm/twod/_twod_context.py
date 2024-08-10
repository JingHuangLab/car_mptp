#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""2D simulation Context."""

from functools import partial

import numpy as np

from pycospath.comm.twod import TwoDSystem, TwoDIntegrator

class TwoDContext:
  """The TwoDContext."""
  
  def __init__(self, 
               twod_system:     TwoDSystem,
               twod_integrator: TwoDIntegrator,
               ) -> None:
    """Create an TwoDContext.

      Args:
        twod_system (TwoDPotential):
          The 2D System object.
        twod_integrator (TwoDIntegrator):
          The 2D Integrator object.
    """
    # Initialize internal states at zeros.
    self._coordinates = np.zeros((2, ))
    self._velocities  = np.zeros((2, ))

    # Initialize System.
    assert isinstance(twod_system, TwoDSystem), "Illegal twod_system type."
    self._system = twod_system

    # Initialize Integrator.
    assert isinstance(twod_integrator, TwoDIntegrator), "Illegal twod_integrator type."
    self._integrator = twod_integrator
    self._integrator.implement(fn_set_coordinates=partial(self.set_coordinates, in_place=True),
                               fn_set_velocities =partial(self.set_velocities,  in_place=True), 
                               fn_get_coordinates=partial(self.get_coordinates, in_place=True), 
                               fn_get_velocities =partial(self.get_velocities,  in_place=True),
                               fn_get_mass_per_dof       =self.get_mass_per_dof, 
                               fn_get_potential_gradients=self.get_potential_gradients, )

  def get_twod_system(self) -> TwoDSystem:
    """Get the 2D System object."""
    return self._system

  def get_twod_integrator(self) -> TwoDIntegrator:
    """Get the 2D Integrator object."""
    return self._integrator
  
  # TwoDContext runtime. ---------------------------------------------------------------------------
  
  def get_mass_per_dof(self) -> np.ndarray:
    """Get the masses tied to each 2D System DOF."""
    return np.ones((2, ))*self.get_twod_system().get_particle_mass()
  
  def set_coordinates(self,
                      coordinates: np.ndarray, 
                      in_place: bool = False,
                      ) -> None:
    """Set the coordinates of the 2D particle. If in_place==True, the internal state of coordinates
      will share the same memory allocation with the input.
    """
    if in_place == False:
      assert isinstance(coordinates, np.ndarray), "Illegal coordinates type."
      assert coordinates.shape == (2, ),          "Illegal coordinates spec."
      coordinates = np.copy(coordinates)

    self._coordinates = coordinates

  def set_velocities(self, 
                     velocities: np.ndarray, 
                     in_place: bool = False,
                     ) -> None:
    """Set the velocities of the 2D particle. If in_place==True, the internal state of velocities
      will share the same memory allocation with the input.
    """
    if in_place == False:
      assert isinstance(velocities, np.ndarray), "Illegal velocities type."
      assert velocities.shape == (2, ),          "Illegal velocities spec."
      velocities = np.copy(velocities)
      
    self._velocities = velocities

  def get_coordinates(self, in_place: bool = False) -> np.ndarray:
    """Get a copy of the coordinates of the 2D particle. If in_place==True, the returned value will
      share the same memory allocation with the internal state of coordinates. 
    """
    return np.copy(self._coordinates) if in_place == False else self._coordinates

  def get_velocities(self, in_place: bool = False) -> np.ndarray:
    """Get a copy of the velocities of the 2D particle. If in_place==True, the returned value will
      share the same memory allocation with the internal state of velocities. 
    """
    return np.copy(self._velocities) if in_place == False else self._velocities
  
  def get_potential_gradients(self) -> np.ndarray:
    """Get the potential gradient at the current coordinates. 
        
      Returns:
        potential_gradients (np.ndarray): 
          The potential gradients, shape (2, ).
    """
    return self.get_twod_system().get_potential_gradients(coordinates=self._coordinates)
  
  def get_potential_energy(self) -> float:
    """Get the potential energy at the current coordinates. 
        
      Returns:
        potential_energy (float):
          The potential energy.
    """
    return self.get_twod_system().get_potential_energy(coordinates=self._coordinates)
  
  def get_kinetic_energy(self) -> np.ndarray:
    """Get the kinetic energy at the current velocities.
        
      Returns:
        kinetic_energy (float):
          The kinetic energy.
    """
    mass = self.get_mass_per_dof()
    velocities = self.get_velocities()

    #Integrator may hold half step velocities at (t+.5*dt)
    if (offset:=(integrator:=self.get_twod_integrator()).get_velocities_time_offset()) != 0.:
      # V(dt) = V(.5dt) - .5dt * \frac{\partial U}{\partial x(dt)} / m
      velocities -= offset * integrator.get_potential_gradients() * mass**-1
    
    return np.sum(.5*mass*velocities**2)


