#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The 2D Integrators."""

from typing import Callable

from abc import ABC, abstractmethod

import numpy as np

class TwoDIntegrator(ABC):
  """The abstract class of 2D Integrator."""

  def __init__(self, 
               timestep_size: float = .0001,
               ) -> None:
    """Create a 2D Integrator.

      Args:
        timestep_size (float, optional):
          The stepsize for each timestep.
          Default: .0001, allowed values are positive.
    """
    assert isinstance(timestep_size, float), "Illegal timestep_size type."
    assert timestep_size > 0.,               "Illegal timestep_size spec."
    self._timestep_size = timestep_size

  # External interfaces. ---------------------------------------------------------------------------

  def implement(self, 
                fn_set_coordinates:         Callable[[np.ndarray], None      ], 
                fn_set_velocities:          Callable[[np.ndarray], None      ], 
                fn_get_coordinates:         Callable[[],           np.ndarray], 
                fn_get_velocities:          Callable[[],           np.ndarray], 
                fn_get_mass_per_dof:        Callable[[],           np.ndarray], 
                fn_get_potential_gradients: Callable[[],           np.ndarray], 
                ) -> None:
    """To implement the TwoDContext state query functions."""
    assert callable(fn_set_coordinates),         "Illegal non-callable fn_set_coordinates."
    assert callable(fn_set_velocities),          "Illegal non-callable fn_set_velocities."
    assert callable(fn_get_coordinates),         "Illegal non-callable fn_get_coordinates."
    assert callable(fn_get_velocities),          "Illegal non-callable fn_get_velocities."
    assert callable(fn_get_mass_per_dof),        "Illegal non-callable fn_get_mass_per_dof."
    assert callable(fn_get_potential_gradients), "Illegal non-callable fn_get_potential_gradients."

    self.set_coordinates = fn_set_coordinates
    self.set_velocities  = fn_set_velocities
    self.get_coordinates = fn_get_coordinates
    self.get_velocities  = fn_get_velocities
    self.get_mass_per_dof = fn_get_mass_per_dof
    self.get_potential_gradients = fn_get_potential_gradients

  # External interface prompts - TwoDContext runtime.

  def set_coordinates(self, coordinates: np.ndarray) -> None:
    """Prompt: Set the TwoDContext coordinates."""
    raise RuntimeError("Prompt method not realized in TwoDIntegrator.implement().")

  def set_velocities(self, velocities: np.ndarray) -> None:
    """Prompt: Set the TwoDContext velocities."""
    raise RuntimeError("Prompt method not realized in TwoDIntegrator.implement().")

  def get_coordinates(self) -> np.ndarray:
    """Prompt: Get the TwoDContext coordinates."""
    raise RuntimeError("Prompt method not realized in TwoDIntegrator.implement().")

  def get_velocities(self) -> np.ndarray:
    """Prompt: Get the TwoDContext velocities."""
    raise RuntimeError("Prompt method not realized in TwoDIntegrator.implement().")

  def get_mass_per_dof(self) -> np.ndarray:
    """Prompt: Get the TwoDContext per-DOF masses."""
    raise RuntimeError("Prompt method not realized in TwoDIntegrator.implement().")

  def get_potential_gradients(self) -> np.ndarray:
    """Prompt: Get the TwoDContext potential gradients."""
    raise RuntimeError("Prompt method not realized in TwoDIntegrator.implement().")

  # TwoDIntegrator properties. ---------------------------------------------------------------------

  def get_timestep_size(self) -> float:
    """Get the stepsize for each timestep."""
    return self._timestep_size
  
  def get_velocities_time_offset(self) -> float:
    """Get the time interval by which the velocities are offset from the positions."""
    return 0.

  # TwoDIntegrator MD runtime. ---------------------------------------------------------------------
  
  def initialize_velocities(self, inverse_beta: float) -> None:
    """Initialize the velocities of the 2D particle at desinated temperature.
    
      Args
        inverse_beta (float):
          The inverse $\beta$ ($\frac{1}{\beta} = k_{b}*T$).
    """
    mass = self.get_mass_per_dof()
    velocities = np.random.normal(loc=0., scale=inverse_beta/mass, size=(2, ))

    # If velocities is integrated at half timesteps.
    if (time_offset:=self.get_velocities_time_offset()) != 0.:
      # V(.5dt) = V(dt) + .5dt*\frac{\partial U}{\partial x(dt)} * m**-1
      velocities += time_offset * self.get_potential_gradients() * mass**-1

    self.set_velocities(velocities=velocities)

  @abstractmethod
  def step(self, num_steps: int) -> None:
    """Advance the simulation through time by taking steps timesteps.
    
      Args:
        num_steps (int):
          The number of timesteps to take. 
    """



class TwoDLangevinIntegrator(TwoDIntegrator):
  """The 2D Langevin Leapfrog Integrator."""

  def __init__(self, 
               timestep_size: float =   .0001,
               friction_coef: float =  5.    ,
               inverse_beta:  float = 10.    , 
               ) -> None:
    """Create a 2D Langevin Leapfrog Integrator.

      Args:
        timestep_size (float, optional):
          The stepsize for each timestep.
          Default: .0001, allowed values are positive.
        friction_coef (float, optional):
          The friction coefficient.
          Default: 5., allowed values are non-negative. 
        inverse_beta (float, optional):
          The inverse $\beta$ ($\frac{1}{\beta} = k_{b}*T$).
          Default: 10., allowed values are non-negative.
    """
    TwoDIntegrator.__init__(self, 
                            timestep_size=timestep_size, )

    assert isinstance(friction_coef, float), "Illegal friction_coef type."
    assert friction_coef >= 0.,              "Illegal friction_coef spec."
    self._friction_coef = friction_coef

    assert isinstance(inverse_beta,  float), "Illegal inverse_beta type."
    assert inverse_beta >= 0.,               "Illegal inverse_beta spec."
    self._inverse_beta = inverse_beta

  # External interfaces. ---------------------------------------------------------------------------

  def implement(self, 
                fn_set_coordinates:         Callable[[np.ndarray], None      ], 
                fn_set_velocities:          Callable[[np.ndarray], None      ], 
                fn_get_coordinates:         Callable[[],           np.ndarray], 
                fn_get_velocities:          Callable[[],           np.ndarray], 
                fn_get_mass_per_dof:        Callable[[],           np.ndarray], 
                fn_get_potential_gradients: Callable[[],           np.ndarray], 
                ) -> None:
    """To realize the Context phase state functions."""
    TwoDIntegrator.implement(self, 
                             fn_set_coordinates=fn_set_coordinates, 
                             fn_set_velocities =fn_set_velocities, 
                             fn_get_coordinates=fn_get_coordinates, 
                             fn_get_velocities =fn_get_velocities,
                             fn_get_mass_per_dof       =fn_get_mass_per_dof, 
                             fn_get_potential_gradients=fn_get_potential_gradients, )
    
    # Scaling factors on the three terms in the Langevin numerical integration.
    # v(t') = v(t) * a - gradient * b / m + randnorm * c * sqrt(kB*T/m)
    self._a = np.exp(-self._friction_coef*self._timestep_size)

    self._b = self._timestep_size # Zero-damped velocities Verlet gradient scale: dt
    if self._friction_coef != 0.: #      Damped gradient scale.
      self._b = (1.-np.exp(-self._friction_coef*self._timestep_size)) / self._friction_coef

    self._c = np.sqrt(1.-np.exp(-2.*self._friction_coef*self._timestep_size))
  
  # TwoDIntegrator properties. ---------------------------------------------------------------------

  def get_velocities_time_offset(self) -> float:
    """Get the time interval by which the velocities are offset from the positions."""
    return .5*self._timestep_size

  # TwoDIntegrator MD runtime. ---------------------------------------------------------------------

  def step(self, 
           num_steps: int = 1, 
           ) -> None:
    """Advance the simulation through time by taking steps timesteps.
    
      Args:
        num_steps (int, optional):
          The number of timesteps to take. 
          Default: 1.
    """
    gauss_rand = np.random.normal(loc=0, scale=1, size=(num_steps, 2))

    for i_step in range(num_steps):
      m = self.get_mass_per_dof()         # mass.
      x = self.get_coordinates()          # coordinates.
      v = self.get_velocities()           # velocities.
      g = self.get_potential_gradients()  # potential gradient.
      r = gauss_rand[i_step, :]           # random noise ~ N(0,1).

      # integrate halfstep velocities.
      v_half = v*self._a - g*self._b/m + r*self._c*np.sqrt(self._inverse_beta/m)
      # update fullstep coordinates.
      x_full_next = x + v_half*self._timestep_size
      # update halfstep velocities.
      v_half_next = (x_full_next - x) / self._timestep_size

      # update in context.
      self.set_velocities (velocities =v_half_next)
      self.set_coordinates(coordinates=x_full_next)



class TwoDScaledLangevinIntegrator(TwoDLangevinIntegrator):
  """The 2D Langevin Integrator with potential gradients scaling."""

  def __init__(self, 
               timestep_size: float =   .0001,
               friction_coef: float =  5.    ,
               inverse_beta:  float = 10.    ,
               scaling_coef:  float =  1.    ,
               ) -> None:
    """Create a 2D Langevin integrator with potential gradients scaling.

      Args:
        timestep_size (float, optional):
          The stepsize for each timestep.
          Default: .0001, allowed values are positive.
        friction_coef (float, optional):
          The friction coefficient. 
          Default: 5., allowed values are non-negative. 
        inverse_beta (float, optional):
          The inverse $\beta$ ($\frac{1}{\beta} = k_{b}*T$).
          Default: 10, allowed values are non-negative.
        scaling_coef (float, optional):
          The scaling factor for the potential gradient.
          Default: 1., allowed values are non-negative.
    """
    TwoDLangevinIntegrator.__init__(self, 
                                    timestep_size=timestep_size, 
                                    friction_coef=friction_coef, 
                                    inverse_beta =inverse_beta, )
    
    assert isinstance(scaling_coef, float), "Illegal scaling_coef type."
    assert scaling_coef > 0.,               "Illegal scaling_coef spec."
    self._scaling_coef = scaling_coef

  # External interfaces. ---------------------------------------------------------------------------

  def implement(self, 
                fn_set_coordinates:         Callable[[np.ndarray], None      ], 
                fn_set_velocities:          Callable[[np.ndarray], None      ], 
                fn_get_coordinates:         Callable[[],           np.ndarray], 
                fn_get_velocities:          Callable[[],           np.ndarray], 
                fn_get_mass_per_dof:        Callable[[],           np.ndarray], 
                fn_get_potential_gradients: Callable[[],           np.ndarray], 
                ) -> None:
    """To realize the Context phase state functions."""
    def _apply_potential_gradients_scaling(fn_get_potential_gradients: Callable[[], np.ndarray]):
      """Wrapper that applies gradients scaling to self.get_potential_gradients()."""
      def wrapper():
        """Prompt: Get the TwoDContext potential gradients."""
        return self._scaling_coef*fn_get_potential_gradients()
      return wrapper
    
    TwoDLangevinIntegrator.implement(self, 
                                     fn_set_coordinates=fn_set_coordinates, 
                                     fn_set_velocities =fn_set_velocities, 
                                     fn_get_coordinates=fn_get_coordinates, 
                                     fn_get_velocities =fn_get_velocities,
                                     fn_get_mass_per_dof       =fn_get_mass_per_dof, 
                                     fn_get_potential_gradients=fn_get_potential_gradients, )
    
    # Replace gradient return with the scaled version.
    ## Note that this does not change the get_potential_gradient function in the TwoDContext.
    self.get_potential_gradients = _apply_potential_gradients_scaling(
                                          fn_get_potential_gradients=self.get_potential_gradients, )



class TwoDMassScaledLangevinIntegrator(TwoDLangevinIntegrator):
  """The 2D Langevin Integrator with mass scaling."""

  def __init__(self, 
               timestep_size: float =   .0001,
               friction_coef: float =  5.    ,
               inverse_beta:  float = 10.    ,
               scaling_mass:  np.ndarray = np.ones((2, )),
               ) -> None:
    """Create a 2D Langevin intergator with mass scaling.
    
      Args:
        timestep_size (float, optional):
          The stepsize for each timestep.
          Default: .0001, allowed values are positive.
        friction_coef (float, optional):
          The friction coefficient. 
          Default: 5., allowed values are non-negative. 
        inverse_beta (float, optional):
          The inverse $\beta$ ($\frac{1}{\beta} = k_{b}*T$).
          Default: 10, allowed values are non-negative.
        scaling_mass (np.ndarray, optional):
          The scaling vector for the per-DOF masses in the TwoDContext, shape (num_replica_dofs, ).
          Default: np.ones((2, )), allowed values are positives.
    """
    TwoDLangevinIntegrator.__init__(self, 
                                    timestep_size=timestep_size, 
                                    friction_coef=friction_coef, 
                                    inverse_beta =inverse_beta, )
    
    assert isinstance(scaling_mass, np.ndarray), "Illegal scaling_mass type."
    assert scaling_mass.shape == (2, ),          "Illegal scaling_mass spec."
    assert (scaling_mass > 0.).all(),            "Illegal scaling_mass spec."
    self._scaling_mass = np.copy(scaling_mass)

  # External interfaces. ---------------------------------------------------------------------------

  def implement(self, 
                fn_set_coordinates:         Callable[[np.ndarray], None      ], 
                fn_set_velocities:          Callable[[np.ndarray], None      ], 
                fn_get_coordinates:         Callable[[],           np.ndarray], 
                fn_get_velocities:          Callable[[],           np.ndarray], 
                fn_get_mass_per_dof:        Callable[[],           np.ndarray], 
                fn_get_potential_gradients: Callable[[],           np.ndarray], 
                ) -> None:
    """To realize the Context phase state functions."""
    def _apply_mass_per_dof_scaling(fn_get_mass_per_dof: Callable[[], np.ndarray], ):
      """Wrapper that applies per-DOF masses scaling to self.get_mass_per_dof()."""
      def wrapper():
        """Prompt: Get the TwoDContext per-DOF masses."""
        return self._scaling_mass*fn_get_mass_per_dof()
      return wrapper
    
    TwoDLangevinIntegrator.implement(self, 
                                     fn_set_coordinates=fn_set_coordinates, 
                                     fn_set_velocities =fn_set_velocities, 
                                     fn_get_coordinates=fn_get_coordinates, 
                                     fn_get_velocities =fn_get_velocities, 
                                     fn_get_mass_per_dof       =fn_get_mass_per_dof, 
                                     fn_get_potential_gradients=fn_get_potential_gradients, )
    
    # Replace per-DOF return with the scaled version.
    ## Note that this does not change the get_mass_per_dof function in the TwoDContext.
    self.get_mass_per_dof=_apply_mass_per_dof_scaling(fn_get_mass_per_dof=self.get_mass_per_dof, )


