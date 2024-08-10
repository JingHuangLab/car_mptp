#
# pyCoSPath: A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""DataTape for recording the underdamped Onsager-Machlup stochastic action."""

from typing import Self

import numpy as np

from pycospath.io.tapes import Tape

# NOTE (Z.S.): 
# Onsager Path Integral must be integrated in the same unit system on which the Langevin trajectory 
# was propagated. It is the products of a set of unit Gaussian random variable that is added to the 
# velocities measured in the specific unit system. Upon change of unit systems, the corresponding 
# unit random variables are no longer Gaussian ones. I suspect that the Jacobian for such change of
# units is non-linear. 

def _compute_onsager_path_integral(timestep_size: float, 
                                   friction_coef: float, 
                                   inverse_beta:  float,
                                   mass_per_dof:  np.ndarray, 
                                   coordinates: np.ndarray, 
                                   velocities:  np.ndarray, 
                                   gradients:   np.ndarray, 
                                   ) -> tuple[float, int]:
  """Integrates the Onsager path action.

    Args;
      timestep_size (float):
        The timestep size, unit: ps.
      friction_coef (float):
        The friction coefficient, unit: ps**-1.
      inverse_beta (float):
        The inverse $\beta$ ($\frac{1}{\beta} = k_{B}T$), unit: kcal/mol.
      mass_per_dof (np.ndarray):
        The per-DOF masses in the simulation Context, shape depends on Replica.
      coordinates (np.ndarray):
        The coordinates, shape (num_states, depends on Replica).
      velocities (np.ndarray):
        The velocities, shape (num_states, depends on Replica).
      gradients (np.ndarray):
        The potential gradients, shape (num_states-1, depends on Replica).

    Returns:
      S_action (float):
        The value of the Onsager path action, unit: kcal*ps/mol
      S_length (int):
        The number of states integrated in the S_action.
  """
  # Extend mass_per_dof to be shape compatible.
  mass_per_dof = mass_per_dof[np.newaxis, :]

  # Prefactors on the Four terms in the OM action functional. 
  ## Prefactor on the momentum & dissipation square term.
  pre_momentum_damping  = timestep_size / (4.*friction_coef*inverse_beta)
  ## Prefactor on the momentum & drifting cross term.
  pre_momentum_drifting =            1. / (2.*friction_coef*inverse_beta)
  ## Prefactor on the dissipation & drifting cross term.
  pre_damping_drifting  =            1. / (2.*inverse_beta)
  ## Prefactor on the drifting square term.
  pre_drifting_drifting = pre_momentum_damping

  # dx = x_{i+1} - x_{i},  (num_states-1, num_dofs)
  # vi = v_{i},            (num_states-1, num_dofs)
  # dv = v_{i+1} - v_{i},  (num_states-1, num_dofs)
  # dU = dU(x_i) / d(x_i), (num_states-1, num_dofs)
  dx = coordinates[1:  , :] - coordinates[0:-1, :]
  vi = velocities [0:-1, :]
  dv = velocities [1:  , :] - vi
  dU = gradients  [0:-1, :]
  # Four terms that are summed into the Onsager path action (with the prefactors).
  # All terms are of shape (num_states-1, num_dofs).

  momentum_damping  = (((dv/timestep_size + vi*friction_coef) * mass_per_dof)**2) / mass_per_dof
  momentum_drifting = dv * dU
  damping_drifting  = dx * dU
  drifting_drifting = dU * dU / mass_per_dof

  # Compute Onsager path action per step.
  S_action_per_step: np.ndarray = np.sum(pre_momentum_damping  * momentum_damping  + \
                                         pre_momentum_drifting * momentum_drifting + \
                                         pre_damping_drifting  * damping_drifting  + \
                                         pre_drifting_drifting * drifting_drifting, 
                                         axis=1, )
  
  return np.sum(S_action_per_step), S_action_per_step.shape[0]


class OnsagerPathActionTape(Tape):
  """DataTape for recording the Onsager-Machlup action for the Langevin equation."""

  @classmethod
  def erase(cls, instance: Self) -> Self:
    """Replicate a new instance of the class with an empty data pool from an existing instance.
    
      Args:
        instance (OnsagerPathActionTape):
          The datatape instance to replicate from.
    """
    assert isinstance(instance, OnsagerPathActionTape), 'Illegal instance type.'

    return cls(timestep_size=instance._timestep_size, 
               friction_coef=instance._friction_coef, 
               inverse_beta =instance._inverse_beta, 
               mass_per_dof =instance._mass_per_dof, )
  
  def __init__(self, 
               timestep_size: float, 
               friction_coef: float, 
               inverse_beta : float, 
               mass_per_dof : np.ndarray, 
               ) -> None:
    """Create a DataTape for recoding the Onsager-Machlup action for the Langevin equation.

      Args:
        timestep (float):
          The stepsize for each timestep, unit: ps.
        friction_coef (float):
          The friction coefficient, unit: ps**-1.
        inverse_beta (float):
          The inverse $\beta$ ($\frac{1}{\beta} = k_{B}T$), unit: kcal/mol.
        mass_per_dof (np.ndarray):
          The per-DOF masses in the simulation Context, shape depends on Replica.
    """
    # Sanity checks. 
    assert isinstance(timestep_size, float), "Illegal timestep_size type."
    self._timestep_size: float = timestep_size

    assert isinstance(friction_coef, float), "Illegal friction_coef type."
    self._friction_coef: float = friction_coef

    assert isinstance(inverse_beta,  float), "Illegal inverse_beta type."
    self._inverse_beta:  float = inverse_beta

    assert isinstance(mass_per_dof, np.ndarray), "Illegal mass_per_dof type."
    self._mass_per_dof: np.ndarray = np.copy(mass_per_dof)
    
    # Data pool: stack. 
    self._data_S_action_nobias = None
    self._data_S_length_nobias = None
    self._data_S_action_biased = None # For action reweighting from modified Langevin.
    self._data_S_length_biased = None # For action reweighting from modified Langevin.
    
    # Data pool: temporal per md step, refreshed per path.
    self._data_temp_coordinates = None
    self._data_temp_velocities  = None
    self._data_temp_gradients   = None
    self._data_temp_num_states  = 0
  
  def write(self, cargo: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Write cargo to the temporal DataTape.
    
      Args:
        cargo (tuple[np.ndarray, np.ndarray, np.ndarray]):
          A tuple of (sequentially in the tuple):
            1. Coordinates at current step, without any modification;
            2. Velocities at current step, without any modification;
            3. Potential gradients (the negative force) at current step, without any modification.
          All np.ndarrays are of shape (num_context_atoms, num_dofs_per_atom, ).
    """
    # Unpack cargo.
    coordinates, velocities, gradients = cargo

    coordinates = np.expand_dims(coordinates, axis=0)
    velocities  = np.expand_dims(velocities,  axis=0)
    gradients   = np.expand_dims(gradients,   axis=0)

    # Append values.
    self._data_temp_coordinates = coordinates if  self._data_temp_coordinates is None else \
                                  np.concatenate((self._data_temp_coordinates, coordinates), axis=0)
    self._data_temp_velocities  = velocities  if  self._data_temp_velocities  is None else \
                                  np.concatenate((self._data_temp_velocities, velocities), axis=0)
    self._data_temp_gradients   = gradients   if  self._data_temp_gradients   is None else \
                                  np.concatenate((self._data_temp_gradients, gradients), axis=0)
    self._data_temp_num_states += 1

  def integrate(self) -> None:
    """Integrate the recorded temporal data pool to the Onsager path actions. Will also clean up all
      currently recorded data.
    """
    # No integration if only two or less entries presents in the temperal data. 
    if self._data_temp_num_states < 2:
      self._data_temp_coordinates = None
      self._data_temp_velocities  = None
      self._data_temp_gradients   = None
      self._data_temp_num_states  = 0
      return 
    
    # Integrate Onsager path action and compute the integrated timesteps.
    S_action_nobias, S_length_nobias = self._integrate_nobias_impl()
    
    # Integrate Onsager path action and compute the integrated timesteps on the biased trajectory.
    S_action_biased, S_length_biased = self._integrate_biased_impl()
    
    # Convert to legal type.
    S_action_nobias = np.asarray([S_action_nobias], dtype=float)
    S_length_nobias = np.asarray([S_length_nobias], dtype=int  )
    S_action_biased = np.asarray([S_action_biased], dtype=float)
    S_length_biased = np.asarray([S_length_biased], dtype=int  )

    # Append values. 
    self._data_S_action_nobias=S_action_nobias if self._data_S_action_nobias is None else \
                               np.concatenate((self._data_S_action_nobias, S_action_nobias), axis=0)
    self._data_S_length_nobias=S_length_nobias if self._data_S_length_nobias is None else \
                               np.concatenate((self._data_S_length_nobias, S_length_nobias), axis=0)
    self._data_S_action_biased=S_action_biased if self._data_S_action_biased is None else \
                               np.concatenate((self._data_S_action_biased, S_action_biased), axis=0)
    self._data_S_length_biased=S_length_biased if self._data_S_length_biased is None else \
                               np.concatenate((self._data_S_length_biased, S_length_biased), axis=0)
    
    # Clean up buffers.
    self._data_temp_coordinates = None
    self._data_temp_velocities  = None
    self._data_temp_gradients   = None
    self._data_temp_num_states  = 0

  def _integrate_nobias_impl(self) -> tuple[float, int]:
    """The integration of the Onsager path action along a trajectory."""
    return _compute_onsager_path_integral(timestep_size=self._timestep_size, 
                                          friction_coef=self._friction_coef,
                                          inverse_beta =self._inverse_beta, 
                                          mass_per_dof =np.copy(self._mass_per_dof), 
                                          coordinates=np.copy(self._data_temp_coordinates), 
                                          velocities =np.copy(self._data_temp_velocities ), 
                                          gradients  =np.copy(self._data_temp_gradients  ), )
  
  def _integrate_biased_impl(self) -> tuple[float, int]:
    """The integration of the Onsager path action along a trajectory, potentially on a 
      modified Langevin dynamics trajectory. 
    """
    return 0., 0
  
  def serialize(self, to_file: str) -> None:
    """Serialize all taped data to file."""
    np.savez(to_file, 
             S_action_nobias=self._data_S_action_nobias, 
             S_length_nobias=self._data_S_length_nobias, 
             S_action_biased=self._data_S_action_biased,
             S_length_biased=self._data_S_length_biased, )



class ScaledOnsagerPathActionTape(OnsagerPathActionTape):
  """DataTape for recording the Onsager-Machlup action for the Langevin equation with scaled 
    potential gradients.
  """

  @classmethod
  def erase(cls, instance: Self) -> Self:
    """Replicate a new instance of the class with an empty data pool from an existing instance.
    
      Args:
        instance (ScaledOnsagerPathActionTape):
          The datatape instance to replicate from.
    """
    assert isinstance(instance, ScaledOnsagerPathActionTape), 'Illegal instance type.'

    return cls(timestep_size=instance._timestep_size, 
               friction_coef=instance._friction_coef, 
               inverse_beta =instance._inverse_beta, 
               mass_per_dof =instance._mass_per_dof, 
               scaling_coef =instance._scaling_coef, )

  def __init__(self, 
               timestep_size: float, 
               friction_coef: float, 
               inverse_beta: float, 
               mass_per_dof: np.ndarray, 
               scaling_coef: float, 
               ) -> None:
    """Create a DataTape for recoding the Onsager-Machlup action for the Langevin equation with
      scaled potential gradients. 
      NOTE: For use with scaled MD (all gradients in the system are scaled).

      Args:
        timestep (float):
          The stepsize for each timestep, unit: ps.
        friction_coef (float):
          The friction coefficient, unit: ps**-1.
        inverse_beta (float):
          The inverse $\beta$ ($\frac{1}{\beta} = k_{B}T$), unit: kcal/mol.
        mass_per_dof (np.ndarray):
          The per-DOF masses in the simulation Context, shape depends on Replica.
        scaling_coef (float):
          The gradient scaling factor.
    """
    OnsagerPathActionTape.__init__(self, 
                                   timestep_size=timestep_size, 
                                   friction_coef=friction_coef, 
                                   inverse_beta=inverse_beta,
                                   mass_per_dof=mass_per_dof, )
    
    # Sanity checks.
    assert isinstance(scaling_coef, float), "Illegal scaling_coef type."
    self._scaling_coef = scaling_coef
  
  def _integrate_biased_impl(self) -> tuple[float, int]:
    """The integration of the Onsager path action along a trajectory with scaled gradients."""
    scaled_gradients = self._scaling_coef * self._data_temp_gradients
    
    return _compute_onsager_path_integral(timestep_size=self._timestep_size, 
                                          friction_coef=self._friction_coef,
                                          inverse_beta =self._inverse_beta, 
                                          mass_per_dof =np.copy(self._mass_per_dof), 
                                          coordinates=np.copy(self._data_temp_coordinates), 
                                          velocities =np.copy(self._data_temp_velocities ), 
                                          gradients  =scaled_gradients, )



class ReplicaScaledOnsagerPathActionTape(OnsagerPathActionTape):
  """DataTape for recording the Onsager-Machlup action for the Langevin equation with scaled Replica
    potential gradients.
  """
  @classmethod
  def erase(cls, instance: Self) -> Self:
    assert isinstance(instance, ReplicaScaledOnsagerPathActionTape), "Illegal instance type."
    return cls(timestep_size=instance._timestep_size, 
               friction_coef=instance._friction_coef, 
               inverse_beta =instance._inverse_beta, 
               mass_per_dof =instance._mass_per_dof, 
               replica_atom_indices=instance._replica_atom_indices, 
               replica_scaling_per_dof=instance._replica_scaling_per_dof)
  
  def __init__(self, 
               timestep_size: float, 
               friction_coef: float, 
               inverse_beta:  float, 
               mass_per_dof:  np.ndarray, 
               replica_atom_indices: np.ndarray, 
               replica_scaling_per_dof: np.ndarray, 
               ) -> None:
    """Create a DataTape for recording the Onsager-Machlup action for the Langevin equation with 
      scaled Replica potential gradients.

      Args:
        timestep (float):
          The stepsize for each timestep, unit: ps.
        friction_coef (float):
          The friction coefficient, unit: ps**-1.
        inverse_beta (float):
          The inverse $\beta$ ($\frac{1}{\beta} = k_{B}T$), unit: kcal/mol.
        mass_per_dof (np.ndarray):
          The per-DOF masses in the simulation Context, shape depends on Replica.
        replica_atom_indices (np.ndarray):
          The indices of the Replica region atoms, shape (num_replica_atoms, ).
        replica_scaling_per_dof (np.ndarray):
          The scaling factor of the per-atom dofs potential gradients in the Replica region.
          shape (num_replica_atoms, num_dofs_per_atom).
    """
    OnsagerPathActionTape.__init__(self, 
                                   timestep_size=timestep_size, 
                                   friction_coef=friction_coef, 
                                   inverse_beta =inverse_beta, 
                                   mass_per_dof =mass_per_dof, )

    # Sanity checks.
    assert isinstance(replica_atom_indices, np.ndarray), "Illegal replica_atom_indices type."
    assert ( replica_atom_indices >= 0 ).all(),          "Illegal replica_atom_indices spec."
    self._replica_atom_indices = np.copy(replica_atom_indices)

    assert isinstance(replica_scaling_per_dof, np.ndarray), "Illegal replica_scaling_per_dof type."
    assert ( replica_scaling_per_dof >= 0. ).all(),         "Illegal replica_scaling_per_dof spec."
    self._replica_scaling_per_dof = np.copy(replica_scaling_per_dof)

    # Take only the Replica region.
    self._replica_mass_per_dof = np.copy(self._mass_per_dof[self._replica_atom_indices, :])

    assert self._replica_mass_per_dof.shape == self._replica_scaling_per_dof.shape, "Illegal mass shape inequal to the per-dof scaling."

  def write(self, cargo: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Write cargo to the temporal DataTape.
    
      Args:
        cargo (tuple[np.ndarray, np.ndarray, np.ndarray]):
          A tuple of (sequentially in the tuple):
            1. Coordinates at current step, without any modification;
            2. Velocities at current step, without any modification;
            3. Potential gradients (the negative force) at current step, without any modification.
          All np.ndarrays are of shape (num_context_atoms, num_dofs_per_atom, ).
    """
    # Unpack cargo.
    coordinates, velocities, gradients = cargo

    # Take only the replica region.
    coordinates = np.copy(coordinates[self._replica_atom_indices, :])
    velocities  = np.copy(velocities [self._replica_atom_indices, :])
    gradients   = np.copy(gradients  [self._replica_atom_indices, :])

    OnsagerPathActionTape.write(self, cargo=(coordinates, velocities, gradients))

  def _integrate_nobias_impl(self) -> tuple[float, int]:
    """The integration of the Onsager path action along a trajectory."""
    return _compute_onsager_path_integral(timestep_size=self._timestep_size, 
                                          friction_coef=self._friction_coef,
                                          inverse_beta =self._inverse_beta, 
                                          mass_per_dof =np.copy(self._replica_mass_per_dof), 
                                          coordinates=np.copy(self._data_temp_coordinates), 
                                          velocities =np.copy(self._data_temp_velocities ), 
                                          gradients  =np.copy(self._data_temp_gradients  ), )

  def _integrate_biased_impl(self) -> tuple[float, int]:
    scaled_gradients = self._replica_scaling_per_dof[np.newaxis, :] * self._data_temp_gradients
    
    return _compute_onsager_path_integral(timestep_size=self._timestep_size, 
                                          friction_coef=self._friction_coef, 
                                          inverse_beta =self._inverse_beta, 
                                          mass_per_dof =self._replica_mass_per_dof, 
                                          coordinates  =np.copy(self._data_temp_coordinates), 
                                          velocities   =np.copy(self._data_temp_velocities ), 
                                          gradients    =np.copy(scaled_gradients), )



class MassScaledOnsagerPathActionTape(OnsagerPathActionTape):
  """DataTape for recording the Onsager-Machlup action for the Langevin equation with scaled mass
    tensors.
  """

  @classmethod
  def erase(cls, instance: Self) -> Self:
    """Replicate a new instance of the class with an empty data pool from an existing instance.
    
      Args:
        instance (MassScaledOnsagerPathActionTape):
          The datatape instance to replicate from.
    """
    assert isinstance(instance, MassScaledOnsagerPathActionTape), "Illegal instance type."

    return cls(timestep_size=instance._timestep_size, 
               friction_coef=instance._friction_coef, 
               inverse_beta =instance._inverse_beta, 
               mass_per_dof =instance._mass_per_dof,
               replica_atom_indices=instance._replica_atom_indices, 
               replica_scaling_mass=instance._replica_scaling_mass, )
  
  def __init__(self, 
               timestep_size: float, 
               friction_coef: float, 
               inverse_beta:  float, 
               mass_per_dof:  np.ndarray, 
               replica_atom_indices: np.ndarray, 
               replica_scaling_mass: np.ndarray, 
               ) -> None:
    """Create a DataTape for recording the Onsager-Machlup action for the Langevin equation with 
      scaled mass tensors.
      NOTE: For use with mass tensor MD (Replica region atom masses in the system are scaled).

      Args:
        timestep (float):
          The stepsize for each timestep, unit: ps.
        friction_coef (float):
          The friction coefficient, unit: ps**-1.
        inverse_beta (float):
          The inverse $\beta$ ($\frac{1}{\beta} = k_{B}T$), unit: kcal/mol.
        mass_per_dof (np.ndarray):
          The per-DOF masses in the simulation Context, shape depends on Replica.
        replica_atom_indices (np.ndarray):
          The indices of the Replica region atoms, shape (num_replica_atoms, ).
        replica_scaling_mass (np.ndarray):
          The scaling factor of the per-atom masses Replica region, shape (num_replica_atoms, 
          num_dofs_per_atom, ).
    """
    OnsagerPathActionTape.__init__(self, 
                                   timestep_size=timestep_size, 
                                   friction_coef=friction_coef, 
                                   inverse_beta =inverse_beta, 
                                   mass_per_dof =mass_per_dof, )

    # Sanity checks.
    assert isinstance(replica_atom_indices, np.ndarray), "Illegal replica_atom_indices type."
    assert ( replica_atom_indices >= 0 ).all(),          "Illegal replica_atom_indices spec."
    self._replica_atom_indices = np.copy(replica_atom_indices)

    assert isinstance(replica_scaling_mass, np.ndarray), "Illegal replica_scaling_mass type."
    assert ( replica_scaling_mass > 0. ).all(),          "Illegal replica_scaling_mass spec."
    self._replica_scaling_mass = np.copy(replica_scaling_mass)

    # Take only the Replica region.
    self._mass_per_dof = np.copy(self._mass_per_dof[self._replica_atom_indices, :])

    assert self._mass_per_dof.shape == self._replica_scaling_mass.shape, \
           "Illegal mass tensor shape inequal to the scaling tensor."

  def write(self, cargo: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Write cargo to the temporal DataTape.
    
      Args:
        cargo (tuple[np.ndarray, np.ndarray, np.ndarray]):
          A tuple of (sequentially in the tuple):
            1. Coordinates at current step, without any modification;
            2. Velocities at current step, without any modification;
            3. Potential gradients (the negative force) at current step, without any modification.
          All np.ndarrays are of shape (num_context_atoms, num_dofs_per_atom, ).
    """
    # Unpack cargo.
    coordinates, velocities, gradients = cargo

    # Take only the replica region.
    coordinates = np.copy(coordinates[self._replica_atom_indices, :])
    velocities  = np.copy(velocities [self._replica_atom_indices, :])
    gradients   = np.copy(gradients  [self._replica_atom_indices, :])

    OnsagerPathActionTape.write(self, cargo=(coordinates, velocities, gradients))

  def _integrate_biased_impl(self) -> tuple[float, int]:
    """The integration of the Onsager path action along a trajectory with scaled masses."""
    scaled_mass_per_dof = self._replica_scaling_mass*self._mass_per_dof

    return _compute_onsager_path_integral(timestep_size=self._timestep_size, 
                                          friction_coef=self._friction_coef, 
                                          inverse_beta =self._inverse_beta, 
                                          mass_per_dof =scaled_mass_per_dof, 
                                          coordinates=np.copy(self._data_temp_coordinates), 
                                          velocities =np.copy(self._data_temp_velocities ), 
                                          gradients  =np.copy(self._data_temp_gradients  ), )
