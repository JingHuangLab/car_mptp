#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The base class of Samplers."""

from typing import Callable, TypeVar

from abc import ABC, abstractmethod

import numpy as np

from pycospath.rep import Replica

# Sampler. =========================================================================================

class Sampler(ABC):
  """The base class of Sampler."""

  def __init__(self, 
               fn_get_weighted_aligned:     Callable[[np.ndarray, np.ndarray], np.ndarray], 
               fn_get_rowwise_weighted_rms: Callable[[np.ndarray, np.ndarray], np.ndarray], 
               ) -> None:
    """Create a Sampler.
    
      Args:
        fn_get_weighted_aligned (Callable[[np.ndarray, np.ndarray], np.ndarray]):
          The get_weighted_aligned callable function. 
        fn_get_rowwise_weighted_rms (Callable[[np.ndarray, np.ndarray], np.ndarray]):
          The get_rowwise_weighted_rms callable function.
    """
    # Path colvar.
    self._path_colvar = None

    # Realize constructor prompt methods.
    assert callable(fn_get_weighted_aligned), "Illegal non-callable fn_get_weighted_aligned."
    self.get_weighted_aligned = fn_get_weighted_aligned

    assert callable(fn_get_rowwise_weighted_rms),"Illegal non-callable fn_get_rowwise_weighted_rms."
    self.get_rowwise_weighted_rms = fn_get_rowwise_weighted_rms
  
    self._sampler_strategy: TSamplerStrategy = None

  # Path colvar. -----------------------------------------------------------------------------------

  def set_path_colvar(self, path_colvar: np.ndarray) -> None:
    """Set the Path colvar."""
    self._path_colvar = np.copy(path_colvar)

  def get_path_colvar(self, in_place: bool = False) -> np.ndarray:
    """Get the path_colvar. If in_place == True, the returned Path colvar will share the same memory
      allocation with the internal self._path_colvar object.
    """
    assert not self._path_colvar is None, "Illegal None self._path_colvar."
    return np.copy(self._path_colvar) if in_place == False else self._path_colvar
  
  # Constructor prompt methods. --------------------------------------------------------------------

  def get_weighted_aligned(self,
                           array_to_refer: np.ndarray, 
                           array_to_align: np.ndarray, 
                           ) -> np.ndarray:
    """Prompt: Align array_to_align onto array_to_refer, return the aligned array_to_align."""
    raise RuntimeError("Prompt method not realized in Sampler.__init__().")

  def get_rowwise_weighted_rms(self,
                               array0: np.ndarray, 
                               array1: np.ndarray, 
                               ) -> np.ndarray:
    """Prompt: Get the weighted row-wise RMS distances between the arrays array0 and array1."""
    raise RuntimeError("Prompt method not realized in Sampler.__init__().")
  
  # External interfaces. ---------------------------------------------------------------------------

  def extend(self, replica: Replica) -> None:
    """To realize Sampler communication interfaces with Replica."""
    # Replica properties & operations realizations.
    self.get_whoami = replica.get_whoami
    self.get_context_mass_per_dof = replica.get_context_mass_per_dof
    self.get_replica_mass_per_dof = replica.get_replica_mass_per_dof

    ## Coordinates. 
    self.obtain_context_coordinates = replica.obtain_context_coordinates
    self.update_context_coordinates = replica.update_context_coordinates
    self.cast_to_replica_coordinates = replica.cast_to_replica_coordinates
    self.cast_to_context_coordinates = replica.cast_to_context_coordinates
    self.obtain_replica_coordinates = replica.obtain_replica_coordinates
    self.update_replica_coordinates = replica.update_replica_coordinates

    ## Velocities.
    self.initialize_context_velocities = replica.initialize_context_velocities
    self.obtain_context_velocities = replica.obtain_context_velocities
    self.update_context_velocities = replica.update_context_velocities
    self.cast_to_fullstep_velocities = replica.cast_to_fullstep_velocities
    self.cast_to_replica_velocities = replica.cast_to_replica_velocities
    self.cast_to_context_velocities = replica.cast_to_context_velocities
    self.obtain_replica_velocities = replica.obtain_replica_velocities
    self.update_replica_velocities = replica.update_replica_velocities

    ## Replica computables.
    self.compute_context_temperature = replica.compute_context_temperature
    self.compute_context_potential_gradients = replica.compute_context_potential_gradients
    self.compute_replica_potential_gradients = replica.compute_replica_potential_gradients
    self.md_execute_steps = replica.md_execute_steps

    # self.set_bonding_atom_indices = replica.set_bonding_atom_indices
    # self.get_bonding_atom_indices = replica.get_bonding_atom_indices
    # self.cast_bonding_to_context_velocities = replica.cast_bonding_to_context_velocities
    # self.cast_context_to_bonding_velocities = replica.cast_context_to_bonding_velocities

  # def set_bonding_atom_indices(self, indices) -> None:
  #   raise RuntimeError("Prompt method not realized in Sampler.extend().")

  # def get_bonding_atom_indices(self) -> np.ndarray:
  #   raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  # def cast_bonding_to_context_velocities(self, 
  #                                        bonding_velocities: np.ndarray, 
  #                                        context_velocities: object, ) -> object:
  #   raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  # def cast_context_to_bonding_velocities(self, context_velocities: object) -> np.ndarray:
  #   raise RuntimeError("Prompt method not realized in Sampler.extend().")

  # External interface prompts - Replica properties. 

  def get_whoami(self) -> int:
    """Prompt: Get the identity index of the Replica."""
    raise RuntimeError("Prompt method not realized in Sampler.extend().")

  def get_context_mass_per_dof(self) -> np.ndarray:
    """Prompt: Get a copy of the np.ndarray that holds the per-DOF masses in the simulation Context.
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def get_replica_mass_per_dof(self) -> np.ndarray:
    """Prompt: Get a copy of the np.ndarray that holds the per-DOF masses on the Replica region 
      atoms, shape (num_replica_dofs, ).
  """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")

  # External interface prompts - Replica coordinates. 
  
  def obtain_context_coordinates(self, asarray: bool = False) -> object | np.ndarray:
    """Prompt: Obtain the simulation Context coordinates.
    
      Args:
        asarray (bool, optional):
          If cast the returned simulation Context coordinates to np.ndarray.
          Default: False.

      Returns:
        context_coordinates (object | np.ndarray):
          The simulation Context coordinates. 
          If asarray==True,  returns a np.ndarray, shape (num_context_atoms, num_dofs_per_atom); 
          If asarray==False, returns the communicator-specific data structure.
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def update_context_coordinates(self, context_coordinates: object) -> None:
    """Prompt: Update the simulation Context coordinates.
    
      Args:
        context_coordinates (object):
          The simulation Context coordinates in the communicator-specific data structure.
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")

  def cast_to_replica_coordinates(self, context_coordinates: object) -> np.ndarray:
    """Prompt: Cast the simulation Context coordinates to Replica region coordinates.
    
      Args:
        context_coordinates (object):
          The simulation Context coordinates in the communicator-specific data structure.

      Returns:
        casted_replica_coordinates (np.ndarray):
          The Replica region coordinates, shape (num_replica_dofs, ).
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def cast_to_context_coordinates(self,
                                  replica_coordinates: np.ndarray, 
                                  context_coordinates: object, 
                                  ) -> object:
    """Prompt: Cast the Replica region coordinates to simulation Context coordinates. 

      Args:
        replica_coordinates (np.ndarray):
          The Replica region coordinates, shape (num_replica_dofs, ).
        context_coordinates (object):
          The simulation Context coordinates in the communicator-specific data structure.
      
      Returns:
        casted_context_coordinates (object):
          The simulation Context coordinates in the communicator-specific data structure with the 
          Replica region coordinates overridden.
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def obtain_replica_coordinates(self) -> np.ndarray:
    """Prompt: Obtain the Replica region coordinates.

      Returns: 
        replica_coordinates (np.ndarray):
          The Replica region coordinates, shape (num_replica_dofs, ).
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")

  def update_replica_coordinates(self, replica_coordinates: np.ndarray) -> None:
    """Prompt: Update the Replica region coordinates.

      Args:
        replica_coordinates: 
          The Replica region coordinates, shape (num_replica_dofs, ).
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  # External interface prompts - Replica velocities. 

  def initialize_context_velocities(self, temperature: float) -> None:
    """Prompt: Initialize the simulation Context velocities at designated temperature.
    
      Args:
        temperature (float):
          The temperature at which the simulation Context velocities are initialized, unit: kelvin.
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def obtain_context_velocities(self, asarray: bool = False) -> object | np.ndarray:
    """Prompt: Obtain the simulation Context velocities.
    
      Args:
        asarray (bool, optional):
          If cast the returned simulation Context velocities to np.ndarray.
          Default: False.

      Returns:
        context_velocities (object | np.ndarray):
          The simulation Context velocities.
          If asarray==True,  returns a np.ndarray, shape (num_context_atoms, num_dofs_per_atom); 
          If asarray==False, returns the communicator-specific data structure.
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def update_context_velocities(self, context_velocities: object) -> None:
    """Prompt: Update the simulation Context velocities.
    
      Args:
        context_velocities (object):
          The simulation Context velocities in the communicator-specific data structure.
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def cast_to_fullstep_velocities(self, 
                                  velocities:          np.ndarray, 
                                  mass_per_dof:        np.ndarray, 
                                  potential_gradients: np.ndarray, 
                                  ) -> np.ndarray:
    """Prompt: Cast the velocities to full timestep by time offseting.
    
      Args:
        velocities (np.ndarray):
          The velocities to be offset in time.
        mass_per_dof (np.ndarray):
          The per-DOF masses. 
        potential_gradients (np.ndarray):
          The potential gradients. 
      
      Returns:
        velocities_offset (np.ndarray):
          The velocities offset to full timestep. 
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def cast_to_replica_velocities(self, context_velocities: object) -> np.ndarray:
    """Prompt: Cast the simulation Context velocities to Replica region velocities.
    
      Args:
        context_velocities (object):
          The simulation Context velocities in the communicator-specific data structure.

      Returns:
        casted_replica_velocities (np.ndarray):
          The Replica region velocities, shape (num_replica_dofs, ).
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def cast_to_context_velocities(self, 
                                 replica_velocities: np.ndarray, 
                                 context_velocities: np.ndarray, 
                                 ) -> object:
    """Prompt: Cast the Replica region velocities to simulation Context velocities.

      Args:
        replica_velocities (np.ndarray):
          The Replica region velocities, shape (num_replica_dofs, ).
        context_velocities (object):
          The simulation Context velocities in the communicator-specific data structure.
      
      Returns:
        casted_context_velocities (object):
          The simulation Context velocities in the communicator-specific data structure with the 
          Replica region velocities overridden.
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def obtain_replica_velocities(self) -> np.ndarray:
    """Prompt: Obtain the Replica region velocities.

      Returns: 
        replica_velocities (np.ndarray):
          The Replica region velocities, shape (num_replica_dofs, ).
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")

  def update_replica_velocities(self, replica_velocities: np.ndarray) -> None:
    """Prompt:Update the Replica region velocities.

      Args:
        replica_velocities: 
          The Replica region velocities, shape (num_replica_dofs, ).
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  # External interface prompts - Replica computables. 

  def compute_context_temperature(self) -> float:
    """Prompt: Get the instantaneous temperature of the simulation Context at its current phase 
      state.
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  def compute_context_potential_gradients(self) -> np.ndarray:
    """Prompt: Get the potential gradients on the simulation Context coordinates. 
    
      Returns:
        context_gradients (np.ndarray):
          The gradients, unit: kcal/mol/Angstrom, shape (num_context_atoms, num_dofs_per_atom).
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")

  def compute_replica_potential_gradients(self) -> np.ndarray:
    """Prompt: Get the potential gradients on the Replica region coordinates. 
    
      Returns:
        context_gradients (np.ndarray):
          The gradients, unit: kcal/mol/Angstrom, shape (num_context_atoms, num_dofs_per_atom).
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  # External interface prompts - Replica MD runtime.
  
  def md_execute_steps(self, num_steps: int) -> None:
    """Prompt: Execute MD integration for num_steps steps.

      Args:
        num_steps (int):
          The number of MD steps to integrate.
    """
    raise RuntimeError("Prompt method not realized in Sampler.extend().")
  
  # Sampler Strategies. ----------------------------------------------------------------------------
  
  def md_execute_sampling(self, 
                          num_batches:         int, 
                          num_steps_per_batch: int = 1, 
                          ) -> None:
    """Execute one MD sampling Epoch. One MD Epoch consists of num_batches MD batches. Each MD batch
      consists of num_steps_per_batch steps. 

      Args:
        num_batches (int):
          The number of MD batches to execute. 
        num_steps_per_batch (int, optional):
          The number of MD steps to integrate in each batch. 
    """
    assert not self._sampler_strategy is None, "Illegal None self._sampler_strategy."
    self._sampler_strategy.md_execute_strategy(sampler=self, 
                                               num_batches=num_batches, 
                                               num_steps_per_batch=num_steps_per_batch, )
  
  @abstractmethod
  def set_sampler_strategy(self, strategy: str, **kwargs) -> None:
    """Set the SamplerStrategy adopted by this Sampler."""



# SamplerStrategy. =================================================================================

class SamplerStrategy(ABC):
  """The base class of MD SamplerStrategy."""

  def __init__(self) -> None:
    """Create an MD SamplerStrategy."""
    self._runtime_md_batch_counter = 0

  def get_num_md_batches(self) -> int:
    """Get the number of executed MD batches in the current MD epoch."""
    return self._runtime_md_batch_counter

  def md_execute_strategy(self, 
                          sampler: Sampler, 
                          num_batches:         int, 
                          num_steps_per_batch: int, 
                          ) -> None:
    """Execute the MD strategy with the Sampler for one Epoch. One MD Epoch consists of num_batches 
      MD batches. Each MD batch consists of num_steps_per_batch steps. 

      Args:
        sampler (Sampler):
          The agent Sampler.
        num_batches (int):
          The number of MD batches to execute. 
        num_steps_per_batch (int, optional):
          The number of MD steps to integrate in each batch. 
    """
    # Sanity checks.
    assert isinstance(num_batches, int),          "Illegal num_batches type."
    assert num_batches > 0,                       "Illegal num_batches spec."
    assert isinstance(num_steps_per_batch, int),  "Illegal num_steps_per_batch type."
    assert num_steps_per_batch > 0,               "Illegal num_steps_per_batch spec."

    # Sampling runtime.
    self.md_execute_on_epoch_begin(sampler=sampler)
    self._runtime_md_batch_counter = 0

    for _ in range(num_batches):
      self.md_execute_on_batch_begin(sampler=sampler)
      sampler.md_execute_steps(num_steps=num_steps_per_batch)
      self._runtime_md_batch_counter += 1
      self.md_execute_on_batch_end(sampler=sampler)
    
    self.md_execute_on_epoch_end(sampler=sampler)
  
  @abstractmethod
  def md_execute_on_epoch_begin(self, sampler: Sampler) -> None:
    """SamplerStrategy performed at the beginning of the MD epoch."""

  @abstractmethod
  def md_execute_on_batch_begin(self, sampler: Sampler) -> None:
    """SamplerStrategy performed at the beginning of the MD batch."""

  @abstractmethod
  def md_execute_on_batch_end(self, sampler: Sampler) -> None:
    """SamplerStrategy performed at the end of the MD batch."""

  @abstractmethod
  def md_execute_on_epoch_end(self, sampler: Sampler) -> None:
    """SamplerStrategy performed at the end of the MD epoch."""



TSamplerStrategy = TypeVar("TSamplerStrategy", bound=SamplerStrategy)
