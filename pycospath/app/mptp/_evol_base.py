#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The base class of PathEvolvers."""

from typing import Callable

from abc import ABC, abstractmethod

import numpy as np

from pycospath.samp import Sampler

class PathEvolver(ABC):
  """The abstract class for PathEvolver."""

  CONFIG_PATH_FIX_MODE = ['none', 'both', 'head', 'tail']

  def __init__(self, 
               config_path_fix_mode: str = 'both', 
               ) -> None:
    """Create a PathEvolver.
    
      Args:
        config_path_fix_mode (str, optional):
          Specifies if the end points on the Path colvar are to be fixed during the evolution.
          Defualt: 'both'.
            'both': Coodinates of all replicas except the two end-point replicas (the first and the 
                    last) will be evolved;
            'head': Coordinates of all replicas except the first replica will be evolved;
            'tail': Coordinates of all replicas except the  last replica will be evolved; 
            'none': Coordinates of all replicas will be evolved. 
    """
    # Sanity checks.
    assert isinstance(config_path_fix_mode, str),             "Illegal config_path_fix_mode type."
    assert config_path_fix_mode in self.CONFIG_PATH_FIX_MODE, "Illegal config_path_fix_mode spec."
    self._config_path_fix_mode = config_path_fix_mode

  # External interfaces. ---------------------------------------------------------------------------

  def implement(self, 
                fn_get_weighted_aligned:     Callable[[np.ndarray, np.ndarray], np.ndarray], 
                fn_get_rowwise_weighted_rms: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                ) -> None:
    """To realize Evolver communication functions."""
    # Weighted aligned function.
    assert callable(fn_get_weighted_aligned), "Illegal non-callable fn_get_weighted_aligned."
    self.get_weighted_aligned = fn_get_weighted_aligned

    # Rowwise weighted RMS function.
    assert callable(fn_get_rowwise_weighted_rms),"Illegal non-callable fn_get_rowwise_weighted_rms."
    self.get_rowwise_weighted_rms = fn_get_rowwise_weighted_rms
  
  # External interface prompts - get_weighted_aligned_fn & get_weighted_rowwise_rms methods. 

  def get_weighted_aligned(self,
                           array_to_refer: np.ndarray, 
                           array_to_align: np.ndarray, 
                           ) -> np.ndarray:
    """Prompt: Align array_to_align onto array_to_refer, return the aligned array_to_align."""
    raise RuntimeError("Prompt method not realized in PathEvolver.implement().")

  def get_rowwise_weighted_rms(self,
                               array0: np.ndarray, 
                               array1: np.ndarray, 
                               ) -> np.ndarray:
    """Prompt: Get the weighted row-wise RMS distances between the arrays array0 and array1."""
    raise RuntimeError("Prompt method not realized in PathEvolver.implement().")
  
  # PathEvolver runtime. ---------------------------------------------------------------------------

  def apply_path_fixing(self, 
                        path_colvar:         np.ndarray, 
                        path_colvar_evolved: np.ndarray, 
                        ) -> np.ndarray:
    """Applies the Path colvar fixing scheme: Fixes the end-point Replica colvars if specified.
    
      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
        path_colvar_evolved (np.ndarray):
          The evolved Path colvar, shape (num_replicas, num_replica_dofs).
      
      Returns:
        path_colvar_evolved_fixed (np.ndarray):
          The evolved Path colvar and with Path fixing, shape (num_replicas, num_replica_dofs).
    """
    # Apply the Path colvar fixing scheme.
    path_colvar_evolved_fixed = np.copy(path_colvar_evolved)

    if self._config_path_fix_mode in ['both', 'head']:
      path_colvar_evolved_fixed[ 0, :] = np.copy(path_colvar[ 0, :])
    
    if self._config_path_fix_mode in ['both', 'tail']:
      path_colvar_evolved_fixed[-1, :] = np.copy(path_colvar[-1, :])

    return path_colvar_evolved_fixed
  
  # PathEvolver runtime per Sampler. ---------------------------------------------------------------
  # NOTE: for MPI impls.
  
  @abstractmethod
  def create_sampler(self,) -> Sampler:
    """Create a PathEvolver-compatible Sampler."""

  @abstractmethod
  def get_evolved_replica_coordinates(self, sampler: Sampler) -> np.ndarray:
    """Returns the evolved Replica region coordinates from the sampled statistics in the Sampler.

      Args:
        sampler (Sampler):
          The Sampler.

      Returns:
        evolved_replica_coordinates (np.ndarray):
          The evolved Replica region coordinates of the sampled ensemble, shape (num_replica_dofs,).
    """


