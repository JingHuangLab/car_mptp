#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The base class of Chain-of-States methods."""

from typing import Union, Callable

from abc import ABC, abstractmethod

import numpy as np

class CoS(ABC):
  """The abstract class of a Chain-of-States algorithm."""

  CONSTRAINT_BASED = False
  RESTRAINT_BASED  = False

  # External interfaces. ---------------------------------------------------------------------------

  def implement(self, 
                fn_get_path_tangent:      Callable[[np.ndarray], np.ndarray], 
                fn_get_path_weighted_rms: Callable[[np.ndarray, bool], 
                                                   Union[np.ndarray,tuple[np.ndarray, np.ndarray]]], 
                ) -> None:
    """To realize the Path functions."""
    # Sanity checks.
    assert callable(fn_get_path_tangent),      "Illegal non-callable fn_get_path_tangent."
    self.get_path_tangent = fn_get_path_tangent

    assert callable(fn_get_path_weighted_rms), "Illegal non-callable fn_get_path_weighted_rms."
    self.get_path_weighted_rms = fn_get_path_weighted_rms
  
  # External interface prompts - Path computables. 

  def get_path_tangent(self, 
                       path_colvar: np.ndarray,
                       ) -> np.ndarray:
    """Prompt: Get the Path tangent vector."""
    raise RuntimeError("Prompt method not realized in CoS.implement().")
  
  def get_path_weighted_rms(self, 
                            path_colvar: np.ndarray, 
                            return_grad: bool = False, 
                            ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """PromptL Get the Path rms vector."""
    raise RuntimeError("Prompt method not realized in CoS.implement().")
  
  # CoS runtime. -----------------------------------------------------------------------------------

  @classmethod
  def is_restraint_based(cls) -> bool:
    """Returns if the CoS Path condition is based on restraint potentials. Restraints in here refer 
      to any modification to the Path potential gradients in the most general sense (I.e., includes 
      gradient projection) that happens before the Path evolution.
    """
    return cls.RESTRAINT_BASED

  @classmethod
  def is_constraint_based(cls) -> bool:
    """Returns if the CoS Path condition is based on constraint coordinates. Constraints in here 
      refer to any modification to the Path colvar in the most general sense that happens after the 
      Path evolution. 
    """
    return cls.CONSTRAINT_BASED
  
  @abstractmethod
  def apply_restraint(self,
                      path_colvar:    np.ndarray, 
                      path_energies:  np.ndarray, 
                      path_gradients: np.ndarray, 
                      ) -> np.ndarray:
    """Apply the CoS restraint condition on the Path gradients.gradient. 

      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
        path_energies (np.ndarray):
          The Path energies, shape (num_replicas, ).
        path_gradients (np.ndarray):
          The Path gradients, shape (num_replicas, num_replica_dofs).
    
      Returns:
        restrained_path_gradients (np.ndarray):
        The restrained Path gradients, shape (num_replicas, num_replica_dofs).
    """
    raise NotImplementedError("To be overloaded by subclasses.")

  @abstractmethod
  def apply_constraint(self,
                       path_colvar:   np.ndarray, 
                       path_energies: np.ndarray, 
                       ) -> np.ndarray:
    """Apply the CoS constraint condition on the Path colvar. 

      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
        path_energies (np.ndarray):
          The Path energies, shape (num_replicas, ).

      Returns:
        constrained_path_colvar (np.ndarray):
          The constrained Path colvar, shape (num_replicas, num_replica_dofs).
    """
    raise NotImplementedError("To be overloaded by subclasses.")


