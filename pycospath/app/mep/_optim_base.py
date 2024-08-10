#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The base class of PathOptimizer."""

from abc import ABC, abstractmethod

import numpy as np

class PathOptimizer(ABC):
  """The abstract class for PathOptimizer."""

  PATH_FIX_MODE = ['none', 'both', 'head', 'tail']

  def __init__(self, 
               config_path_max_gradient: float = 5., 
               config_path_fix_mode: str = 'both', 
               ) -> None:
    """Create a PathOptimizer.
    
      Args:
        config_path_max_gradient (float, optional):
          The gradient clipping threshold. If the largest unsigned component in the Path gradients 
          is larger than config_path_max_gradient, the entire Path gradients are uniformly scaled 
          down to have config_path_max_gradient as its largest unsigned component. 
          Default 5. kcal/mol/A, allowed values are positive floats. 
        config_path_fix_mode (str, optional):
          Specifies if the end points on the Path colvar are to be fixed during the optimization.
          Defualt: 'both'.
            'both': Coodinates of all replicas except the two end-point replicas (the first and the 
                    last) will be optimized;
            'head': Coordinates of all replicas except the first replica will be optimized;
            'tail': Coordinates of all replicas except the  last replica will be optimized; 
            'none': Coordinates of all replicas will be optimized. 
    """
    # Sanity checks.
    assert isinstance(config_path_max_gradient, float), "Illegal config_path_max_gradient type."
    assert config_path_max_gradient > 0.,               "Illegal config_path_max_gradient spec."

    self._config_path_max_gradient = config_path_max_gradient

    assert isinstance(config_path_fix_mode, str),      "Illegal config_path_fix_mode type."
    assert config_path_fix_mode in self.PATH_FIX_MODE, "Illegal config_path_fix_mode spec."

    self._config_path_fix_mode = config_path_fix_mode

  def apply_path_fixing(self, 
                        path_colvar:         np.ndarray, 
                        path_colvar_descent: np.ndarray, 
                        ) -> np.ndarray:
    """Applies the Path colvar fixing scheme: Fixes the end-point Replica colvars if specified.
    
      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
        path_colvar_descent (np.ndarray):
          The optimized Path colvar, shape (num_replicas, num_replica_dofs).
      
      Returns:
        path_colvar_descent_fixed (np.ndarray):
          The optimized Path colvar and with Path fixing, shape (num_replicas, num_replica_dofs).
    """
    # Apply the Path colvar fixing scheme.
    path_colvar_descent_fixed = np.copy(path_colvar_descent)

    if self._config_path_fix_mode in ['both', 'head']:
      path_colvar_descent_fixed[ 0, :] = np.copy(path_colvar[ 0, :])
    
    if self._config_path_fix_mode in ['both', 'tail']:
      path_colvar_descent_fixed[-1, :] = np.copy(path_colvar[-1, :])

    return path_colvar_descent_fixed

  def apply_path_descent(self, 
                         path_colvar:    np.ndarray, 
                         path_energies:  np.ndarray, 
                         path_gradients: np.ndarray, 
                         ) -> np.ndarray:
    """Applies the Path colvar descent scheme: Optimizes the Path colvar for one step.
    
      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape shape (num_replicas, num_replica_dofs).
        path_energies (np.ndarray):
          The Path energy on the Path colvar, shape (num_replicas, ).
        path_gradients (np.ndarray):
          The Path gradients on the Path colvar, shape (num_replicas, num_replica_dofs).
      
      Returns:
        path_colvar_descent (np.ndarray):
          The optimized Path colvar, shape (num_replicas, num_replica_dofs).
    """
    # Apply the Path gradient clipping scheme.
    path_grad_max_val = np.amax(np.absolute(path_gradients))

    if path_grad_max_val > self._config_path_max_gradient:
      path_gradients *= (self._config_path_max_gradient / path_grad_max_val)

    # Actual optimization.
    path_colvar_descent = self.apply_path_descent_impl(path_colvar=path_colvar,
                                                       path_energies=path_energies, 
                                                       path_gradients=path_gradients, )
    
    return path_colvar_descent
  
  @abstractmethod
  def apply_path_descent_impl(self, 
                              path_colvar:    np.ndarray, 
                              path_energies:  np.ndarray, 
                              path_gradients: np.ndarray, 
                              ) -> np.ndarray:
    """The implementation of the Path colvar descent scheme.
    
      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
        path_energies (np.ndarray):
          The Path energy on the Path colvar, shape (num_replicas, ).
        path_gradients (np.ndarray):
          The Path gradients on the Path colvar, shape (num_replicas, num_replica_dofs).
    
      Returns:
        path_colvar_descent (np.ndarray):
          The optimized Path colvar, shape (num_replicas, num_replica_dofs).
    """


