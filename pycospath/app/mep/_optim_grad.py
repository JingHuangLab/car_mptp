#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The Gradient-based PathOptimizers."""

import numpy as np

from pycospath.app.mep import PathOptimizer

# Standard Gradient Descent. =======================================================================

class StdGradientDescentPathOptimizer(PathOptimizer):
  """The standard Gradient Descent PathOptimizer."""

  def __init__(self, 
               grad_lr:         float = 0.01,
               grad_lr_scaling: float = 1., 
               config_path_max_gradient: float = 5., 
               config_path_fix_mode:     str = 'both', 
               ) -> None:
    """Initialize a standard Gradient Descent PathOptimizer.

      Args:
        grad_lr (float, optional):
            The learning rate of the PathOptimizer. 
            Default: 0.01.
        grad_lr_scaling (float, optional):
            The scaling factor of the learning rate after each descent step. 
            Default: 1.0.
            grad_lr_scaling = 1., constant learning rates;
            grad_lr_scaling < 1., decreasing learning rates. 
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
    PathOptimizer.__init__(self, 
                           config_path_max_gradient=config_path_max_gradient, 
                           config_path_fix_mode=config_path_fix_mode, )
    
    assert isinstance(grad_lr, float), "Illegal grad_lr type."
    assert grad_lr > 0.,               "Illegal grad_lr spec."

    self._grad_lr = grad_lr

    assert isinstance(grad_lr_scaling, float), "Illegal grad_lr_scaling type."
    assert grad_lr_scaling > 0.,               "Illegal grad_lr_scaling spec."
    
    self._grad_lr_scaling = grad_lr_scaling

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
    # Descend for one step.
    path_colvar_optimized = path_colvar - self._grad_lr * path_gradients

    # Adaptive learning rates schedule.
    self._grad_lr *= self._grad_lr_scaling

    return path_colvar_optimized


