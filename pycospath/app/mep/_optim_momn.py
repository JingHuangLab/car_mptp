#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The Momentum-based PathOptimizers."""

import numpy as np

from pycospath.app.mep import PathOptimizer

# Standard Adaptive Momentum. ======================================================================

class StdAdaptiveMomentumPathOptimizer(PathOptimizer):
  """The standard Adaptive Momentum PathOptimizer.
  
    Adam: A Method for Stochastic Optimization.
    D. P. Kingma, J. Ba, 3rd International Conference on Learning Representations, 2015, San Diego, 
    CA, USA.
  """

  def __init__(self, 
               adam_lr:      float = .01,  
               adam_beta1:   float = .9,   
               adam_beta2:   float = .999, 
               adam_epsilon: float = 1e-8, 
               config_path_max_gradient: float = 5., 
               config_path_fix_mode:     str = 'both', 
               ) -> None:
    """Initialize a standard Adaptive Momentum Path geometric optimization Evolver

      Args:
        adam_lr (float, optional):
            The learning rate of the Path Ecolver. 
            Default: 0.01
        adam_beta1 (float, optional):
            The AdaM $\beta_1$. 
            Default: 0.9.
        adam_beta2 (float, optional):
            The AdaM $\beta_2$. 
            Default: 0.999.
        adam_epsilon (float, optional): 
            The AdaM $\epsilon$. 
            Default: 1e-8.
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
    
    assert isinstance(adam_lr, float), "Illegal adam_lr type."
    assert adam_lr > 0.,               "Illegal adam_lr spec."

    self._adam_lr = adam_lr

    self._adam_beta1   = adam_beta1
    self._adam_beta2   = adam_beta2
    self._adam_epsilon = adam_epsilon
    self._adam_i_step  = 0          # Adam step counter.

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
    if self._adam_i_step == 0:  # initialize AdaM M and V at first call.
        self._am_M = 0
        self._am_V = 0

    self._adam_i_step += 1

    # step1: 
    # M^{t+1} = beta_1 * M^{t} + (1-beta_1) * grad;
    self._am_M = self._adam_beta1 * self._am_M + (1-self._adam_beta1) * path_gradients

    # step2: 
    # V^{t+1} = beta_2 * V^{t} + (1-beta_2) * grad**2;
    self._am_V = self._adam_beta2 * self._am_V + (1-self._adam_beta2) * path_gradients**2

    # step3: 
    # M^{hat,t+1} = M^{t+1} / (1 - beta_1**t);
    Mh = self._am_M / (1 - self._adam_beta1**self._adam_i_step)

    # step4: 
    # V^{hat,t+1} = V^{t+1} / (1 - beta_2**t);
    Vh = self._am_V / (1 - self._adam_beta2**self._adam_i_step)

    # step5: 
    # x^(t+1) = x^{t} - eta * M^{hat,t+1} / (V^{hat,t+1} ** 0.5  + epsilon);
    path_colvar_optimized = path_colvar - self._adam_lr * Mh / (Vh**0.5 + self._adam_epsilon)

    return path_colvar_optimized


