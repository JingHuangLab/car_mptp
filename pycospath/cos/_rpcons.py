#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""Reaction Path with Holonomic Constraints."""

import numpy as np

from pycospath.utils.geometry import get_rowwise_projection

from pycospath.cos import CoS

class RPCons(CoS):
  """Implements the Reaction Path with Holonomic Constraints (RPCons) method.
  
    Reaction Path Optimization with Holonomic Constraints and Kinetic Energy Potentials. 
    J. Brokaw, K. Haas, J.-W. Chu, J. Chem. Theory Comput. 2008, 5 2050-2061. DOI:10.1021/ct9001398

    Note that the RPCons constraint without gradient projections will not converge to the MEP. 
    The kinetic energy potential proposed in this work is not *yet implemented. 
  """

  CONSTRAINT_BASED = True
  RESTRAINT_BASED  = True # For gradient projections

  def __init__(self,
               use_gradient_projection: bool  = False,
               cons_convergence_thresh: float = 1e-8,
               cons_convergence_maxitr: int   = 200,
               ) -> None:
    """Initializes a Reaction Path with Holonomic Constraints CoS condition 
      instance.
    
      Args:
        use_gradient_projection (bool, optional):
          Specifies if Gradient Projection should be used for the RPCons method. Note that without 
          gradient projections the RPCons method will *never* converge to the MEP. The default 
          option refers to the method \'described\' in the original work. 
          Default: False;
        cons_convergence_thresh (float, optional):
          The threshold (in \AA) under which the holonomic constraints on the equal RMS distances 
          between all pairs of adjacent Replicas are viewed as converged. 
          Default: 1e-8;
        cons_convergence_maxitr (int, optional):
          The maximum number of iteration that the Lagrange multipliers solver are allowed to update 
          the Replica coordinates towards the equal RMS constraint. 
          Default: 200;
    """
    self._use_grad_proj = True if use_gradient_projection==True else False
    self._cons_thresh = cons_convergence_thresh
    self._cons_maxitr = cons_convergence_maxitr

  def apply_restraint(self, 
                      path_colvar:    np.ndarray, 
                      path_gradients: np.ndarray,
                      **kwargs) -> np.ndarray:
    """Apply the RPCons restraint condition on the Path gradients.
    
      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
        path_gradients (np.ndarray):
          The Path gradients, shape (num_replicas, num_replica_dofs).
    
      Returns:
        restrained_path_gradients (np.ndarray):
          The restrained Path gradients, shape (num_replicas, num_replica_dofs).
    """
    # Apply gradient projection.
    if self._use_grad_proj == True:
      path_tan = self.get_path_tangent(path_colvar=path_colvar)
      path_gradients[1:-1] -= get_rowwise_projection(array_to_refer=path_tan[1:-1], 
                                                     array_to_project=path_gradients[1:-1], )
      
    return path_gradients

  def apply_constraint(self, 
                       path_colvar: np.ndarray, 
                       **kwargs) -> np.ndarray:
    """Apply the RPCons constraint condition on the Path colvar. 

      Args:
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
    
      Returns:
        constrained_path_colvar (np.ndarray):
          The constrained Path colvar, shape (num_replicas, num_replica_dofs).
    """
    # Control varibles for the iterative updates of Constrained coordinates. 
    cons_convergence = self._cons_thresh + 1.
    cons_num_iterations = 0

    # Determine the target RMS. 
    cons_rms = np.mean(self.get_path_weighted_rms(path_colvar=path_colvar))

    # Scratches of the path colvars at context iterations.
    prev_pcv = np.copy(path_colvar) # (n-1)-th step.
    curr_pcv = np.copy(path_colvar) #   (n)-th step.

    while cons_convergence >= self._cons_thresh and cons_num_iterations < self._cons_maxitr:
      # The Path colvar RMS and RMS gradients at the current (n)-th step. 
      curr_rms, curr_rms_grad = self.get_path_weighted_rms(path_colvar=curr_pcv, return_grad=True, )
      # The Path colvar         RMS gradients at the previous (n-1)-th step.
      _,        prev_rms_grad = self.get_path_weighted_rms(path_colvar=prev_pcv, return_grad=True, )

      # LHS of the constraint equation -> coefficient matrix on lambdas.
      lamd_coefs = self._cons_lamd_coefs(curr_rms_grad, prev_rms_grad)

      # RHS of the constraint equation -> column of scalars as the cost.
      cost_rms = cons_rms - curr_rms

      # Solve for the lambdas.
      lamds = np.linalg.solve(lamd_coefs, cost_rms)

      # Update the coordinates.
      curr_pcv[1:-1] = curr_pcv[1:-1] - lamds[0:-1, np.newaxis] * curr_rms_grad[0:-1] \
                                      + lamds[1:  , np.newaxis] * curr_rms_grad[1:  ]
      
      # Convergence detection.
      rms_new = self.get_path_weighted_rms(path_colvar=curr_pcv)
      cons_convergence = np.max(np.abs(cons_rms - rms_new)) 
      cons_num_iterations += 1
    
    return curr_pcv

  def _cons_lamd_coefs(self,
                       curr_rms_grad: np.ndarray,
                       prev_rms_grad: np.ndarray,
                       ) -> np.ndarray:
    """Return the LHS of the constraint equation as a matrix of coefficients on the Lagrange 
      multipliers.
    """
    # No. lambdas = num_replicas - 1 (i.e., the no. RMS distances).
    num_lamd = curr_rms_grad.shape[0]

    # The previous step RMS gradients on the Path colvars. 
    coef_l_iminus1 = -1. * prev_rms_grad[0:num_lamd-1]
    coef_l_i       =  2. * prev_rms_grad[ :        ]
    coef_l_iplus1  = -1. * prev_rms_grad[1:        ]

    # coefficient on the lambdas - a tridiagonal matrix.
    # NOTE: Both rms_grad and path_tan on one replica is one dimenstion and thus a sum along axis=1 
    #       on the element wise products is implemented. 
    ## upper/on/lower-diagonal:
    coef_lamd_iplus1  = np.sum(curr_rms_grad[0:-1] * coef_l_iplus1,  axis=1)
    coef_lamd_i       = np.sum(curr_rms_grad[ :  ] * coef_l_i,       axis=1)
    coef_lamd_iminus1 = np.sum(curr_rms_grad[1:  ] * coef_l_iminus1, axis=1)

    # Build the coefficient matrix of shape (num_replicas-1, num_replicas-1).
    lamd_coef_mat = np.zeros((num_lamd, num_lamd))
    lamd_coef_mat.ravel()[num_lamd::num_lamd+1] = coef_lamd_iminus1
    lamd_coef_mat.ravel()[     0::num_lamd+1] = coef_lamd_i
    lamd_coef_mat.ravel()[     1::num_lamd+1] = coef_lamd_iplus1

    return lamd_coef_mat


