#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""Constant Advance Replicas method."""

import numpy as np

from pycospath.utils.geometry import get_rowwise_cosine

from pycospath.cos import CoS

class CAR(CoS):
  """Implements the Constant Advance Replicas method.
  
    Constant advance replicas method for locating minimum energy paths and transition states.
    Z. Song, Y. Ding, J. Huang, J. Comput. Chem. 2023, 44, 2042-2057. DOI:10.1002/jcc.27178
  """

  CONSTRAINT_BASED = True
  RESTRAINT_BASED  = False

  def __init__(self, 
               cons_convergence_thresh: float = 1e-8,
               cons_convergence_maxitr: int   = 10,
               cons_regulr_curv_thresh: float = 30.,
               cons_regulr_grow_thresh: float = 1.25,
               cons_regulr_grow_dscale: float = .8, 
               ) -> None:
    """Initializes a Constant Advance Replicas CoS condition instance. 

      Args:
        cons_convergence_thresh (float, optional):
          The threshold (in \AA) under which the holonomic constraints on the equal RMS distances 
          between all pairs of adjacent Replicas are viewed as converged. 
          Default: 1e-8;
        cons_convergence_maxitr (int, optional):
          The maximum number of iteration that the Lagrange multipliers solver are allowed to update
          the Replica coordinates towards the equal RMS constraint. 
          Default: 10;
        cons_regulr_curv_thresh (float, optional):
          The threshold (in \circ) above which the Path curvature regularization protocol for 
          preventing large kinks is enabled. This regularization scheme explicitly controls the 
          maximum angle between two conformation vectors $r_{i-1}-r_{i+1}$ and $r{i-1}-r{i}$ where 
          $r_{i-1}$, $r_{i}$, $r_{i+1}$ are the Replica vectors of any three neighboring Replica on 
          the Path.
          Default: 30., allowed values are (0, 45];
        cons_regulr_grow_thresh (float, optional):
          The threshold above which the Path growth regularization protocol for preventing 
          overstepping of the Lagrange multiplier updates is enabled. This regularization scheme 
          explicitly monitors if the update step of the Lagrange multipliers have exceeded the RMS 
          distance between any adjacent pairs by this threshold factor, in which case the Lagrange 
          multipliers are uniformly scaled-down under this threshold using a line search scheme. 
          Default: 1.25, allowed values are (1., 1.8].
        cons_regulr_grow_dscale (float, optional):
          The down-scaling factor for the Path growth regularization protocol. 
          Default: .8, allowed values are (0., 1.).
    """
    self._cons_thresh = cons_convergence_thresh
    self._cons_maxitr = cons_convergence_maxitr
    self._regular_curv_thresh = np.cos(cons_regulr_curv_thresh / 180. * np.pi)
    self._regular_grow_thresh = cons_regulr_grow_thresh
    self._regular_grow_dscale = cons_regulr_grow_dscale

  def apply_restraint(self, **kwargs) -> None:
    raise NotImplementedError("CAR is constraint-based.")

  def apply_constraint(self, 
                       path_colvar: np.ndarray, 
                       **kwargs) -> np.ndarray:
    """Apply the Chain-of-States (CoS) constraint condition on the Path colvar. 

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

    while cons_convergence >= self._cons_thresh and cons_num_iterations < self._cons_maxitr:
      # Path curvature regularization.
      path_colvar = self._path_curvature_regulr(path_colvar=path_colvar)

      # Vars needed to setup the constraint equations.
      path_rms, path_rms_grad = self.get_path_weighted_rms(path_colvar=path_colvar,return_grad=True)
      path_tangent = self.get_path_tangent(path_colvar=path_colvar)
      
      # LHS of the constraint equation -> coefficient matrix on lambdas. 
      lamd_coefs = self._cons_lamd_coefs(path_rms_grad=path_rms_grad, path_tangent=path_tangent)

      # RHS of the constraint equation -> column of scalars as the cost.
      targ_rms = np.mean(path_rms)
      cost_rms = targ_rms - path_rms

      # Solve for the lambdas.
      # 0. QR decomp: Q -> orthogonal; and R -> upper-triangular.
      q_lamd_coefs, r_lamd_coefs = np.linalg.qr(a=lamd_coefs, mode='reduced')
      lamds = np.linalg.solve(r_lamd_coefs, q_lamd_coefs.T @ cost_rms)

      # Path growth regularization.
      lamds = self._path_growth_regulr(lamds=lamds, 
                                       path_colvar=path_colvar, 
                                       path_tangent=path_tangent, 
                                       path_rms=path_rms)
      
      # Update the coordinates. 
      path_colvar[1:-1] += path_tangent[1:-1] * lamds[:, None]

      # Convergence detection.
      rms_new = self.get_path_weighted_rms(path_colvar=path_colvar)
      cons_convergence  = np.max(np.abs(targ_rms - rms_new))
      cons_num_iterations += 1

    return path_colvar

  def _path_curvature_regulr(self, 
                             path_colvar: np.ndarray, 
                             ) -> np.ndarray:
    """Implements the Path curvature regularization protocol to dekink the Path, returns the 
      dekinked Path colvar.
    """
    # For each intermediate Replica $r_{i}$, if the angle between the conformation vectors
    # $r_{i-1}-r_{i+1}$ and $r_{i-1}-r_{i}$ is larger than the predefined thresholds, the Replica is 
    # directly set to be the mid-point between its adjacent neighbors. 
    # cosine:
    diff_0 = path_colvar[0:-2] - path_colvar[1:-1] # The r_{i-1} - r_{i}   vecs.
    diff_1 = path_colvar[0:-2] - path_colvar[2:  ] # The r_{i-1} - r_{i+1} vecs.
    cosines = get_rowwise_cosine(array0=diff_0, array1=diff_1) # Compute cosine row-wise.

    # If Replica cosines exceed the threshold.
    i_rep_to_regulr: np.ndarray = np.where(cosines <= self._regular_curv_thresh)[0]
    if i_rep_to_regulr.size != 0: # regularization is needed.
      cvec_mid = (path_colvar[0:-2] + path_colvar[2: ]) / 2.
      path_colvar[i_rep_to_regulr+1] = cvec_mid[i_rep_to_regulr] # Note index shift.
    return path_colvar

  def _path_growth_regulr(self, 
                          lamds:        np.ndarray,
                          path_colvar:  np.ndarray,
                          path_tangent: np.ndarray,
                          path_rms:     np.ndarray,
                          ) -> np.ndarray:
    """Implements the Path growth regularization protocol to prevent overstep of the Lagrange 
      multipliers and essentially to generate better preconditions such that the constrained 
      coordinates can be solved in subsequent updates.
    """
    # Regularize the step size of lambda updates to prevent extremely large step from taken, which 
    # leads to possible switch of replicas' geomtrical order. That is, for one or some of the 
    # replicas, their neighboring replicas are no more their nearest neighbors, causing the 
    # formation of "circular" segments which prevents subsequent coordinates updates to  converge. 
    # In almost all cases, we see that extremely large values of lambdas appear and the coordinates 
    # would become np.inf and then NaN, which led to a SVD related exception from numpy. 
    rms_thresh = path_rms * self._regular_grow_thresh
    lamd_to_regulr = True

    while lamd_to_regulr == True:
      # 1. Compute RMS after the trial move.
      path_colvar_trial = np.copy(path_colvar)
      path_colvar_trial[1:-1] += path_tangent[1:-1] * lamds[:, None]
      path_rms_trial = self.get_path_weighted_rms(path_colvar=path_colvar_trial)

      # 2. Determine if the trial move can be accepted.
      if (path_rms_trial > rms_thresh).any():
        lamds *= self._regular_grow_dscale
        lamd_to_regulr = True
      else:
        lamd_to_regulr = False

    return lamds
        
  def _cons_lamd_coefs(self, 
                       path_rms_grad: np.ndarray,
                       path_tangent:  np.ndarray,
                       ) -> np.ndarray:
    """Return the LHS of the constraint equation as a matrix of coefficients on the Lagrange 
      multipliers. 
    """
    # no. lambdas = num_replicas-2 (minus 2 as both end points are not updated).
    num_lamd = path_rms_grad.shape[0]-1

    # Path tangent vectors. 
    # path_tan[1:num_lamd+1]: tangents on intermediate replicas.
    # Both are of shape (num_replicas-2, num_replica_dofs).
    coef_tan_i      =  1. * path_tangent[1:num_lamd+1] #  tan( r_{i  }^(n-1) )
    coef_tan_iplus1 = -1. * path_tangent[1:num_lamd+1] # -tan( r_{i+1}^(n-1) )

    # Coefficients on the lambdas.
    # NOTE: Both rms_grad and path_tan on one replica is one dimenstion and thus a sum along axis=1 
    #       on element wise product is implemented. 
    # path_rms_grad,  shape (num_replicas-1, num_replica_dofs); 
    # coef_tan_plus1, shape (num_replicas-2, num_replica_dofs).
    coef_lamd_iplus1 = np.sum(path_rms_grad[0:-1] * coef_tan_iplus1, axis=1)
    coef_lamd_i      = np.sum(path_rms_grad[1:  ] * coef_tan_i     , axis=1)

    # Build the coefficients matrix of shape (num_replicas-1, num_replicas-2).
    lamd_coef_mat = np.zeros((num_lamd+1, num_lamd))
    lamd_coef_mat.ravel()[     0::num_lamd+1] = coef_lamd_iplus1  # on lamd_{i+1}.
    lamd_coef_mat.ravel()[num_lamd::num_lamd+1] = coef_lamd_i     # on lamd_{i}  .

    return lamd_coef_mat


