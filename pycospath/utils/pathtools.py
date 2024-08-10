#
# pyCoSPath: A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""Path helper functions."""

from typing import Callable, Union

from functools import partial

import numpy as np

from scipy.interpolate import CubicSpline

# Path RMS functions. ==============================================================================
# Helpers. -----------------------------------------------------------------------------------------

def getfn_get_path_weighted_rms(weight_per_dof:  np.ndarray, 
                                num_dofs_per_atom: int,
                                ) -> Callable[[np.ndarray, bool], 
                                              Union[np.ndarray, tuple[np.ndarray, np.ndarray]]]: 
  """Get the function for computing the per-DOF weighted/scaled RMS distances between adjacent 
    Replicas in the Path colvar, and optionally also the gradients of the RMS distances to the 
    preceding Replica colvar.

    Args:
      weight_per_dof: np.ndarray
        The shape (num_replica_dofs, ) np.ndarray as the the per-DOF weighting factors.
      num_dofs_per_atom: int
        The number of DOFs on each particle in the system.
    
    Returns:
      fn_get_weighted_path_rms (Callable[[np.ndarray, bool], 
                                         Union[np.ndarray, tuple[np.ndarray, np.ndarray]]]):
        The get_weighted_path_rms callable function.
  """
  # Sanity checks.
  assert isinstance(weight_per_dof, np.ndarray), "Illegal weight_per_dof type."
  assert (weight_per_dof >= 0).all(),            "Illegal weight_per_dof spec."

  assert isinstance(num_dofs_per_atom, int), "Illegal num_dofs_per_atom type."
  assert num_dofs_per_atom > 0,              "Illegal num_dofs_per_atom spec."

  return partial(_path_weighted_rms, 
                 weight_per_dof=weight_per_dof, 
                 num_dofs_per_atom=num_dofs_per_atom, )


# Implementation. ----------------------------------------------------------------------------------

def _path_weighted_rms(path_colvar:     np.ndarray,
                       weight_per_dof:  np.ndarray,
                       num_dofs_per_atom: int, 
                       return_grad:     bool = False,
                       ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
  """Get the per-DOF weighted/scaled RMS distances between adjacent Replicas in the Path colvar, and
    optionally also the gradients of the RMS distances to the preceding Replica colvar.

    Args:
      path_colvar (np.ndarray):
        The Path colvar, shape (num_replicas, num_replica_dofs).
      weight_per_dof (np.ndarray):
        The per-DOF weighting factors, shape (num_replica_dofs, ).
      per_atom_dof (int):
        The number of DOFs on each particle in the system.
      return_grad (bool, optional): 
        If the gradients of RMS distances to the preceding Replica colvar should also be returned. 
        Default: False.
    
    Returns:
      weighted_path_rms (np.ndarray):
        The RMS distances between adjacent Replica colvars, shape (num_replicas-1, ).
      weighted_path_rms_grad (np.ndarray):
        The RMS gradients to the preceding Replica colvar, shape (num_replicas-1, num_replica_dofs). 
        Returns if return_grad == True.
  """
  diff = path_colvar[0:-1, :] - path_colvar[1:  , :]     # shape (num_replicas-1, num_replica_dofs).
  inv_sum_weight = num_dofs_per_atom / np.sum(weight_per_dof)   # shape (1, ).

  # RMS
  rms = np.sqrt(np.sum(diff**2 * weight_per_dof[np.newaxis, :], axis=1) * inv_sum_weight)

  # RMS gradient (if return_grad == True).
  if return_grad == True:
    rms_grad = diff * weight_per_dof[np.newaxis, :] * inv_sum_weight / rms[:, np.newaxis]
    return rms, rms_grad
  
  return rms


# Path tangent functions. ==========================================================================
# Helpers. -----------------------------------------------------------------------------------------

def getfn_get_path_tangent(method_path_tangent: str):
  """Wraps the `get_path_tangent` as a function that computes the Path tangent vector with the 
    specified method.

    Args: 
      method_path_tangent (str):
        The name of the Path tangent method.  
  """
  assert method_path_tangent in METHODS_PATH_TANGENT, "Illegal method_path_tangent spec."
  return METHODS_PATH_TANGENT[method_path_tangent]


# Implementations. ---------------------------------------------------------------------------------

def _path_tangent_basic(path_colvar: np.ndarray) -> np.ndarray:
  """Implement the Path tangent vector at each replica as:
      $tan(r_{i}) = r_{i} - r_{i+1}$;
    with boundary conditions:
      $tan(r_{n}) = r_{n-1} - r_{n}$.

    Args:
      path_colvar (np.ndarray):
        The Path colvar, shape (num_replicas, num_replica_dofs).

    Returns:
      path_tangent_basic (np.ndarray):
        The Path tangent vector, shape (num_replicas, num_replica_dofs).
  """
  path_tangent_basic = np.zeros(path_colvar.shape)
  path_tangent_basic[0:-1, :] = path_colvar[0:-1, :] - path_colvar[1:, :]
  path_tangent_basic[  -1, :] = path_tangent_basic[-2, :] # last = second last
  return path_tangent_basic


def _path_tangent_context(path_colvar: np.ndarray) -> np.ndarray:
  """Implement the Path tangent vector at each replica as:
      $tan(r_{i}) = r_{i-1} - r_{i+1}$;
    with boundary conditions:
      $tan(r_{0}) = r_{0}   - r_{1}$;
      $tan(r_{n}) = r_{n-1} - r_{n}$.

    Args:
      path_colvar (np.ndarray):
        The Path colvar, shape (num_replicas, num_replica_dofs).

    Returns:
      path_tangent_context (np.ndarray):
        The Path tangent vector, shape (num_replicas, num_replica_dofs).
  """
  path_tangent_context = np.zeros(path_colvar.shape)
  path_tangent_context[1:-1, :] = path_colvar[0:-2, :] - path_colvar[2:  , :]
  path_tangent_context[0   , :] = path_colvar[0   , :] - path_colvar[1   , :] # first = second
  path_tangent_context[  -1, :] = path_colvar[  -2, :] - path_colvar[  -1, :]# last = second last
  return path_tangent_context


def _path_tangent_cspline(path_colvar: np.ndarray) -> np.ndarray:
  """Implement the Path tangent vector at each replica as the derivatives to the cubic spline path
    function that is used to fit the Path colvar.

    Args:
      path_colvar (np.ndarray):
        The Path colvar, shape (num_replicas, num_replica_dofs).

    Returns:
      path_tangent_cspline (np.ndarray):
        The Path tangent vector, shape (num_replicas, num_replica_dofs).
  """
  # Compute the normalized Path segment coordinates ($\alpha$s).
  rms = np.zeros((path_colvar.shape[0], ))
  rms[1:] = np.linalg.norm(path_colvar[1:, :] - path_colvar[:-1, :], axis=1)
  path_nodes = np.cumsum(rms) / np.sum(rms)

  # Compute the path tangent for each dof.
  path_tangent_cspline = np.zeros(path_colvar.shape)

  for i_dof in range(path_colvar.shape[1]):
    intpol = CubicSpline(x=path_nodes, y=path_colvar[:, i_dof])
    path_tangent_cspline[:, i_dof] = intpol.__call__(x=path_nodes, nu=1)

  return path_tangent_cspline


# Global Keywords
METHODS_PATH_TANGENT = {'basic'  : _path_tangent_basic, 
                        'context': _path_tangent_context,
                        'cspline': _path_tangent_cspline, }
