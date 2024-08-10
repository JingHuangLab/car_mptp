#
# pyCoSPath: A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""Geometric calculations and manipulations."""

from typing import Callable

from functools import wraps, partial

import numpy as np

# Rowwise functions - unweighted. ==================================================================
# Implementations. ---------------------------------------------------------------------------------

def get_rowwise_projection(array_to_refer:   np.ndarray, 
                           array_to_project: np.ndarray, 
                           ) -> np.ndarray:
  """Perform row-wise vector projections of array_to_project onto array_to_refer. 

    Args:
      array_to_refer (np.ndarray):
        The reference array to be projected on.
      array_to_project (np.ndarray):
        The array to be projected.
    
    Returns:
      rowwise_projection (np.ndarray):
        The projected array. 
  """
  return array_to_refer * ( np.sum(array_to_refer*array_to_project, axis=1) /
                            np.sum(array_to_refer**2              , axis=1)   )[:, np.newaxis]


def get_rowwise_cosine(array0: np.ndarray, 
                       array1: np.ndarray, 
                       ) -> np.ndarray:
  """Perform row-wise cosine calculations between the arrays array0 and array1.

    Args:
      array0 (np.ndarray):
        The 2D array.
      array1 (np.ndarray):
        The other 2D array.
    
    Returns:
      rowwise_cosine (np.ndarray):
        The cosine array. 
  """
  return np.sum(array0*array1, axis=1) / ( np.linalg.norm(array0, axis=1) * 
                                           np.linalg.norm(array1, axis=1)   )


# Rowwise functions - weighted. ====================================================================
# Helpers. -----------------------------------------------------------------------------------------

def getfn_get_rowwise_weighted_rms(weight_per_dof: np.ndarray, 
                                   num_dofs_per_atom: int, 
                                   ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
  """Get the function for computing the weighted row-wise RMS calculation between `array0` and 
    `array1`.

    Args:
      weight_per_dof (np.ndarray):
        The per-DOF weighting factors, shape (num_dofs, ).
      num_dofs_per_atom (int):
        The number of DOFs on each particle in the system.
    
    Returns:
      fn_get_rowwise_weighted_rms (Callable[[np.ndarray, np.ndarray], np.ndarray]):
        The get_rowwise_weighted_rms callable function. Takes `array0` and `array1`, returns the 
        rowwise weighted RMS between `array0` and `array1`.
  """
  # Sanity checks.
  assert isinstance(weight_per_dof, np.ndarray), "Illegal weight_per_dof type."
  assert (weight_per_dof >= 0).all(),            "Illegal weight_per_dof spec."

  assert isinstance(num_dofs_per_atom, int), "Illegal num_dofs_per_atom type."
  assert num_dofs_per_atom > 0,              "Illegal num_dofs_per_atom spec."

  return partial(_rowwise_weighted_rms, 
                 weight_per_dof=weight_per_dof, 
                 num_dofs_per_atom=num_dofs_per_atom, )


# Implementations. ---------------------------------------------------------------------------------

def _rowwise_weighted_rms(array0: np.ndarray, 
                          array1: np.ndarray, 
                          weight_per_dof: np.ndarray, 
                          num_dofs_per_atom: int,
                          ) -> np.ndarray:
  """Perform weighted row-wise RMS calculation between the arrays array0 and array1.

    Args:
      array0 (np.ndarray):
        The 2D array, shape (n, m).
      array1 (np.ndarray):
        The other 2D array, shape (n, m).
      weight_per_dof (np.ndarray):
        The per-DOF weighting factors, shape (m, ).
      num_dofs_per_atom (int):
        The number of DOFs on each particle in the system.
    
    Returns:
      weighted_rowwise_rms (np.ndarray):
        Weighted rowwise RMS between the two arrays.
  """
  inv_sum_weight = num_dofs_per_atom / np.sum(weight_per_dof)     # (1, )
  return np.sqrt(np.sum((array0-array1)**2 * weight_per_dof[np.newaxis,:], axis=1) * inv_sum_weight)


# Weighted alignment functions. ====================================================================
# Helpers. -----------------------------------------------------------------------------------------

def getfn_get_weighted_aligned(method_alignment: str, 
                               weight_per_dof:   np.ndarray, 
                               ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
  """Wraps the `_aligner` as a function that applies `_alignment_solver` on `array_to_align` w.r.t. 
    the `array_to_refer`, returns only the aligned array_to_align.
  
    Args:
      method_alignment (str):
        The name of the alignment solver.
      weight_per_dof (np.ndarray):
        The per-DOF weighting factors, shape (num_dofs, ).

    Returns:
      fn_get_weighted_aligned (Callable[[np.ndarray, np.ndarray], np.ndarray]):
        The get_weighted_aligned callable function. Takes `array_to_refer` and `array_to_align`.
        Returns the aligned `array_to_align`.
  """
  def _aligner(array_to_refer: np.ndarray, 
               array_to_align: np.ndarray, 
               weight_per_dof: np.ndarray, 
               fn_weighted_aligned: Callable[[np.ndarray, np.ndarray, np.ndarray],
                                             tuple[np.ndarray, np.ndarray]], 
               ) -> np.ndarray:
    return fn_weighted_aligned(array_to_refer=array_to_refer, 
                               array_to_align=array_to_align, 
                               weight_per_dof=weight_per_dof, )[0]
  
  # Sanity checks.
  assert isinstance(method_alignment, str),     "Illegal method_alignment type."
  assert method_alignment in METHODS_ALIGNMENT, "Illegal method_alignment spec."

  assert isinstance(weight_per_dof, np.ndarray), "Illegal weight_per_dof type."
  assert (weight_per_dof >= 0.).all(),           "Illegal weight_per_dof spec."
  
  return partial(_aligner, 
                 weight_per_dof=weight_per_dof, 
                 fn_weighted_aligned=METHODS_ALIGNMENT[method_alignment], )


def getfn_get_weighted_rotated(method_alignment: str, 
                               weight_per_dof: np.ndarray, 
                               ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
  """Wraps the `rotator` as a function that applies `_alignment_solver` on `array_to_align` w.r.t. 
    the `array_to_refer`, gets the rotational matrix, and applies to the `array_to_rotate`, returns 
    only the rotated `array_to_rotate`.
  
    Args:
      method_alignment (str):
        The name of the alignment solver.
      weight_per_dof (np.ndarray):
        The per-DOF weighting factors, shape (num_dofs, ).

    Returns:
      fn_get_weighted_rotated (Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
        The get_weighted_rotated callable function. Takes `array_to_refer`, `array_to_align`, and 
        `array_to_rotate`. Returns the rotated `array_to_rotate`. 
  """
  def _rotator(array_to_refer:  np.ndarray, 
               array_to_align:  np.ndarray, 
               array_to_rotate: np.ndarray, 
               weight_per_dof:  np.ndarray, 
               fn_weighted_aligned: Callable[[np.ndarray, np.ndarray, np.ndarray],
                                             tuple[np.ndarray, np.ndarray]], 
               ) -> np.ndarray:
    _, rotate_mat = fn_weighted_aligned(array_to_refer=array_to_refer, 
                                        array_to_align=array_to_align, 
                                        weight_per_dof=weight_per_dof, )
    if rotate_mat is None:  # Pseudo solver ('noronotr').
      return array_to_rotate
    
    return (array_to_rotate.reshape(-1, 3) @ rotate_mat).flatten()
  
  # Sanity checks.
  assert isinstance(method_alignment, str),     "Illegal method_alignment type."
  assert method_alignment in METHODS_ALIGNMENT, "Illegal method_alignment spec."

  assert isinstance(weight_per_dof, np.ndarray), "Illegal weight_per_dof type."
  assert (weight_per_dof >= 0.).all(),           "Illegal weight_per_dof spec."

  return partial(_rotator, 
                 weight_per_dof=weight_per_dof, 
                 fn_weighted_aligned=METHODS_ALIGNMENT[method_alignment], )


def _alignment_solver(is_pseudo_solver: bool = False):
  """Decorator for all alignment solvers. This decorator reshapes the input array_to_refer and the
    array_to_align from 1D to 3D before applying the alignment solver and flatten the output array 
    back from 3D to 1D. 

    Args:
      is_pseudo_solver (bool, optional):
      Specify if the alignment solver does not actually do alignments and returns the untouched 1D 
      np.ndarray (to be compatible with 2D systems). 
      Default: False
  """
  def _align_solver_decorator(fn_weighted_aligned: Callable[[np.ndarray, np.ndarray, np.ndarray], 
                                                            tuple[np.ndarray, np.ndarray]], 
                              ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], 
                                            tuple[np.ndarray, np.ndarray]]:
    
    @wraps(fn_weighted_aligned)
    def _wrapper(array_to_refer: np.ndarray, 
                 array_to_align: np.ndarray, 
                 weight_per_dof: np.ndarray, ): 
      # If a pseudo alignment solver is supplied -> return what is fed.
      if is_pseudo_solver == True:
        return fn_weighted_aligned(array_to_refer=array_to_refer, 
                                   array_to_align=array_to_align, 
                                   weight_per_dof=weight_per_dof, )
      
      # Reshaped.
      array_to_refer = array_to_refer.reshape(-1, 3)
      array_to_align = array_to_align.reshape(-1, 3)
      weight_per_dof = weight_per_dof.reshape(-1, 3)

      # Aligned.
      aligned_array, rotate_mat = fn_weighted_aligned(array_to_refer=array_to_refer, 
                                                      array_to_align=array_to_align, 
                                                      weight_per_dof=weight_per_dof, )

      return aligned_array.flatten(), rotate_mat

    return _wrapper

  return _align_solver_decorator


# Implementations. ---------------------------------------------------------------------------------

@_alignment_solver(is_pseudo_solver=True)
def _weighted_aligned_noronotr(array_to_refer: np.ndarray, 
                               array_to_align: np.ndarray, 
                               weight_per_dof: np.ndarray, 
                               ) -> tuple[np.ndarray, np.ndarray]:
  """Implement the no alignment place holder. No alignment will be performed using this function. 
    Its name follows the CHARMM convention 'noro(tation) notr(anslation)'.

    Args: 
      array_to_refer (np.ndarray):
        The reference array to be aligned on, shape (num_dofs, ).
      array_to_align (np.ndarray):
        The mobile array to align onto the reference, shape (num_dofs, ).
      weight_per_dof (np.ndarray):
        The per-DOF weighting factors, shape (num_dofs, ).

    Returns:
      aligned_array (np.ndarray):
        An unchanged copy of array_to_align.
      rotate_mat (None):
        The rotational matrix, `None`.
  """
  return np.copy(array_to_align), None


from scipy.spatial.transform import Rotation
from scipy.linalg import eigh, svd

@_alignment_solver()
def _weighted_aligned_kabsch(array_to_refer: np.ndarray,
                             array_to_align: np.ndarray, 
                             weight_per_dof: np.ndarray, 
                             ) -> tuple[np.ndarray, np.ndarray]:
  """The weighted Kabsch algorithm for optimal alignment.

    Args: 
      array_to_refer (np.ndarray):
        The reference array to be aligned on, shape (num_dofs, ).
      array_to_align (np.ndarray):
        The mobile array to align onto the reference, shape (num_dofs, ).
      weight_per_dof (np.ndarray):
        The per-DOF weighting factors, shape (num_dofs, ).

    Returns:
      aligned_array (np.ndarray):
        The aligned array_to_align.
      rotate_mat (np.ndarray):
        The rotational matrix, shape (3, 3).
  """
  # Compute the weighted covariance matrix:
  # cov(x,y;w)=sum_i^n{w_i * (x_i - m(x; w)) * (y_i - m(y; w))} / sum_i^n{w_i}
  # where: m(x; w) is the weighted mean: m(x; w) = sum_i^n{w_i * x_i} / sum_i^n{w_i}

  ## Total weight per dim.
  weight_per_dim = np.sum(weight_per_dof, axis=0)  # (3, )

  ## Weighted mean per dim: center-of-mass.
  refer_centroid = np.sum(array_to_refer*weight_per_dof, axis=0) / weight_per_dim # (3, )
  align_centroid = np.sum(array_to_align*weight_per_dof, axis=0) / weight_per_dim # (3, )

  ## Minus the weighted mean per dim: Translate to origin
  refer_to_origin = array_to_refer - refer_centroid
  align_to_origin = array_to_align - align_centroid

  ## Covariance matrix * by weight_per_dim (changes only the scaling of S, which is unused).
  cov = np.transpose(align_to_origin * weight_per_dof) @ refer_to_origin

  # SVD decompose of the weight-per-dim-scaled covariance of reference and origin.
  V, _, W = svd(cov)

  # If determinant of V @ U^{T} is negative, R would produce a mirror fit, to fix this, the sign of 
  # the last row of R is inverted by multiplying -1 on V.
  V[:, -1] *= np.sign(np.linalg.det(V @ W))

  # Rotational matrix.
  rotate_mat = V @ W

  # Aligned array:
  aligned_array = align_to_origin @ rotate_mat + refer_centroid

  return aligned_array, rotate_mat


@_alignment_solver()
def _aligned_kearsley(array_to_refer: np.ndarray, 
                      array_to_align: np.ndarray, 
                      weight_per_dof: np.ndarray, 
                      ) -> tuple[np.ndarray, np.ndarray]:
  """The weighted Kearsley algorithm for optimal alignment.
    https://github.com/martxelo/kearsley-algorithm/blob/main/kearsley/kearsley.py

    Args: 
      array_to_refer (np.ndarray):
        The reference array to be aligned on, shape (num_dofs, ).
      array_to_align (np.ndarray):
        The mobile array to align onto the reference, shape (num_dofs, ).
      weight_per_dof (np.ndarray):
        The per-DOF weighting factors, shape (num_dofs, ).

    Returns:
      aligned_array (np.ndarray):
        The aligned array_to_align.
      rotate_mat (np.ndarray):
        The rotational matrix, shape (3, 3).
  """
  centroid_u = array_to_refer.mean(axis=0)
  centroid_v = array_to_align.mean(axis=0)
  
  # center both sets of points
  x, y = array_to_refer - centroid_u, array_to_align - centroid_v
  
  # diff and sum quantities
  d, s = x - y, x + y
  
  # extract columns to simplify notation
  d0, d1, d2 = d[:,0], d[:,1], d[:,2]
  s0, s1, s2 = s[:,0], s[:,1], s[:,2]
  
  # fill kearsley matrix
  K = np.zeros((4, 4))
  K[0,0] = np.dot(d0, d0) + np.dot(d1, d1) + np.dot(d2, d2)
  K[1,0] = np.dot(s1, d2) - np.dot(d1, s2)
  K[2,0] = np.dot(d0, s2) - np.dot(s0, d2)
  K[3,0] = np.dot(s0, d1) - np.dot(d0, s1)
  K[1,1] = np.dot(s1, s1) + np.dot(s2, s2) + np.dot(d0, d0)
  K[2,1] = np.dot(d0, d1) - np.dot(s0, s1)
  K[3,1] = np.dot(d0, d2) - np.dot(s0, s2)
  K[2,2] = np.dot(s0, s0) + np.dot(s2, s2) + np.dot(d1, d1)
  K[3,2] = np.dot(d1, d2) - np.dot(s1, s2)
  K[3,3] = np.dot(s0, s0) + np.dot(s1, s1) + np.dot(d2, d2)
  
  _, eig_vecs = eigh(K)

  q = eig_vecs[:,0]
  q = np.roll(q, shift=3)

  rot = Rotation.from_quat(q).inv()
  trans = centroid_v - rot.inv().apply(centroid_u)

  aligned_array = rot.apply(array_to_align - trans)

  return aligned_array, rot.as_matrix()

# All available alignment functions.
METHODS_ALIGNMENT = {'noronotr': _weighted_aligned_noronotr, 
                     'kabsch':   _weighted_aligned_kabsch, # To fix: use kearsley only.
                     'kearsley': _aligned_kearsley, }
