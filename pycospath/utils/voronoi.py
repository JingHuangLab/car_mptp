#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""Voronoi Tessellation related methods."""

from typing import Callable

from functools import partial

import numpy as np

# Voronoi Tessellation functions. ==================================================================
# Helpers. -----------------------------------------------------------------------------------------

def getfn_get_voronoi_box_id(
                        method_voronoi_boundary:     str, 
                        fn_get_weighted_aligned:     Callable[[np.ndarray, np.ndarray], np.ndarray], 
                        fn_get_rowwise_weighted_rms: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                        **kwargs) -> Callable[[np.ndarray, np.ndarray], int]:
  """Wraps the _voronoi_box_id_hplane as a function that takes two variables, `voronoi_anchors` and
    `replica_coordinates`, and returns the Voronoi box ID in which the `replica_coordinates` locate. 
    
    The function `fn_get_weighted_aligned` is used to align each rows of `voronoi_anchors` onto the
    `replica_coordinates`; 
    The function `fn_get_rowwise_weighted_rms` is used to compute the distances between two 
    coordinates; 
    The function `fn_voronoi_box_id` is used to compute the Voronoi box ID. `fn_voronoi_box_id` may 
    take additional kwargs. 
  
    Args:
      method_voronoi_boundary (str):
        The name of the function for computing the Voronoi box ID. 
      fn_get_weighted_aligned (Callable[[np.ndarray, np.ndarray], np.ndarray]): 
        The get_weighted_aligned callable function for computing the optimal alignment between two 
        sets of Cartesian coordinates. Takes two kwargs, `array_to_refer` and `array_to_align`. 
        Returns the aligned coordinates of `array_to_align`. 
      fn_get_weighted_rowwise_rms (Callable[[np.ndarray, np.ndarray], np.ndarray]):
        The get_weighted_rowwise_rms callable function for computing the distances between two sets 
        of Cartesian coordinates. Takes two kwargs, `array0` and `array1`. Returns the rowwise 
        weighted RMS distances. 
      Additional kwargs are passed as parameters to `fn_voronoi_box_id`. 

    Returns:
      fn_compute_voronoi_box_id (Callable[[np.ndarray, np.ndarray], int]):
        The compute_voronoi_box_id callable function. Takes np.ndarrays `voronoi_anchors` and 
        `replica_coordinates`. Returns the Voronoi box ID in which `replica_coordinates` locates.
  """
  # Sanity checks.
  assert isinstance(method_voronoi_boundary, str),           "Illegal method_voronoi_boundary type."
  assert method_voronoi_boundary in METHODS_VORONOI_BOUNDARY,"Illegal method_voronoi_boundary spec."

  assert callable(fn_get_weighted_aligned),     "Illegal non-callable fn_get_weighted_aligned."
  assert callable(fn_get_rowwise_weighted_rms), "Illegal non-callable fn_get_rowwise_weighted_rms."

  return partial(METHODS_VORONOI_BOUNDARY[method_voronoi_boundary], 
                 fn_get_weighted_aligned=fn_get_weighted_aligned, 
                 fn_get_rowwise_weighted_rms=fn_get_rowwise_weighted_rms, 
                 **kwargs)


# Implementations. ---------------------------------------------------------------------------------

def _voronoi_box_id_hplane(
                        voronoi_anchors:     np.ndarray, 
                        replica_coordinates: np.ndarray, 
                        fn_get_weighted_aligned:     Callable[[np.ndarray, np.ndarray], np.ndarray], 
                        fn_get_rowwise_weighted_rms: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                        ) -> int:
  """Get the Voronoi box ID under the Voronoi boundary conditions."""
  # Aligns the voronoi_anchors to the replica_coordinates.
  aligned_voronoi_anchors = np.zeros(voronoi_anchors.shape)
  for i in range(voronoi_anchors.shape[0]):
    aligned_voronoi_anchors[i, :] = fn_get_weighted_aligned(array_to_refer=replica_coordinates, 
                                                            array_to_align=voronoi_anchors[i, :], )
    
  # Computes the distances from voronoi_anchors to replica_coordinates.
  rowwise_distances = fn_get_rowwise_weighted_rms(array0=replica_coordinates[np.newaxis, :], 
                                                  array1=aligned_voronoi_anchors, )
  return int( np.argmin(rowwise_distances) )


METHODS_VORONOI_BOUNDARY = {'hplane': _voronoi_box_id_hplane, }
