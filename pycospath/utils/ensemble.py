#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""Ensemble statistics methods."""

from typing import Callable

from functools import partial

import numpy as np

# Ensemble averaging functions. ====================================================================
# Helpers. -----------------------------------------------------------------------------------------

def getfn_get_ensemble_average(
                            method_ensemble_average: str, 
                            fn_get_weighted_aligned: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                            ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
  """Wraps the `_get_ensemble_average` as a function that takes `array_to_refer` and 
    `array_ensemble`, and returns the ensemble average of `array_ensemble`. 

    The function `fn_ensemble_average` is used to compute the ensemble average;
    The function `fn_get_weighted_aligned` is used to align each rows of `array_ensemble` onto the
    `array_to_refer`.

    Args:
      fn_ensemble_average (Callable[[np.ndarray], np.ndarray]):
        The _ensemble_average callable function for computing the averaged array. Takes one kwarg, 
        `array_ensemble`. Returns the averaged array. 
      fn_get_weighted_aligned (Callable[[np.ndarray, np.ndarray], np.ndarray]): 
        The get_weighted_aligned callable function for computing the optimal alignment between two 
        sets of Cartesian coordinates. Takes two kwargs, `array_to_refer` and `array_to_align`. 
        Returns the aligned coordinates of `array_to_align`. 
     
    Return:
      fn_get_ensemble_average (Callable[[np.ndarray, np.ndarray], np.ndarray]):
        The compute_ensemble_average callable function. Takes `array_to_refer` and `array_ensemble`.
        Returns the ensemble average of `array_ensemble`. 
  """
  def _get_ensemble_average(array_to_refer: np.ndarray, 
                            array_ensemble: np.ndarray, 
                            fn_ensemble_average:     Callable[[np.ndarray,           ], np.ndarray], 
                            fn_get_weighted_aligned: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                            ) -> np.ndarray:
    # Aligns the array_ensemble to the anchor.
    array_ensemble_aligned = np.zeros(array_ensemble.shape)
    for i_sample in range(array_ensemble.shape[0]):
      array_ensemble_aligned[i_sample, :] = fn_get_weighted_aligned(
                                                      array_to_refer=array_to_refer, 
                                                      array_to_align=array_ensemble[i_sample, :], )
    return fn_ensemble_average(array_ensemble=array_ensemble_aligned)
  
  # Sanity checks.
  assert isinstance(method_ensemble_average, str),           "Illegal method_ensemble_average type."
  assert method_ensemble_average in METHODS_ENSEMBLE_AVERAGE,"Illegal method_ensemble_average spec."
  assert callable(fn_get_weighted_aligned), "Illegal non-callable fn_get_weighted_aligned."

  return partial(_get_ensemble_average, 
                 fn_ensemble_average    =METHODS_ENSEMBLE_AVERAGE[method_ensemble_average], 
                 fn_get_weighted_aligned=fn_get_weighted_aligned, )


# Implementations. ---------------------------------------------------------------------------------

def _ensemble_average_arithmetic(array_ensemble: np.ndarray):
  """Computes the arithmetic average of the array_ensemble."""
  return np.mean(array_ensemble, axis=0)


METHODS_ENSEMBLE_AVERAGE = {'arithmetic': _ensemble_average_arithmetic, }
