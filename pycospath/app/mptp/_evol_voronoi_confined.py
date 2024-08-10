
#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""The PathEvolver for ReaxpathVoronoiConfinedSampler."""

from typing import Callable

import numpy as np

from pycospath.utils.ensemble import getfn_get_ensemble_average

from pycospath.utils.voronoi import getfn_get_voronoi_box_id

from pycospath.app.mptp import PathEvolver

from pycospath.samp import VoronoiConfinedSampler

class VoronoiConfinedPathEvolver(PathEvolver):
  """The PathEvolver for ReaxpathVoronoiConfinedSampler."""

  def __init__(self, 
               method_ensemble_average: str = 'arithmetic', 
               method_voronoi_boundary: dict = {'voronoi_type': 'hplane', }, 
               config_path_fix_mode: str = 'both', 
               ) -> None:
    """Create a PathEvolver for ReaxpathVoronoiConfinedSampler.
    
      Args:
        method_ensemble_average (str, optional):
          Specifies the method for computing the average Path from each of the Voronoi cells.
          Default: 'arithmetic'. 
        method_voronoi_boundary (dict, optional):
          Specifies the Voronoi boundary condition applied during the Path sampling in each of the 
          Voronoi cells.
          Default: {'type': 'hplane', }.
        config_path_fix_mode (str, optional):
          Specifies if the end points on the Path colvar are to be fixed during the evolution.
          Default: 'both'.
            'both': Coodinates of all replicas except the two end-point replicas (the first and the 
                    last) will be evolved;
            'head': Coordinates of all replicas except the first replica will be evolved;
            'tail': Coordinates of all replicas except the  last replica will be evolved; 
            'none': Coordinates of all replicas will be evolved. 
    """
    PathEvolver.__init__(self, 
                         config_path_fix_mode=config_path_fix_mode, )
    
    # Sanity checks.
    assert isinstance(method_ensemble_average, str), "Illegal method_ensemble_average type."
    self._method_ensemble_average = method_ensemble_average

    assert isinstance(method_voronoi_boundary, dict), "Illegal method_voronoi_boundary type."
    method_voronoi_boundary = method_voronoi_boundary.copy()
    self._method_voronoi_boundary = method_voronoi_boundary.pop('voronoi_type') # key removed.
    self._kwargs_voronoi_boundary = method_voronoi_boundary # Additional kwargs for _voronoi_box_id.

  # External interfaces. ---------------------------------------------------------------------------

  def implement(self, 
                fn_get_weighted_aligned:     Callable[[np.ndarray, np.ndarray], np.ndarray], 
                fn_get_rowwise_weighted_rms: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                ) -> None:
    """To realize Evolver communication functions."""
    PathEvolver.implement(self, 
                          fn_get_weighted_aligned=fn_get_weighted_aligned, 
                          fn_get_rowwise_weighted_rms=fn_get_rowwise_weighted_rms, )
    
    self.get_ensemble_average = getfn_get_ensemble_average(
                                              method_ensemble_average=self._method_ensemble_average, 
                                              fn_get_weighted_aligned=self.get_weighted_aligned, )
    
    self.get_voronoi_box_id = getfn_get_voronoi_box_id(
                                          method_voronoi_boundary=self._method_voronoi_boundary, 
                                          fn_get_weighted_aligned =self.get_weighted_aligned, 
                                          fn_get_rowwise_weighted_rms=self.get_rowwise_weighted_rms, 
                                          **self._kwargs_voronoi_boundary, )
  
  # External interface prompts - compute_voronoi_box_id & compute_ensemble_average methods. 

  def get_voronoi_box_id(self, 
                         voronoi_anchors:     np.ndarray, 
                         replica_coordinates: np.ndarray, 
                         ) -> int:
    """Prompt: Compute the Voronoi box ID of the array `replica_coordinates` under the Voronoi 
      tessellation by the array `voronoi_anchors`.
    """
    raise RuntimeError("Prompt method not realized in VoronoiConfinedPathEvolver.implement().")

  def get_ensemble_average(self, 
                           array_to_refer: np.ndarray, 
                           array_ensemble: np.ndarray, 
                           ) -> np.ndarray:
    """Prompt: Compute the ensemble average of array `array_ensemble` after aligned each sample to
      the array `array_to_refer`.
    """
    raise RuntimeError("Prompt method not realized in VoronoiConfinedPathEvolver.implement().")
    
  # VoronoiConfinedPathEvolver runtime per ReaxpathVoronoiConfinedSampler. -------------------------
  # NOTE: for MPI impls.

  def create_sampler(self) -> VoronoiConfinedSampler:
    """Create a ReaxpathVoronoiConfinedSampler."""
    return VoronoiConfinedSampler(fn_get_weighted_aligned    =self.get_weighted_aligned, 
                                  fn_get_rowwise_weighted_rms=self.get_rowwise_weighted_rms, 
                                  fn_get_voronoi_box_id      =self.get_voronoi_box_id, )

  def get_evolved_replica_coordinates(self, sampler: VoronoiConfinedSampler) -> np.ndarray:
    """Returns the evolved Replica region coordinates from the sampled statistics in the Sampler.

      Args:
        sampler (ReaxpathVoronoiConfinedSampler):
          The ReaxpathVoronoiConfinedSampler.

      Returns:
        evolved_replica_coordinates (np.ndarray):
          The evolved Replica region coordinates of the sampled ensemble, shape (num_replica_dofs,).
    """
    array_to_refer = sampler.get_path_colvar()[sampler.get_whoami(), :]
    array_ensemble = sampler.get_replica_trajectory_tape()._data_replica_coordinates
    return self.get_ensemble_average(array_to_refer=array_to_refer, 
                                     array_ensemble=array_ensemble, )


