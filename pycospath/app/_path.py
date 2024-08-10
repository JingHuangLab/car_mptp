#
# pyCoSPath:  A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""Path-conditioned Chain-of-Replicas Ensemble."""

from typing import Type

import numpy as np

from pycospath.rep import Replica, TwoDReplica

from pycospath.utils.geometry import (getfn_get_weighted_aligned, 
                                      getfn_get_weighted_rotated, 
                                      getfn_get_rowwise_weighted_rms, )

from pycospath.utils.pathtools import (getfn_get_path_weighted_rms, 
                                       getfn_get_path_tangent, )

class Path:
  """The Serial Path-conditioned Chain-of-Replicas Ensemble (the Path)."""

  def __init__(self, 
               context_coordinates_list: list, 
               replica_class:  Type[Replica], 
               replica_kwargs: dict, 
               method_alignment:    str = 'kabsch', 
               method_path_tangent: str = 'context',
               ) -> None:
    """Initializes a Path-conditioned Chain-of-Replicas Ensemble (the Path).

      Args:
        context_coordinates_list (list):
          The Python list of context Coordinates objects in Replica;
        replica_class (Type[Replica]):
          The class (not instances) of the Replica;
        replica_kwargs (dict):
          Additional keywords passed to the constructor of the Replica.
        method_alignment (str, optional):
          The method used for molecular alignment for the Chain-of-Replicas. 
          Default: 'kabsch'.
        method_path_tangent (str, optional):
          The method used for Path tangent vector evaluation for the CoS. This option does not 
          impact the String Method as it uses its reparametrized Path function method for tangent 
          estimation.
          Default: 'context'. 
    """
    # Init. Replica list. 
    assert isinstance(context_coordinates_list, list ), "Illegal context_coordinates_list type."
    
    assert issubclass(replica_class,  Replica), "Illegal replica_class type."
    assert isinstance(replica_kwargs, dict   ), "Illegal replica_kwargs type."

    self._replica_list: list[Replica] = []

    for i_replica in range( len(context_coordinates_list) ):
      
      replica = replica_class(whoami=i_replica, 
                              context_coordinates=context_coordinates_list[i_replica], 
                              **replica_kwargs)
      
      self._replica_list.append(replica)
    
    # Init. Path properties. 
    replica_list = self.get_replica_list()

    reference_replica = replica_list[0]

    self._num_path_dofs          = reference_replica.get_replica_num_dofs()
    self._num_path_replicas      = len(replica_list)
    self._num_path_dofs_per_atom = reference_replica.get_context_num_dofs_per_atom()

    self._path_weight_per_dof = reference_replica.get_replica_weight_per_dof()

    # Init. Path colvar and assert for Replica homology. 
    # NOTE: the following AssertionErrors are not included in the unit-tests.
    self._path_colvar = np.zeros((self._num_path_replicas, self._num_path_dofs))

    for i_replica in range(self._num_path_replicas):
      replica = self._replica_list[i_replica]
      replica_coordinates = replica.obtain_replica_coordinates()
      replica_weight_per_dof  = replica.get_replica_weight_per_dof()
      replica_num_dofs_per_atom = replica.get_context_num_dofs_per_atom()

      assert (replica_weight_per_dof == self._path_weight_per_dof).all(), \
             "Illegal replica_dof_weights inequal to Path reference."
      assert replica_num_dofs_per_atom == self._num_path_dofs_per_atom, \
             "Illegal replica_num_dofs_per_atom inequal to Path reference."
      
      if i_replica != 0:
        assert replica_coordinates.shape == (self._num_path_dofs, ), \
               "Illegal replica_coordinates.shape inequal to Path reference."
        
      self._path_colvar[i_replica, :] = replica_coordinates

    # Realize Path and rowwise weighted RMS functions. 
    self.get_path_weighted_rms = getfn_get_path_weighted_rms(
                                                  weight_per_dof=self._path_weight_per_dof, 
                                                  num_dofs_per_atom=self._num_path_dofs_per_atom, )
    
    self.get_rowwise_weighted_rms = getfn_get_rowwise_weighted_rms(
                                                  weight_per_dof=self._path_weight_per_dof, 
                                                  num_dofs_per_atom=self._num_path_dofs_per_atom, )
    
    # Realize Path alignment functions. 
    if issubclass(replica_class, TwoDReplica): # No alignment to TwoDReplicas.
      method_alignment = 'noronotr'

    self.get_weighted_aligned = getfn_get_weighted_aligned(
                                                        method_alignment=method_alignment, 
                                                        weight_per_dof=self._path_weight_per_dof, )
    self.get_weighted_rotated = getfn_get_weighted_rotated(
                                                        method_alignment=method_alignment, 
                                                        weight_per_dof=self._path_weight_per_dof, )

    # Realize Path tangent functions. 
    self.get_path_tangent = getfn_get_path_tangent(method_path_tangent=method_path_tangent)

    # Initial align Path colvars. 
    self.align_path_colvar()

  # Runtime objects. -------------------------------------------------------------------------------

  def get_replica_list(self) -> list[Replica]:
    """Get the list of Replicas on this Path."""
    return self._replica_list
  
  # Path constructor prompts. ----------------------------------------------------------------------
  
  def get_path_tangent(self, 
                       path_colvar: np.ndarray,
                       ) -> np.ndarray:
    """Get the Path tangent vector."""
    raise RuntimeError('Prompt method not realized in Path.__init__().')

  def get_path_weighted_rms(self, 
                            path_colvar: np.ndarray, 
                            return_grad: bool = False, 
                            ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Get the Path rms vector."""
    raise RuntimeError('Prompt method not realized in Path.__init__().')
  
  def get_weighted_aligned(self,
                           array_to_refer: np.ndarray, 
                           array_to_align: np.ndarray, 
                           ) -> np.ndarray:
    """Align array_to_align onto array_to_refer, return the aligned array_to_align."""
    raise RuntimeError('Prompt method not realized in Path.__init__().')
  
  def get_weighted_rotated(self,
                           array_to_refer: np.ndarray, 
                           array_to_align: np.ndarray, 
                           array_to_rotate: np.ndarray, 
                           ) -> np.ndarray:
    """Align array_to_align onto the array_to_refer, get the rotational matrix; Apply the rotational
      matrix to the array_to_rotate, return the rotated array_to_rotate.
    """
    raise RuntimeError('Prompt method not realized in Path.__init__().')
  
  def get_rowwise_weighted_rms(self, 
                               array0=np.ndarray, 
                               array1=np.ndarray, 
                               ) -> np.ndarray:
    """Get the weighted row-wise RMS distances between the arrays array0 and array1."""
    raise RuntimeError('Prompt method not realized in Path.__init__().')
  
  # Path properties. -------------------------------------------------------------------------------
  
  def get_num_path_dofs(self) -> int:
    """Get the number of degrees of freedom in the rows of Path colvar."""
    return self._num_path_dofs
  
  def get_num_path_replicas(self) -> int:
    """Get the number of Replicas on this Path."""
    return self._num_path_replicas
  
  def get_num_path_dofs_per_atom(self) -> int:
    """Get the number of DOFs on each atom in the Path Replicas."""
    return self._num_path_dofs_per_atom
  
  def get_path_weight_per_dof(self) -> np.ndarray:
    """Get a copy of the vector that holds the per-DOF RMS weighting factors in the Path Replica."""
    return np.copy(self._path_weight_per_dof)
  
  # Path colvars. ----------------------------------------------------------------------------------
  
  def align_path_colvar(self):
    """Applies in place alignment to the Path colvar."""
    for i in range(1, self.get_num_path_replicas()):
      self._path_colvar[i, :] = self.get_weighted_aligned(array_to_refer=self._path_colvar[i-1,:],  
                                                          array_to_align=self._path_colvar[i  ,:], )

  def get_path_colvar(self, 
                      retrieve_from_replicas: bool = False, 
                      ) -> np.ndarray:
    """Get a copy of the Path colvar. 

      Args:
        retrieve_from_replica (np.ndarray, optional):
          If the Path colvar should also be rebuilt from the Replicas. If True, the instantaneous 
          Replica region coordinates are retrieved from all Replicas and are aligned to override the
          present Path colvar. 
          Default: False.
      
      Returns:
        path_colvar (np.ndarray):
          A copy of the Path colvar, shape (num_replicas, num_replica_dofs).
    """
    # If rebuild and override Path colvar from the instantaneous Replica region coordinates.
    if retrieve_from_replicas == True:
    
      for i_replica in range(self.get_num_path_replicas()):
        self._path_colvar[i_replica, :] = self._replica_list[i_replica].obtain_replica_coordinates()

      self.align_path_colvar()

    return np.copy(self._path_colvar)
  
  def set_path_colvar(self, 
                      path_colvar: np.ndarray, 
                      update_to_replicas: bool = False, 
                      ) -> None:
    """Set the Path colvar and update the Replica coordinates of each Replica on this Path.
    
      Args: 
        path_colvar (np.ndarray):
          The Path colvar, shape (num_replicas, num_replica_dofs).
        update_to_replicas (bool, optional):
          If the Path colvar should also be updated to the Replicas. If True, the instantaneous 
          Replica region coordinates are updated for all Replicas.
          Default: False.
    """
    # Also update Replica coordinates for each Replica.
    if update_to_replicas == True:

      for i in range(self.get_num_path_replicas()):
        replica: Replica = self.get_replica_list()[i]

        # Align each row of Path colvar to the instantaneous Replica region coordinates.
        replica_coordinates_ref = replica.obtain_replica_coordinates()
        replica_coordinates_new = self.get_weighted_aligned(array_to_refer=replica_coordinates_ref, 
                                                            array_to_align=path_colvar[i, :], )
        
        # And update to Replica.
        replica.update_replica_coordinates(replica_coordinates=replica_coordinates_new)
    
    # Update the Path colvar.
    self._path_colvar = np.copy(path_colvar)
    self.align_path_colvar()


