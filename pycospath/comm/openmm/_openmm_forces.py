#
# pyCoSPath: A Python Library for Chain-of-States Calculations. 
#
# Authors: Zilin Song
# Contrinutors:
# 

"""OpenMM Custom Forces."""

import numpy as np

from openmm import (CustomCVForce as OpenMMCustomCVForce, 
                    RMSDForce     as OpenMMAlignedRMSDForce, )

from pycospath.comm.openmm import (OpenMMQuantity, 
                                   akma_coordinates_to_openmm, 
                                   akma_force_constant_to_openmm, )

class OpenMMRmsdCVForce(OpenMMCustomCVForce):
  """The RMSD biased CV force for OpenMM."""

  _zero_force_constant = akma_force_constant_to_openmm(akma_force_constant=0.)
  _zero_force_cutoff_d = akma_coordinates_to_openmm   (akma_coordinates   =0.)

  def __init__(self, 
               force_constant:        float | OpenMMQuantity, 
               force_cutoff_distance: float | OpenMMQuantity, 
               reference_coordinates: OpenMMQuantity, 
               replica_atom_indices:  np.ndarray | list[int], 
               use_aligned_rmsd:      bool = True, 
               ) -> None:
    """Create a RMSD bias CustomCVForce for OpenMM.
    
      Args:
        force_constant (float | OpenMMQuantity]):
          The force constant of the RMSD bias force, unit: kcal/mol/Angstrom**2. 
          Allowed values are non-negative.
        force_cutoff_distance (float | OpenMMQuantity):
          The distance beyond which the RMSD bias force is active, unit: Angstrom.
          Allowed values are non-negative.
        reference_coordinates (OpenMMQuantity):
          The reference Context coordinates to which the RMSD from the instantaneous coordinates are 
          calculated, unit: Angstrom.
        replica_atom_indices: (np.ndarray | list[int]):
          The indices of the Replica region atoms that is used to compute the RMSD distances. 
        use_aligned_rmsd (bool, optional):
          If optimal alignment should be applied for computing the RMSD. 
          Must be True. 
    """
    # floats to OpenMMQuantity, np.ndarray to list[int]
    if isinstance(force_constant, float):
      force_constant = akma_force_constant_to_openmm(akma_force_constant=force_constant)
    if isinstance(force_cutoff_distance, float):
      force_cutoff_distance = akma_coordinates_to_openmm(akma_coordinates=force_cutoff_distance)
    if isinstance(replica_atom_indices, np.ndarray):
      replica_atom_indices = list(replica_atom_indices)
      
    # Sanity checks.
    assert isinstance(force_constant, OpenMMQuantity),  "Illegal force_constant type."
    assert force_constant >= self._zero_force_constant, "Illegal force_constant spec."

    self._force_constant = force_constant

    assert isinstance(force_cutoff_distance, OpenMMQuantity),  "Illegal force_cutoff_distance type."
    assert force_cutoff_distance >= self._zero_force_cutoff_d, "Illegal force_cutoff_distance spec."

    self._force_cutoff_distance = force_cutoff_distance

    assert isinstance(reference_coordinates, OpenMMQuantity), "Illegal reference_coordinates type."

    assert isinstance(replica_atom_indices, list), "Illegal replica_atom_indices type."

    # TODO: RMSD without alignment requires an OpenMM plugin.
    if use_aligned_rmsd == False:
      raise NotImplementedError("OpenMMRmsdCVForce must be defined using optimal-aligned RMSDs.")
    
    # Actual init. ---------------------------------------------------------------------------------
    # Note that the energy expression follows the CHARMM convention (without the 1/2 prefactor).
    OpenMMCustomCVForce.__init__(self, f"(K_rmsd)*max(0, CV_rmsd-CVmax_rmsd)^2")
    
    # if use_aligned_rmsd == False:
    #  self._rmsd_calculator = OpenMMUnalignedRMSDForce(reference_coordinates, replica_atom_indices)
    # else: 
    self._rmsd_colvar = OpenMMAlignedRMSDForce(reference_coordinates, replica_atom_indices)

    self.addCollectiveVariable('CV_rmsd', self._rmsd_colvar)
    self.addGlobalParameter('CVmax_rmsd', self._force_cutoff_distance)
    self.addGlobalParameter('K_rmsd',     self._force_constant)


