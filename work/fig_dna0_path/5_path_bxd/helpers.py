# Transition PathCV optimization on DNA WC HG transition: Helper functions.
# Zilin Song, 20231208
# 

from openmm import (unit, 
                    CustomBondForce, 
                    System as OpenMMSystem, 
                    Platform as OpenMMPlatform, 
                    LangevinIntegrator as OpenMMLangevinIntegrator, )
from openmm.app import (CharmmParameterSet, 
                        CharmmPsfFile, 
                        PME, HBonds, 
                        PDBFile, DCDFile, )
from pycospath.utils.pathtools import getfn_get_path_tangent, getfn_get_path_weighted_rms
from pycospath.utils.geometry import getfn_get_weighted_aligned, getfn_get_rowwise_weighted_rms
from pycospath.utils.voronoi  import getfn_get_voronoi_box_id


def get_openmm_system(constraints=True) -> OpenMMSystem:
  """Get the OpenMM System."""
  ffs = CharmmParameterSet('./ms_scalebxd/fig_dna0_path/0_setups/toppar/par_all36_na.prm',
                           './ms_scalebxd/fig_dna0_path/0_setups/toppar/top_all36_na.rtf',
                           './ms_scalebxd/fig_dna0_path/0_setups/toppar/toppar_water_ions.str', )
  psf = CharmmPsfFile('./ms_scalebxd/fig_dna0_path/2_mep_box/cors/dna.psf')
  psf.setBox(62.6*unit.angstrom, 62.6*unit.angstrom, 62.6*unit.angstrom, )
  
  openmm_system: OpenMMSystem = psf.createSystem(ffs, 
                                                 nonbondedMethod=PME, 
                                                 nonbondedCutoff=12*unit.angstrom, 
                                                 switchDistance =10*unit.angstrom, 
                                                 constraints    =HBonds if constraints else None, )  

  # Enhanced bonds. 10 kcal/mol/A^2 in charmm ff convention.
  bp_restraints = CustomBondForce('k_bps*step(r-r0_bps)*(r-r0_bps)^2')
  bp_restraints.addPerBondParameter('k_bps')
  bp_restraints.addPerBondParameter('r0_bps')
  bp_restraints.addBond( 16, 741, (2100*unit.kilojoule_per_mole*unit.nanometer**-2, .31*unit.nanometer)) #  1-O2 -- 24-N2
  bp_restraints.addBond( 17, 746, (2100*unit.kilojoule_per_mole*unit.nanometer**-2, .31*unit.nanometer)) #  1-N3 -- 24-N1
  bp_restraints.addBond( 19, 749, (2100*unit.kilojoule_per_mole*unit.nanometer**-2, .31*unit.nanometer)) #  1-N4 -- 24-O6
  bp_restraints.addBond(369, 394, (2100*unit.kilojoule_per_mole*unit.nanometer**-2, .31*unit.nanometer)) # 12-O2 -- 13-N2
  bp_restraints.addBond(370, 399, (2100*unit.kilojoule_per_mole*unit.nanometer**-2, .31*unit.nanometer)) # 12-N3 -- 13-N1
  bp_restraints.addBond(372, 402, (2100*unit.kilojoule_per_mole*unit.nanometer**-2, .31*unit.nanometer)) # 12-N4 -- 13-O6
  openmm_system.addForce(bp_restraints)

  return openmm_system


def get_openmm_topology() -> object:
  """Get the OpenMMTopology."""
  return CharmmPsfFile('./ms_scalebxd/fig_dna0_path/2_mep_box/cors/dna.psf').topology


def get_openmm_system_and_integrator(constraints=True) -> tuple[OpenMMSystem, OpenMMLangevinIntegrator]:
  """Get the OpenMMSystem and the integrator."""
  openmm_system = get_openmm_system(constraints=constraints)
  openmm_integrator = OpenMMLangevinIntegrator(300.   *unit.kelvin, 
                                                 2.   *unit.picosecond**-1, 
                                                  .001*unit.picosecond, )
  return openmm_system, openmm_integrator



import numpy as np

from openmm import (CustomCVForce as OpenMMCustomCVForce, 
                    RMSDForce     as OpenMMAlignedRMSDForce, )

from pycospath.comm.openmm import (OpenMMQuantity, 
                                   akma_coordinates_to_openmm, 
                                   akma_force_constant_to_openmm, )

class OpenMMRmsdCVForceX(OpenMMCustomCVForce):
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
    OpenMMCustomCVForce.__init__(self, f"(K_rmsdx)*max(0, CV_rmsdx-CVmax_rmsdx)^2")
    
    # if use_aligned_rmsd == False:
    #  self._rmsd_calculator = OpenMMUnalignedRMSDForce(reference_coordinates, replica_atom_indices)
    # else: 
    self._rmsd_colvar = OpenMMAlignedRMSDForce(reference_coordinates, replica_atom_indices)

    self.addCollectiveVariable('CV_rmsdx', self._rmsd_colvar)
    self.addGlobalParameter('CVmax_rmsdx', self._force_cutoff_distance)
    self.addGlobalParameter('K_rmsdx',     self._force_constant)