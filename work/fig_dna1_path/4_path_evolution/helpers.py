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


def get_openmm_system() -> OpenMMSystem:
  """Get the OpenMM System."""
  ffs = CharmmParameterSet('./ms_scalebxd/fig_dna0_path/0_setups/toppar/par_all36_na.prm',
                           './ms_scalebxd/fig_dna0_path/0_setups/toppar/top_all36_na.rtf',
                           './ms_scalebxd/fig_dna0_path/0_setups/toppar/toppar_water_ions.str', )
  psf = CharmmPsfFile('./ms_scalebxd/fig_dna1_path/2_mep_box/cors/dna.psf')
  psf.setBox(62.6*unit.angstrom, 62.6*unit.angstrom, 62.6*unit.angstrom, )
  
  openmm_system: OpenMMSystem = psf.createSystem(ffs, 
                                                 nonbondedMethod=PME, 
                                                 nonbondedCutoff=12*unit.angstrom, 
                                                 switchDistance =10*unit.angstrom, 
                                                 constraints    =HBonds, )  

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
  return CharmmPsfFile('./ms_scalebxd/fig_dna1_path/2_mep_box/cors/dna.psf').topology


def get_openmm_system_and_integrator() -> tuple[OpenMMSystem, OpenMMLangevinIntegrator]:
  """Get the OpenMMSystem and the integrator."""
  openmm_system = get_openmm_system()
  openmm_integrator = OpenMMLangevinIntegrator(300.   *unit.kelvin, 
                                                 2.   *unit.picosecond**-1, 
                                                  .001*unit.picosecond, )
  return openmm_system, openmm_integrator