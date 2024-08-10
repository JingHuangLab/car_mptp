# Transition PathCV optimization on Di-Alanine isomerization: Helper functions.
# Zilin Song, 20230915
# 

from openmm import unit, System as OpenMMSystem, Platform as OpenMMPlatform, LangevinIntegrator as OpenMMLangevinIntegrator
from openmm.app import CharmmParameterSet, CharmmPsfFile, NoCutoff, HBonds, PDBFile, DCDFile


def get_openmm_system() -> OpenMMSystem:
  """Get the OpenMM System."""
  ffs = CharmmParameterSet('./examples/alad_c36m/toppar/par_all36m_prot.prm', 
                           './examples/alad_c36m/toppar/top_all36_prot.rtf', )
  psf = CharmmPsfFile('./examples/alad_c36m/alad.psf')
  return psf.createSystem(ffs, nonbondedCutoff=NoCutoff, constraints=HBonds)


def get_openmm_topology() -> object:
  """Get the OpenMM Topology."""
  return CharmmPsfFile('./examples/alad_c36m/alad.psf').topology


def get_openmm_system_and_integrator() -> tuple[OpenMMSystem, OpenMMLangevinIntegrator]:
  """Get the OpenMMSystem and the integrator."""
  openmm_system = get_openmm_system()
  openmm_integrator = OpenMMLangevinIntegrator(350.   *unit.kelvin, 
                                                30.   *unit.picosecond**-1, 
                                                  .001*unit.picosecond, )
  return openmm_system, openmm_integrator