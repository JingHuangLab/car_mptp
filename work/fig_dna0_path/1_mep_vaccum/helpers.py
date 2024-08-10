# MEP in vaccume for DNA wc -> hg transition.
# Zilin Song, 20231204
# 

import copy
from openmm import unit, VerletIntegrator
from openmm.app import (CharmmParameterSet, 
                        CharmmCrdFile, 
                        CharmmPsfFile,
                        DCDFile, 
                        PDBFile, 
                        Topology, 
                        
                        NoCutoff, )

for na_resname in ['ADE', 'GUA', 'CYT', 'URA', 'THY']:
  PDBFile._standardResidues.append(na_resname) # So that writeModel does not output HETATM.

def read_top(file_dir: str) -> Topology:
  """Load the dna topology."""
  return CharmmPsfFile(file_dir).topology

def read_pdb(file_dir: str) -> unit.Quantity:
  """Load the PDBFiles positions."""
  return PDBFile(file=file_dir).positions

def read_cor(file_dir: str) -> unit.Quantity:
  """Load the CharmmCrdFile positions."""
  return CharmmCrdFile(file_dir).positions


# Convert back to CHARMM res/atom-names before writing PDBs or DCDz (needed for CHARMM solvation & neutralization).
pdb_to_charmm_resnames = { 'A': 'ADE',  'G': 'GUA',  'C': 'CYT',  'U': 'URA', 
                          'DA': 'ADE', 'DG': 'GUA', 'DC': 'CYT', 'DT': 'THY', }

pdb_to_charmm_atomnames = {'HO5\'': 'H5T', 'HO3\'': 'H3T', 'C7': 'C5M', 'OP1': 'O1P', 'OP2': 'O2P', }


def write_pdb(to_file: str, topology: Topology, coordinate: unit.Quantity) -> None:
  """Write coordinate as PDB."""
  top = copy.copy(topology)

  for residue in top.residues():
    if residue.name in pdb_to_charmm_resnames.keys():
      residue.name = pdb_to_charmm_resnames[residue.name]

  for atom in top.atoms():
    if atom.name in pdb_to_charmm_atomnames.keys():
      atom.name = pdb_to_charmm_atomnames[atom.name]

  with open(f'{to_file}', 'w') as f:
    PDBFile.writeModel(file=f, 
                       topology=top, 
                       positions=coordinate, )

def write_dcd(to_file: str, topology: Topology, coordinates: list[unit.Quantity]) -> None:
  """Write the coordinates as DCD."""
  top = copy.copy(topology)

  for residue in top.residues():
    if residue.name in pdb_to_charmm_resnames.keys():
      residue.name = pdb_to_charmm_resnames[residue.name]

  for atom in top.atoms():
    if atom.name in pdb_to_charmm_atomnames.keys():
      atom.name = pdb_to_charmm_atomnames[atom.name]

  with open(f'{to_file}', 'wb') as f:
    dcd = DCDFile(file=f, topology=top, dt=.001)
    for coordinate in coordinates:
      dcd.writeModel(positions=coordinate)

from openmm import System as OpenMMSystem, VerletIntegrator as OpenMMIntegrator

def get_openmm_system_and_integrator() -> tuple[OpenMMSystem, OpenMMIntegrator]:
  """Obtain OpenMMSystem and Integrators for MEP."""
  ffs = CharmmParameterSet('../0_setups/toppar/top_all36_na.rtf', 
                           '../0_setups/toppar/par_all36_na.prm', 
                           '../0_setups/toppar/toppar_water_ions.str', )
  psf = CharmmPsfFile('../0_setups/guess/dna.psf')
  openmm_system: OpenMMSystem = psf.createSystem(ffs, nonbondedCutoff=NoCutoff)
  openmm_integrator = VerletIntegrator(.001)
  print(f'Loading system/integrator', flush=True)
  return openmm_system, openmm_integrator