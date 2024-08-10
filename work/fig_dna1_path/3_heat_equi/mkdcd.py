# Merge PDBs into DCD.
# Zilin Song, 20231207
# 

from openmm.app import PDBFile, DCDFile, CharmmPsfFile

if __name__ == '__main__':
  with open(f'./cors_heat/dna_heat.dcd', 'wb') as f:
    psf = CharmmPsfFile('../2_mep_box/cors/dna.psf')
    dcd = DCDFile(f, topology=psf.topology, dt=.001)

    for irep in range(80):
      with open(f'./cors_heat/r{irep}.pdb', 'r') as f:
        dcd.writeModel(PDBFile(f).positions)
    
  with open(f'./cors_equi/dna_equi.dcd', 'wb') as f:
    dcd = DCDFile(f, topology=psf.topology, dt=.001)

    for irep in range(80):
      with open(f'./cors_equi/r{irep}.pdb', 'r') as f:
        dcd.writeModel(PDBFile(f).positions)