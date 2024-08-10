# MEP in vaccume for DNA wc -> hg transition.
# Zilin Song, 20231204
# 

import sys, numpy as np
sys.dont_write_bytecode=True

from helpers import read_top, read_pdb, read_cor, write_dcd, write_pdb, unit

if __name__ == '__main__':
  top = read_top(file_dir='../0_setups/guess/dna.psf')

  pos0  = np.asarray(read_cor(file_dir='../0_setups/guess/r0.cor' ).value_in_unit(unit=unit.angstrom))
  pos25 = np.asarray(read_cor(file_dir='../0_setups/guess/r25.cor').value_in_unit(unit=unit.angstrom))
  
  for _ in range(16):
    write_pdb(to_file=f'./intpol/r{_}.pdb', topology=top, coordinate=((pos0+(_*(pos25-pos0))/16.)*unit.angstrom))


  pos25 = np.asarray(read_cor(file_dir='../0_setups/guess/r25.cor' ).value_in_unit(unit=unit.angstrom))
  pos26 = np.asarray(read_cor(file_dir='../0_setups/guess/r26.cor').value_in_unit(unit=unit.angstrom))
  
  for _ in range(16, 32):
    write_pdb(to_file=f'./intpol/r{_}.pdb', topology=top, coordinate=((pos25+((_-16)*(pos26-pos25))/16.)*unit.angstrom))
  
  pos26 = np.asarray(read_cor(file_dir='../0_setups/guess/r26.cor').value_in_unit(unit=unit.angstrom))
  pos30 = np.asarray(read_cor(file_dir='../0_setups/guess/r30.cor').value_in_unit(unit=unit.angstrom))
  
  for _ in range(32, 56):
    write_pdb(to_file=f'./intpol/r{_}.pdb', topology=top, coordinate=((pos26+((_-32)*(pos30-pos26))/24.)*unit.angstrom))
  
  pos30 = np.asarray(read_cor(file_dir='../0_setups/guess/r30.cor').value_in_unit(unit=unit.angstrom))
  pos31 = np.asarray(read_cor(file_dir='../0_setups/guess/r31.cor').value_in_unit(unit=unit.angstrom))

  for _ in range(56, 80):
    write_pdb(to_file=f'./intpol/r{_}.pdb', topology=top, coordinate=((pos30+((_-56)*(pos31-pos30))/24.)*unit.angstrom))

  all_coordinates = [read_pdb(file_dir=f'./intpol/r{_}.pdb') for _ in range(80)]
  write_dcd(to_file=f'./intpol/intpol.dcd', topology=top, coordinates=all_coordinates)
