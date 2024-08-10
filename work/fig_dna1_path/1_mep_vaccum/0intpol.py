# MEP in vaccume for DNA wc -> hg transition.
# Zilin Song, 20231204
# 

import sys, numpy as np
sys.dont_write_bytecode=True

from helpers import read_top, read_pdb, read_cor, write_dcd, write_pdb, unit

if __name__ == '__main__':
  top = read_top(file_dir='../0_setups/guess/dna.psf')

  pos0 = np.asarray(read_cor(file_dir='../0_setups/guess/r0.cor').value_in_unit(unit=unit.angstrom))
  pos2 = np.asarray(read_cor(file_dir='../0_setups/guess/r1.cor').value_in_unit(unit=unit.angstrom))
  
  for _ in range(25):
    write_pdb(to_file=f'./intpol/r{_}.pdb', topology=top, coordinate=((pos0+(_*(pos2-pos0))/25.)*unit.angstrom))


  pos2  = np.asarray(read_cor(file_dir='../0_setups/guess/r1.cor' ).value_in_unit(unit=unit.angstrom))
  pos14 = np.asarray(read_cor(file_dir='../0_setups/guess/r14.cor').value_in_unit(unit=unit.angstrom))
  
  for _ in range(25, 60):
    write_pdb(to_file=f'./intpol/r{_}.pdb', topology=top, coordinate=((pos2+((_-25)*(pos14-pos2))/35.)*unit.angstrom))
  
  pos14 = np.asarray(read_cor(file_dir='../0_setups/guess/r14.cor').value_in_unit(unit=unit.angstrom))
  pos31 = np.asarray(read_cor(file_dir='../0_setups/guess/r31.cor').value_in_unit(unit=unit.angstrom))
  
  for _ in range(60, 80):
    write_pdb(to_file=f'./intpol/r{_}.pdb', topology=top, coordinate=((pos14+((_-60)*(pos31-pos14))/20.)*unit.angstrom)) # Note the 11.


  all_coordinates = [read_pdb(file_dir=f'./intpol/r{_}.pdb') for _ in range(80)]

  write_dcd(to_file=f'./intpol/intpol.dcd', topology=top, coordinates=all_coordinates)
