# Heating dynamics with DNA fixed in place.
# Zilin Song, 20231204
# 

import sys

from openmm.app import (Simulation, 
                        CharmmParameterSet, CharmmPsfFile, CharmmCrdFile, PDBFile, 
                        DCDReporter, StateDataReporter, CheckpointReporter, 
                        PME, HBonds, )

from openmm import (Platform, System, State, 
                    LangevinIntegrator, MonteCarloBarostat, 
                    unit, Vec3, XmlSerializer, )

if __name__ == '__main__':
  pressure      =   1.
  friction_coef =   2.
  timestep_size =    .001

  init_temperature =   0.
  incr_temperature =   1.
  num_of_incr      = 300.

  heating_steps = 1_000

  i_replica = str(int(sys.argv[1]))

  """Setups."""
  # FFs. 
  toppar = CharmmParameterSet('../../fig_dna0_path/0_setups/toppar/top_all36_na.rtf', 
                              '../../fig_dna0_path/0_setups/toppar/par_all36_na.prm', 
                              '../../fig_dna0_path/0_setups/toppar/toppar_water_ions.str', )
  
  # System.
  psf = CharmmPsfFile('../2_mep_box/cors/dna.psf')
  psf.setBox(a=65*unit.angstrom, b=65*unit.angstrom, c=65*unit.angstrom)
  system: System = psf.createSystem(toppar, 
                                    nonbondedMethod=PME, 
                                    nonbondedCutoff=12*unit.angstrom, 
                                    switchDistance =10*unit.angstrom, 
                                    constraints    =HBonds, )
  ## Fixing: refer to openmm.app.internal.charmm.topologyobjects.py
  for atom in psf.atom_list:
    # OpenMM standard nucleric acid names.
    if atom.residue.resname in ['A', 'G', 'C', 'U', 'I', 'DA', 'DG', 'DC', 'DT', 'DI']:
      system.setParticleMass(atom.idx, 0.)
  
  # Coordinates: align CHARMM coordinates to OpenMM reference.
  cor, coordinates = CharmmCrdFile(f'../2_mep_box/cors/r{i_replica}.cor'), []
  for coord in cor.positions:
    coordinates.append(coord.value_in_unit(unit.angstrom) + Vec3(32.5, 32.5, 32.5))
  coordinates = coordinates*unit.angstrom

  # Integrator.
  integrator = LangevinIntegrator(init_temperature*unit.kelvin, 
                                  friction_coef*unit.picosecond**-1, 
                                  timestep_size*unit.picosecond, )
  
  # Platform.
  platform = Platform.getPlatformByName('CUDA')
  platprop = dict(CudaPrecision='mixed')

  # Simulation.
  simulation = Simulation(topology          =psf.topology, 
                          system            =system, 
                          integrator        =integrator, 
                          platform          =platform, 
                          platformProperties=platprop, )
  simulation.context.setPositions(coordinates)
  
  """Minimize energy."""
  print('Minimizing...', flush=True)
  init_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
  simulation.minimizeEnergy()
  finl_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()

  print(f'Energy minimization:\n  init_energy: {str(init_energy)};\n  finl_energy: {str(finl_energy)}.', flush=True)

  """Reporters."""
  # simulation.reporters.append(       DCDReporter(f'./logs_heat/r{i_replica}.dcd', 1_000))
  # simulation.reporters.append(CheckpointReporter(f'./logs_heat/r{i_replica}.chk', 1_000))
  simulation.reporters.append( StateDataReporter(f'./logs_heat/r{i_replica}.log', 10_000, 
                                                 step=True, time=True, potentialEnergy=True, temperature=True, 
                                                 progress=True, remainingTime=True, volume=True, speed=True, 
                                                 totalSteps=heating_steps*num_of_incr, ))
  
  """Heating dynamics."""
  simulation.context.setVelocitiesToTemperature(init_temperature*unit.kelvin)

  for i_heat in range(1, int(num_of_incr+1)):
    i_temperature = round(init_temperature + i_heat*incr_temperature)
    integrator.setTemperature(i_temperature*unit.kelvin)
    
    if i_heat % 30 == 0:
      print(f'Replica {i_replica:>2} Heating phase: {i_heat}. T = {i_temperature:>4} Kelvin.', flush=True)

    simulation.step(heating_steps)

  """Outputs."""
  print(f'Done heating.')
  end: State = simulation.context.getState(getPositions=True, getVelocities=True)

  with open(f'./logs_heat/r{i_replica}.xml', 'w') as f:
    f.write(XmlSerializer.serialize(end))

  with open(f'./cors_heat/r{i_replica}.pdb', 'w') as f:
    PDBFile.writeModel(topology=psf.topology, positions=end.getPositions(), file=f)
