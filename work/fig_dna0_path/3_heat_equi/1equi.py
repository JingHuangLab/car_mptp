# Equilibration dynamics with DNA fixed in place.
# Zilin Song, 20231204
# 

import sys

from openmm.app import (Simulation, 
                        CharmmParameterSet, CharmmPsfFile, PDBFile, 
                        DCDReporter, StateDataReporter, CheckpointReporter, 
                        PME, HBonds, )

from openmm import (Platform, System, State, 
                    LangevinIntegrator, MonteCarloBarostat, 
                    unit, XmlSerializer, )

if __name__ == '__main__':
  temperature   = 300.
  pressure      =   1.
  friction_coef =   2.
  timestep_size =    .001

  i_replica = str(int(sys.argv[1]))

  """Setups."""
  # FFs. 
  toppar = CharmmParameterSet('../0_setups/toppar/top_all36_na.rtf', 
                              '../0_setups/toppar/par_all36_na.prm', 
                              '../0_setups/toppar/toppar_water_ions.str', )
  
  # System.
  psf = CharmmPsfFile('../2_mep_box/cors/dna.psf')
  psf.setBox(a=65*unit.angstrom, b=65*unit.angstrom, c=65*unit.angstrom)
  system: System = psf.createSystem(toppar, 
                                    nonbondedMethod=PME, 
                                    nonbondedCutoff=12*unit.angstrom, 
                                    switchDistance =10*unit.angstrom, 
                                    constraints    =HBonds, )
  system.addForce(MonteCarloBarostat(pressure*unit.bar, temperature*unit.kelvin))
  ## Fixing: refer to openmm.app.internal.charmm.topologyobjects.py
  for atom in psf.atom_list:
    # OpenMM standard nucleric acid names.
    if atom.residue.resname in ['A', 'G', 'C', 'U', 'I', 'DA', 'DG', 'DC', 'DT', 'DI', ]:
      system.setParticleMass(atom.idx, 0.)
  
  # Integrator.
  integrator = LangevinIntegrator(temperature*unit.kelvin, 
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

  # State from previous run.
  # with open(f'./logs_heat/r{i_replica}.xml', 'r') as f:
  #   state = XmlSerializer.deserialize(f.read())
  # simulation.context.setState(state)
  simulation.context.setPositions(PDBFile(f'./cors_heat/r{i_replica}.pdb').positions)
  
  """Reporters."""
  # simulation.reporters.append(       DCDReporter(f'./logs_equi/r{i_replica}.dcd', 1_000))
  # simulation.reporters.append(CheckpointReporter(f'./logs_equi/r{i_replica}.chk', 1_000))
  simulation.reporters.append( StateDataReporter(f'./logs_equi/r{i_replica}.log', 10_000, 
                                                 step=True, time=True, potentialEnergy=True, temperature=True, 
                                                 progress=True, remainingTime=True, volume=True, speed=True, 
                                                 totalSteps=200_000, ))

  """Equilibration dynamics."""
  for i_equi in range(200):
    if i_equi % 20 == 0:
      print(f'Replica {i_replica:>2} Equilibration phase: {i_equi}.', flush=True)

    simulation.step(1_000)

  """Outputs."""
  print(f'Done equilibration.')
  end: State = simulation.context.getState(getPositions=True, getVelocities=True)

  with open(f'./logs_equi/r{i_replica}.xml', 'w') as f:
    f.write(XmlSerializer.serialize(end))

  with open(f'./cors_equi/r{i_replica}.pdb', 'w') as f:
    PDBFile.writeModel(topology=psf.topology, positions=end.getPositions(), file=f)
