# Transition PathCV optimization on Di-Alanine isomerization: Evolves the path.
# Zilin Song, 20230915
# 

import sys, os
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

import numpy as np, pickle as pkl, copy

if __name__ == '__main__':
  # INPUTs. ----------------------------------------------------------------------------------------
  ## Parse sys.argv
  num_replicas = int(sys.argv[1].split('num_replicas:')[1])
  dir_input    = str(sys.argv[2].split('dir_input:' )[1])
  dir_output   = str(sys.argv[3].split('dir_output:')[1])

  # OUTPUTs. ---------------------------------------------------------------------------------------
  ## Path colvars & Sampler kwargs.
  path_colvar = np.load(f'{dir_input}/path_colvar.npy')
  samplers_all_dict = pkl.load(open(f'{dir_input}/samplers_all.pkl', 'rb'))
  print(samplers_all_dict.keys())
  
  path_colvar_extended = np.zeros((num_replicas, path_colvar.shape[1]))
  samplers_all_dict_extended = {}
  samplers_all_dict_extended.update({f'weight_per_atom':     samplers_all_dict['weight_per_atom'], 
                                     f'weight_per_dof' :     samplers_all_dict['weight_per_dof'], 
                                     f'method_alignment':    samplers_all_dict['method_alignment'], 
                                     f'method_path_tangent': samplers_all_dict['method_path_tangent'], })

  for i in range(13):
    if i < 7:
      path_colvar_extended[i, :] = np.copy(path_colvar[i, :])
      sampler_dict_extended = {
          f'replica{str(i)}_whoami'         : i, 
          f'replica{str(i)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
      }
      print(i, i)
    elif i == 7:
      path_colvar_extended[i  , :] = np.copy(    path_colvar[i, :]                    )
      path_colvar_extended[i+1, :] = np.copy(.75*path_colvar[i] + .25*path_colvar[i+1])
      path_colvar_extended[i+2, :] = np.copy(.50*path_colvar[i] + .50*path_colvar[i+1])
      path_colvar_extended[i+3, :] = np.copy(.25*path_colvar[i] + .75*path_colvar[i+1])
      sampler_dict_extended = {
          f'replica{str(i  )}_whoami'         : i  , 
          f'replica{str(i  )}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
          f'replica{str(i+1)}_whoami'         : i+1, 
          f'replica{str(i+1)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
          f'replica{str(i+2)}_whoami'         : i+2, 
          f'replica{str(i+2)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
          f'replica{str(i+3)}_whoami'         : i+3, 
          f'replica{str(i+3)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
      }
      print(i, i, i+1, i+2, i+3, )
    elif i == 8:
      path_colvar_extended[i+3, :] = np.copy(path_colvar[i, :])
      path_colvar_extended[i+4, :] = np.copy(.75*path_colvar[i] + .25*path_colvar[i+1])
      path_colvar_extended[i+5, :] = np.copy(.50*path_colvar[i] + .50*path_colvar[i+1])
      path_colvar_extended[i+6, :] = np.copy(.25*path_colvar[i] + .75*path_colvar[i+1])
      sampler_dict_extended = {
          f'replica{str(i+3)}_whoami'         : i+3, 
          f'replica{str(i+3)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
          f'replica{str(i+4)}_whoami'         : i+4, 
          f'replica{str(i+4)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
          f'replica{str(i+5)}_whoami'         : i+5, 
          f'replica{str(i+5)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
          f'replica{str(i+6)}_whoami'         : i+6, 
          f'replica{str(i+6)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
      }
      print(i, i+3, i+4, i+5, i+6)
    elif i == 9:
      path_colvar_extended[i+6, :] = np.copy(path_colvar[i, :])
      path_colvar_extended[i+7, :] = np.copy(.50*path_colvar[i] + .50*path_colvar[i+1])
      sampler_dict_extended = {
          f'replica{str(i+6)}_whoami'         : i+6, 
          f'replica{str(i+6)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
          f'replica{str(i+7)}_whoami'         : i+7, 
          f'replica{str(i+7)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
      }
      print(i, i+6, i+7)
    else: 
      path_colvar_extended[i+7, :] = np.copy(path_colvar[i, :])
      sampler_dict_extended = {
          f'replica{str(i+7)}_whoami'         : i+7, 
          f'replica{str(i+7)}_context_coord'  : copy.deepcopy(samplers_all_dict[f'replica{str(i)}_context_coord']), 
      }
      print(i, i+7)
      
    samplers_all_dict_extended.update(sampler_dict_extended)

  print(path_colvar_extended.shape, path_colvar.shape)
  np.save(f'{dir_output}/path_colvar.npy', path_colvar_extended)

  with open(f'{dir_output}/samplers_all.pkl', 'wb') as fo:
    pkl.dump(samplers_all_dict_extended, fo)
    print(samplers_all_dict_extended.keys(), flush=True)
