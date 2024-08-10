# Brutal force MD for computing MFPTs from Muller upper product to lower reactant.
# Zilin Song, 20231219
# 

import sys, os, numpy as np
sys.dont_write_bytecode=True
sys.path.insert(0, os.getcwd())

if __name__ == '__main__':
  mfpt_steps = []
  with open('mfpt_ref.log', 'r') as fi:
    lines = fi.readlines()
    for line in lines:
      words = line.split(':')
      mfpt_steps.append(int(words[-2].strip()))

  mfpts = np.asarray(mfpt_steps) * .0001

  mean_mfpts = np.mean(mfpts)

  mean_mfpts_bootstrapped = []
  for i in range(10):
    mean_mfpts_bootstrapped.append(np.mean(np.random.choice(mfpts, size=200, replace=False)))

  mean_mfpts_bootstrapped = np.asarray(mean_mfpts_bootstrapped)

  print(np.log10(mean_mfpts), np.log10(7900))
  print(np.log10(mean_mfpts_bootstrapped))
  print(mean_mfpts_bootstrapped)