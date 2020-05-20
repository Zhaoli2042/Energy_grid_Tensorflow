import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
file_dir = 'all_8000_outliers/isotherm/new_isotherm/'
list_files=(glob.glob(file_dir + "*.csv"))
pc_file = open(file_dir + 'pc_structures.txt', 'w')
for a in list_files:
   print(a)
   raw_dataset = pd.read_csv(a)
   structure_name = a.split('.')[0]
   # if the files are in a different directory, need to eliminate the pre-string
   if ("\\" in structure_name):
       structure_name=structure_name.split('\\')[len(structure_name.split('\\')) - 1]
   print(structure_name)
   raw_dataset = raw_dataset.sort_values(by = ['Pres'])
   plot_limit_y = max(raw_dataset['Loading'])
   plot_limit_x = max(raw_dataset['Pres']/1e5)
   plt.figure(figsize = (12,10), dpi = 800)
   plt.text(0.05*plot_limit_x,0.95*plot_limit_y, "Testing Data:" + structure_name,
            fontsize = 20)
   plt.plot(raw_dataset['Pres']/1e5, raw_dataset['Loading'], '-bo')
   plt.ylabel(r"GCMC Loading [cm$^{\rm 3}$/cm$^{\rm 3}$]", fontsize = 20)
   plt.xlabel('Pressure [Bar]', fontsize = 20)
   plt.xticks(fontsize = 20)
   plt.yticks(fontsize = 20)
   plt.savefig(file_dir + "Isotherm_" + structure_name + ".png")
   plt.clf()
   # also detect the largest step change in the isotherms
   for b in range(0, raw_dataset.shape[0]-1):
      sort_dataset = raw_dataset.sort_values(by='Pres', ascending=True).reset_index(drop = True)
      change_loading = sort_dataset['Loading'][b + 1] - sort_dataset['Loading'][b]
      if (b == 0):
          stored_loading = change_loading
          stored_index = b
      else:
          if (change_loading > stored_loading):
              stored_loading = change_loading
              stored_index = b
   print(sort_dataset['Loading'][stored_index], sort_dataset['Loading'][stored_index+1])
   pc_file.write("%s, %i, %i, %.2f\n" % (str(sort_dataset['MOF'][stored_index]), 
                                   sort_dataset['Pres'][stored_index], 
                                   sort_dataset['Pres'][stored_index + 1],
                                   (sort_dataset['Pres'][stored_index] + sort_dataset['Pres'][stored_index + 1])/2))

pc_file.close()