import numpy as np
import pandas as pd

from numpy.random import seed
seed(10)
input_filename = "../Data/latest_tob_simulation_topo_1e6_041920.csv"
output_filename = 'models/models_holdouts_041920/' + "permed_latest_tob_simulation_topo_1e6_041920.csv"

def shuffler(filename):
  df = pd.read_csv(filename, header=0)
  # return the pandas dataframe
  return df.reindex(np.random.permutation(df.index))


def main(outputfilename):
  shuffler(input_filename).to_csv(output_filename, sep=',')

main(output_filename)