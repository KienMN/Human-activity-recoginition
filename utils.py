import pandas as pd
import numpy as np

def load_file(filepath):
  df = pd.read_csv(filepath, header=None, delim_whitespace=True)
  return df

def load_group(filenames, prefix=''):
  loaded = list()
  for name in filenames:
    data = load_file(prefix + name)
    loaded.append(data.values)
  # Stack group so that features are 3-dimension matrix
  loaded = np.dstack(loaded)
  return loaded

def load_dataset(group='train', prefix=''):
  filenames = ['{}_{}_{}.txt'.format(acc, dim, group) for acc in ['total_acc', 'body_acc', 'body_gyro'] for dim in ['x', 'y', 'z']]
  directory = prefix + group + '/Inertial Signals/'
  X = load_group(filenames, directory)
  y = load_file(prefix + group + '/y_' + group + '.txt').values
  return X, y