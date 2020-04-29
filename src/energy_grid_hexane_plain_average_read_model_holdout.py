from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# need to set random seed
from numpy.random import seed
seed(10)
tf.random.set_seed(20)

from tensorflow.keras.models import load_model
# read the models and do an average

# import custom huber function
import custom_huber as huber

# import datetime
import datetime

# # try out the tensorboard code from the book
import os
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

#model_dir = 'models/1e6_data/'
model_dir = 'models/models_holdouts_041920/'
# try tensorflow probability to access the uncertainties of the predictions

# I renamed the energy grid columns, easier for pd to read
#raw_dataset = pd.read_csv("combined_hexane_.1bar_topo_no_ID.csv")
input_filename = model_dir + "permed_latest_tob_simulation_topo_1e6_041920.csv"
raw_dataset = pd.read_csv(input_filename)
dataset = raw_dataset.copy()
dataset.isna().sum()
dataset = dataset.dropna()

def norm(x, trainset):
  return (x - trainset.mean()) / trainset.std()

n_holdouts = 8
for part in range(6, n_holdouts):
  if (part == (n_holdouts - 1)):
    test_dataset = dataset.loc[part*1000:dataset.shape[0], :]
  else:
    test_dataset = dataset.loc[part*1000:((part + 1)*1000-1), :]
  new_model_dir = model_dir + str(part) + '/'

  train_dataset = dataset.drop(test_dataset.index)
  # Now, we try to read dataset with ID, need to pop these IDs out
  train_ID = train_dataset.pop('MOF.ID')
  test_ID = test_dataset.pop('MOF.ID')

  # make the y_act column as the "label": label is the actual value
  train_labels = train_dataset.pop('y_act')
  test_labels = test_dataset.pop('y_act')

  normed_train_data = norm(train_dataset, train_dataset)
  normed_test_data = norm(test_dataset, train_dataset)
  
  n_members = 10

  pred_test_set = pd.DataFrame({'Pred1':[]})
  for i in range(n_members):
    # define filename for this ensemble
    filename = new_model_dir + 'model_' + str(i + 1) + '.h5'
    # load model from file
    model = load_model(filename)
    # add to list of members
    print('>loaded %s' % filename)
    pred = model.predict(normed_test_data)
    pred_test_set['Pred' + str(i)] = np.asarray(list(pred.flat))
  
  averaged_pred = pred_test_set.T.mean()
  k = tf.keras.losses.mse(test_labels, averaged_pred)
  mmm = tf.keras.losses.mae(test_labels, averaged_pred)
  print(np.sqrt(k.numpy()),mmm.numpy())
  RMSE = str.format('{0:.2f}', np.sqrt(k.numpy()))
  MAE = str.format('{0:.2f}', mmm.numpy())
  print(np.sqrt(k.numpy()),mmm.numpy())
  plt.figure(figsize = (12,10), dpi = 800)
  plt.text(10,170, "Testing Data",
           fontsize = 20)
  plt.text(10,160, str(normed_test_data.shape[0]) + " Points", 
           fontsize = 20)
  plt.text(100,20, "RMSE=" + RMSE + ' cm3/cm3', 
           fontsize = 20)
  plt.text(100,10, "MAE=" + MAE + " cm3/cm3", 
           fontsize = 20)
  plt.text(110, 170, "Model_Average", fontsize = 20)
  plt.scatter(test_labels, averaged_pred, c = pred_test_set.T.std(),
              vmin = 0, vmax = np.max(pred_test_set.T.std()))
  plt.xlabel('Test True Values [cm3/cm3]')
  plt.ylabel('Test Mean Predictions [cm3/cm3]')
  lims = [0, 180]
  plt.xlim(lims)
  plt.ylim(lims)
  plt.colorbar()
  _ = plt.plot(lims, lims)
  plt.savefig(new_model_dir + "Model_Average.png")
  plt.clf()

  for i in range(n_members):
    k = tf.keras.losses.mse(test_labels, pred_test_set['Pred' + str(i)])
    mmm = tf.keras.losses.mae(test_labels, pred_test_set['Pred' + str(i)])
    print(np.sqrt(k.numpy()),mmm.numpy())
    RMSE = str.format('{0:.2f}', np.sqrt(k.numpy()))
    MAE = str.format('{0:.2f}', mmm.numpy())
    print(np.sqrt(k.numpy()),mmm.numpy())
    plt.figure(figsize = (12,10), dpi = 800)
    plt.text(10,170, "Testing Data",
             fontsize = 20)
    plt.text(10,160, str(normed_test_data.shape[0]) + " Points", 
             fontsize = 20)
    plt.text(100,20, "RMSE=" + RMSE + ' cm3/cm3', 
             fontsize = 20)
    plt.text(100,10, "MAE=" + MAE + " cm3/cm3", 
             fontsize = 20)
    plt.text(140, 170, "Model" + str(i + 1), fontsize = 20)
    plt.scatter(test_labels, pred_test_set['Pred' + str(i)], c = pred_test_set.T.std(),
                vmin = 0, vmax = np.max(pred_test_set.T.std()))
    plt.xlabel('Test True Values [cm3/cm3]')
    plt.ylabel('Test Predictions [cm3/cm3]')
    lims = [0, 180]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.colorbar()
    _ = plt.plot(lims, lims)
    plt.savefig(new_model_dir + "Model_" + str(i + 1) + ".png")
    plt.clf()
  
  yhat = averaged_pred
  # lets filter out the under-predicted outliers, we have a lot lower pressure sim done
  test_under_pred = test_dataset[(test_labels - yhat) > 30]
  under_pred_locations = test_ID[(test_labels - yhat) > 30]
  test_under_pred.to_excel(new_model_dir + 'over_outlier.xlsx')
  under_pred_locations.to_excel(new_model_dir + 'over_out_names.xlsx')
  test_labels[(test_labels - yhat) > 30].to_excel(new_model_dir + 'over_labels.xlsx')
  abc = pd.Series(yhat)
  abc[(test_labels.values - abc) > 30].to_excel(new_model_dir + 'over_predictions.xlsx')

  #averaged_pred[abs(test_labels.values - yhat) > 30].to_excel(new_model_dir + 'over_plain_average_out.xlsx')
  pred_test_set[abs(test_labels.values - yhat) > 30].to_excel(new_model_dir + 'all_outliers_pred.xlsx')
  # also filter out those with huge standard deviations
  pred_test_set[pred_test_set.T.std() > 30].to_excel(new_model_dir + 'predict_large_std.xlsx')
  test_dataset['ID'] = test_ID
  test_dataset['Actual'] = test_labels
  test_dataset['Pred_Mean'] = averaged_pred
  test_dataset.set_index(np.arange(0,test_dataset.shape[0]))[pred_test_set.T.std() > 30].to_excel(new_model_dir + 'actual_large_std.xlsx')
  test_dataset.set_index(np.arange(0,test_dataset.shape[0]))[abs(test_labels.values - yhat) > 30].to_excel(new_model_dir + 'all_outliers_actual.xlsx')