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
model_dir = 'models/models_holdouts_041920/test_modify_plot/'
read_filename = model_dir + "permed_latest_tob_simulation_topo_1e6_041920.csv"
# try tensorflow probability to access the uncertainties of the predictions

# I renamed the energy grid columns, easier for pd to read
#raw_dataset = pd.read_csv("combined_hexane_.1bar_topo_no_ID.csv")
raw_dataset = pd.read_csv(read_filename)
dataset = raw_dataset.copy()
dataset.isna().sum()
dataset = dataset.dropna()

def norm(x, train_dataset):
  return (x - train_dataset.mean()) / train_dataset.std()

## the model used selu
from functools import partial
RegularizedDense = partial(layers.Dense, activation='selu', 
                           kernel_initializer='lecun_normal')
                           #kernel_regularizer=keras.regularizers.L1L2(0.005,0.005))
def build_model_selu():
  model = keras.Sequential([
    # why -2? because dataset has two more columns (ID and labels)
    layers.Dense(400, activation='selu', input_shape=[len(dataset.keys()) - 2],
                 kernel_initializer='lecun_normal'),
    #layers.AlphaDropout(rate=0.2),
    RegularizedDense(400),
    #layers.AlphaDropout(rate=0.2),
    RegularizedDense(400),
    #layers.AlphaDropout(rate=0.2),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    RegularizedDense(400),
    layers.Dense(1)
    #layers.Dense(1, activation=ashole),
    #layers.Dense(1, activation=ashole_extreme)
    #layers.Dense(1, activation=ashole_high)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0001)

  model.compile(loss='mse',
  #model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

EPOCHS = 500

# now I want to save these models, need iteration number
def fit_with_model_diff_act(act_fxn, iteration, new_dir):
  model_selu=build_model_selu()
  print("training model using activation function:", act_fxn)
  model = eval("model_" + act_fxn)
  #for tensorboard callbacks
  tensorboard_callback = keras.callbacks.TensorBoard(run_logdir)
  earlystop_callback = keras.callbacks.EarlyStopping(monitor ='mse', 
                                                     min_delta = 0.00005,
                                                     patience=100, verbose=1)
  history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.3, verbose=0,
    #callbacks=[earlystop_callback, tfdocs.modeling.EpochDots()])
    callbacks=[tfdocs.modeling.EpochDots()])
    #callbacks=[tensorboard_callback])

  test_predictions = model.predict(normed_test_data).flatten()
  train_predictions = model.predict(normed_train_data).flatten()
  
  # save the model
  
  k = tf.keras.losses.mse(test_labels, test_predictions)
  mmm = tf.keras.losses.mae(test_labels, test_predictions)
  print("model: ",act_fxn)
  print(np.sqrt(k.numpy()),mmm.numpy())
  # if the model's RMSE is too large, label it as "not_good"
  filename = new_dir + 'model_' + str(iteration + 1) + '.h5'
  if (np.sqrt(k.numpy()) > 20):
      filename = new_dir + 'model_' + str(iteration + 1) + 'notgood' + '.h5'
  model.save(filename)
  # tf.summary.histogram(“train_prediction”, train_predictions)
  # tf.summary.histogram(“test_prediction”, test_predictions)
  
  # plt.savefig(test_plot_name)

  test_dataset['Pred']=test_predictions
  test_dataset['y_act']=test_labels
  #filter the dataset frame based on column values
  test_under=test_dataset[(test_dataset['y_act'] - test_dataset['Pred']) > 30]
  test_over=test_dataset[(test_dataset['Pred'] - test_dataset['y_act']) > 30]

  # test_under.to_excel('New_CBCFC_test_under.xlsx')
  # test_over.to_excel('New_CBCFC_test_over.xlsx')

  # get how many outliers
  Num_under_outliers = sum((test_labels - test_predictions) > 50)
  Num_over_outliers = sum((test_predictions - test_labels) > 50)

  Num_under_outliers_train = sum((train_labels - train_predictions) > 50)
  Num_over_outliers_train = sum((train_predictions - train_labels) > 50)

  return test_dataset

#n_holdouts = 8
# holdout 3 has 2 models not good, retrain
n_holdouts = 8
just_train = False # if train all models, make it false
n_members = 10
for part in range(7, n_holdouts):
  if ((part == (n_holdouts - 1)) and (not just_train)):
    test_dataset = dataset.loc[part*1000:dataset.shape[0], :]
  else:
    test_dataset = dataset.loc[part*1000:((part + 1)*1000-1), :]  
  new_model_dir = model_dir + str(part) + '/'
  if (not os.path.isdir(new_model_dir)):
    os.mkdir(new_model_dir)
  train_dataset = dataset.drop(test_dataset.index)
  train_dataset.to_excel(new_model_dir + str(part) + '_trainset.xlsx')
  test_dataset.to_excel(new_model_dir + str(part) + '_testset.xlsx')
  # Now, we try to read dataset with ID, need to pop these IDs out
  train_ID = train_dataset.pop('MOF.ID')
  test_ID = test_dataset.pop('MOF.ID')

  # make the y_act column as the "label": label is the actual value
  train_labels = train_dataset.pop('y_act')
  test_labels = test_dataset.pop('y_act')
   
  # normalize using the training dataset
  normed_train_data = norm(train_dataset, train_dataset)
  normed_test_data = norm(test_dataset, train_dataset)
  
  
  pred_test_set = pd.DataFrame({'Pred1':[]})
  for a in range(2,n_members):
      # start with the normed test set: it is untouched!
      print("Session: ", a)
      test_dataset_huber = fit_with_model_diff_act("selu", a, new_model_dir)
      pred_test_set['Pred' + str(a)] = test_dataset_huber['Pred']