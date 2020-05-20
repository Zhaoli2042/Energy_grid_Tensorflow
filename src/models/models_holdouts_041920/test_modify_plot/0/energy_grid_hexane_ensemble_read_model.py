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

# # try out the tensorboard code from the book
import os

#model_dir = 'models/1e6_data/'
model_dir = os.getcwd()
os.chdir(model_dir)
test_dataset = pd.read_csv("0_testset.csv")
train_dataset = pd.read_csv("0_trainset.csv")
# Now, we try to read dataset with ID, need to pop these IDs out
train_ID = train_dataset.pop('MOF.ID')
test_ID = test_dataset.pop('MOF.ID')

# make the y_act column as the "label": label is the actual value
train_labels = train_dataset.pop('y_act')
test_labels = test_dataset.pop('y_act')

def norm(x):
  return (x - train_dataset.mean()) / train_dataset.std()
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

## the model used selu
from functools import partial
RegularizedDense = partial(layers.Dense, activation='selu', 
                           kernel_initializer='lecun_normal')
                           #kernel_regularizer=keras.regularizers.L1L2(0.005,0.005))

n_members = 10

# load all the saved models
from tensorflow.keras.models import load_model
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'model_' + str(i + 1) + '.h5'
		# load model from file
		model = load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models


members = load_all_models(n_members)
print('Loaded %d models' % len(members))

def define_stacked_model(members):
    for i in range(len(members)):
	    model = members[i]
	    for layer in model.layers:
		    # make not trainable, can change it
		    layer.trainable = False
            
		    # rename to avoid 'unique layer name' issue
            # according to https://stackoverflow.com/questions/56886442/error-when-trying-to-rename-a-pretrained-model-on-tf-keras
		    layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define a multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = layers.concatenate(ensemble_outputs)
    hidden_1 = RegularizedDense(10)(merge)
    hidden_2 = RegularizedDense(10)(hidden_1)
    hidden_3 = RegularizedDense(10)(hidden_2)
    output = layers.Dense(1)(hidden_3)
    model = keras.models.Model(inputs=ensemble_visible, outputs=output)
    keras.utils.plot_model(model, show_shapes=True, 
                           to_file= 'meta_enlarged_model.png')
    # compile the model, need optimizer and loss function
    optimizer = tf.keras.optimizers.RMSprop(0.0005)

    model.compile(loss='mse',
    #model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

stacked_model = define_stacked_model(members)

# then we need to fit the model
def fit_stacked_model(model, inputX, inputy):
    # prepare the input data
    X = [inputX for _ in range(len(model.input))]
    # then model fit
    model.fit(X, inputy, epochs = 1000, verbose = 1,
              validation_split = 0.2)
    return model

# use the test set to get nearly perfect coefficients for the meta-model
fit_stacked_model(stacked_model, normed_test_data, test_labels)

def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose=0)
stacked_model.save('Meta_model.h5')
yhat = predict_stacked_model(stacked_model, normed_test_data)

# finally make prediction
yhat = list(yhat.flat)

yhat = np.asarray(yhat)

#yhat[yhat < 0.0] = 0
#yhat[yhat > 160] = 160
k = tf.keras.losses.mse(test_labels, yhat)
mmm = tf.keras.losses.mae(test_labels, yhat)
RMSE = str.format('{0:.2f}', np.sqrt(k.numpy()))
MAE = str.format('{0:.2f}', mmm.numpy())
print(np.sqrt(k.numpy()),mmm.numpy())

# make a plot
plt.figure(figsize = (10,10), dpi = 800)
# now, add text to plot, number of points and statistics
plt.text(10,170, "Testing Data",
         fontsize = 20)
plt.text(10,160, str(normed_test_data.shape[0]) + " Points", 
         fontsize = 20)
plt.text(110,20, "RMSE=" + RMSE + r' cm$^{\rm 3}$/cm$^{\rm 3}$', 
         fontsize = 20)
plt.text(110,10, "MAE=" + MAE + r" cm$^{\rm 3}$/cm$^{\rm 3}$", 
         fontsize = 20)
plt.scatter(test_labels, yhat)
plt.xlabel(r'Test True Values [cm$^{\rm 3}$/cm$^{\rm 3}$]', fontsize = 20)
plt.ylabel(r'Test weighted Predictions [cm$^{\rm 3}$/cm$^{\rm 3}$]', fontsize = 20)
lims = [0, 180]
plt.xlim(lims)
plt.ylim(lims)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
_ = plt.plot(lims, lims)