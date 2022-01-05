# -*- coding: utf-8 -*-
"""DCN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VIAkLP-BwBmMRTXa01f8PjaKxU1SF9K9
"""

# import required packages
import sys
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

import torch
from sklearn.model_selection import StratifiedShuffleSplit

from fuxictr.datasets import data_generator
from fuxictr.datasets.taobao import FeatureEncoder
from fuxictr.utils import set_logger, print_to_json
from fuxictr.pytorch.models import DeepFM, ONN, FiBiNET, DCN, FGCNN
from fuxictr.pytorch.utils import seed_everything

# read the originial dataset
train_all = pd.read_csv('C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/Project Data/ProjectTrainingData.csv')

# have a look at the shape
train_all.shape

# the ratio of click and not click in the original dataset
train_all.click.value_counts()[1] / train_all.click.value_counts()[0]

# take a sample and keep the ratio of target variables
split = StratifiedShuffleSplit(n_splits=1, train_size=0.1, random_state=42)
for train_index, test_index in split.split(train_all, train_all["click"]):
    strat_set = train_all.loc[train_index]

# the ratio of click and not click in the sampled dataset
strat_set.click.value_counts()[1] / strat_set.click.value_counts()[0]

strat_set.shape

# reset the index
strat_set = strat_set.reset_index().drop(["index"], axis=1)

# split the dataset into training and validation datasets
# keep 80% for training and 20% for validation
# still keep the ratio of click and not click
split = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
for train_index, val_index in split.split(strat_set, strat_set["click"]):
    strat_train_set = strat_set.loc[train_index]
    strat_val_set = strat_set.loc[val_index]

# have a look at the ratios
print(strat_train_set.click.value_counts()[1] / strat_train_set.click.value_counts()[0])
print(strat_val_set.click.value_counts()[1] / strat_val_set.click.value_counts()[0])

# import the test dataset
test = pd.read_csv('C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/Project Data/ProjectTestData.csv')

# have a look at the shape
test.shape[0]

# because the model only accepts dataset with the label to calculate the logloss
# so we randomly assigned half of the labels as 0
slicer = test.shape[0] / 2
slicer = np.round(slicer).astype('int64')
nums = np.ones(test.shape[0])
nums[:slicer] = 0
np.random.shuffle(nums)

# add the labels
test['click'] = nums
test['click'] = test['click'].astype('int64')

# adjust the data format of hour column
# change the column name
from datetime import datetime
strat_train_set.rename(columns={"hour":"time_stamp"}, inplace=True)
strat_train_set['time_stamp'] = strat_train_set['time_stamp'].map(lambda x: datetime.strptime(str(x),"%y%m%d%H"))
strat_val_set.rename(columns={"hour":"time_stamp"}, inplace=True)
strat_val_set['time_stamp'] = strat_val_set['time_stamp'].map(lambda x: datetime.strptime(str(x),"%y%m%d%H"))
test.rename(columns={"hour":"time_stamp"}, inplace=True)
test['time_stamp'] = test['time_stamp'].map(lambda x: datetime.strptime(str(x),"%y%m%d%H"))

strat_train_set.to_csv("C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/train_sample.csv", index=False)
strat_val_set.to_csv("C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/val_sample.csv", index=False)
test.to_csv("C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/test_sample.csv", index=False)

# specify the features and lable
# make "id" not active to exclude this column
# convert "time_stamp" column to the hour, weekday, and is_weekend
# features are all categorical features
feature_cols = [{'name': ["id"], 'active': False, 'dtype': 'str', 'type': 'categorical'},
                {'name': ["time_stamp"], 'active': True, 'dtype': 'str', 'type': 'categorical', 'preprocess': 'convert_hour'},
                {'name': ["C1","banner_pos","site_id","site_domain","site_category","app_id","app_domain","app_category","device_id",
                "device_ip","device_model","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21"],
                'active': True, 'dtype': 'str', 'type': 'categorical'},
                {'name': ["weekday"], 'active': True, 'dtype': 'str', 'type': 'categorical', 'preprocess': 'convert_weekday'},
                {'name': ["weekend"], 'active': True, 'dtype': 'str', 'type': 'categorical', 'preprocess': 'convert_weekend'}]
label_col = {'name': 'click', 'dtype': float}

# set up the parameters for DCN model
# DCN
params = {
    'model_id': 'DCN_final_2',
    'dataset_id': 'DCN_dataset_final_2',

    # specify the file locations
    'train_data': 'C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/train_sample.csv',
    'valid_data': 'C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/val_sample.csv',
    'test_data': 'C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/test_sample.csv',
    'model_root': 'C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/checkpoints/',
    'data_root': 'C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/',
    
    # specify the features and labels
    'feature_cols': feature_cols,
    'label_col': label_col,

    # parameters for DCN models, the explanation could be found in the report

    # batch and epoch parameters
    'batch_norm': False,
    'batch_size': 10000,
    'epochs': 100,
    'every_x_epochs': 1,
    'optimizer': 'adam',
    'min_categr_count': 2,

    'learning_rate': 0.001,
    'task': 'binary_classification',
    'loss': 'binary_crossentropy',
    'metrics': ['logloss', 'AUC'],
    'monitor': {'AUC':1, 'logloss':-1},
    'monitor_mode': 'max',
    
    # parameters for DCN layers
    "crossing_layers": 3,
    'num_cross_layers': 3,
    'model_structure': 'parallel',
    'use_low_rank_mixture': False,
    'low_rank': 32,
    'num_experts': 4,
    'stacked_dnn_hidden_units':[500,500,500],
    'parallel_dnn_hidden_units': [500,500,500],
    'dnn_activations': "relu",
    "dnn_hidden_units": [2000, 2000, 2000],
    'debug': False,

    # embedding methods parameters
    'embedding_dim': 16,
    'embedding_droput': 0,
    'embedding_regularizer': 1e-8,

    # regularization
    'net_dropout': 0,
    'net_regularizer': 0,

    # model running parameters
    'patience': 2,
    'pickle_feature_encoder': True,
    'save_best_only': True,
    'seed': 2021,
    'shuffle': True,
    'use_hdf5': True,
    'verbose': 2,
    'version': 'pytorch',
    'workers': 3,
    'gpu': 0}

# log the results
set_logger(params)
logging.info('Start the demo...')
logging.info(print_to_json(params))
seed_everything(seed=params['seed'])

# set up an encoder to transform the data
feature_encoder = FeatureEncoder(feature_cols, 
                                 label_col, 
                                 dataset_id=params['dataset_id'], 
                                 data_root=params["data_root"],
                                 version=params['version'])
                                 
# transform the data   
feature_encoder.fit(train_data=params['train_data'], 
                    min_categr_count=params['min_categr_count'])

# shuffle and adjust the format of three datasets 
# for the input of neural network models
train_gen, valid_gen, test_gen = data_generator(feature_encoder,
                                                train_data=params['train_data'],
                                                valid_data=params['valid_data'],
                                                test_data=params['test_data'],
                                                batch_size=params['batch_size'],
                                                shuffle=params['shuffle'],
                                                use_hdf5=params['use_hdf5'])

# train the model using parameters
model_DCN = DCN(feature_encoder.feature_map, **params)
model_DCN.fit_generator(train_gen, validation_data=valid_gen, epochs=params['epochs'], verbose=params['verbose'])

logging.info('***** validation/test results *****')
model_DCN.load_weights(model_DCN.checkpoint)
model_DCN.evaluate_generator(valid_gen)
#model_DCN.evaluate_generator(test_gen)

from tqdm import tqdm
test_gen = tqdm(test_gen, disable=False, file=sys.stdout)
y_pred = []
y_true = []
for batch_data in test_gen:
    return_dict = model_DCN.forward(batch_data)
    y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
    y_true.extend(batch_data[1].data.cpu().numpy().reshape(-1))

import numpy as np
y_pred = np.array(y_pred, np.float64)
y_true = np.array(y_true, np.float64)

import pandas as pd
sub = pd.read_csv("C:/Users/xwang/Downloads/FuxiCTR-main_8/FuxiCTR-main/data/sampleSubmission.gz")

sub['click'] = y_pred
sub.to_csv("sampleSubmission_DCN_p6.csv", index=False)