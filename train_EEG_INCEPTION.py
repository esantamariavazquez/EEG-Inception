# Import libraries
import numpy as np
import h5py, os

import models, utils

from keras import backend as K

#%% PARAMETERS

experiment = 'PaperReview2'
dataset = '128Hz_0.5-45Hz_CAR'
model_name = 'EEGInceptionV1_500_250_125_ORIGINAL'

#%% HYPERPARAMETERS

# MEDUSANET hyparams
hyparams = dict()
hyparams["n_cha"] = 8
hyparams["dropout_rate"] = 0.25
hyparams["activation"] = 'elu'
hyparams["n_classes"] = 2
hyparams["lr"] = 0.001

#%% PATHS
split_dataset_path = 'dataset/SPLIT_DATASET_%s.h5' % (dataset)
model_path = "models/%s/%s_%s.h5" % (experiment, model_name, dataset)
weights_path = "models/%s/%s_%s_WEIGHTS.h5" % (experiment, model_name, dataset)

#%% DATASETS

hf = h5py.File(split_dataset_path, 'r')
# Train group
train_group = hf.get('train_set')
train_features = np.array(train_group.get("features"))
train_erp_labels = np.array(train_group.get("erp_labels"))
train_codes = np.array(train_group.get("codes"))
train_trials = np.array(train_group.get("trials"))
train_sequences = np.array(train_group.get("sequences"))
train_subjects = np.array(train_group.get("subjects"))
train_database_ids = np.array(train_group.get("database_ids"))
train_run_indexes = np.array(train_group.get("run_indexes"))
train_matrix_indexes = np.array(train_group.get("matrix_indexes"))
train_target = np.array(train_group.get("target"))
train_matrix_dims = np.array(train_group.get("matrix_dims"))
# Dev group
dev_group = hf.get('dev_set')
dev_features = np.array(dev_group.get("features"))
dev_erp_labels = np.array(dev_group.get("erp_labels"))
dev_codes = np.array(dev_group.get("codes"))
dev_trials = np.array(dev_group.get("trials"))
dev_sequences = np.array(dev_group.get("sequences"))
dev_subjects = np.array(dev_group.get("subjects"))
dev_database_ids = np.array(dev_group.get("database_ids"))
dev_run_indexes = np.array(dev_group.get("run_indexes"))
dev_matrix_indexes =  np.array(dev_group.get("matrix_indexes"))
dev_target = np.array(dev_group.get("target"))
dev_matrix_dims = np.array(dev_group.get("matrix_dims"))
# Test group
test_group = hf.get('test_set')
test_features = np.array(test_group.get("features"))
test_erp_labels = np.array(test_group.get("erp_labels"))
test_codes = np.array(test_group.get("codes"))
test_trials = np.array(test_group.get("trials"))
test_sequences = np.array(test_group.get("sequences"))
test_subjects = np.array(test_group.get("subjects"))
test_database_ids = np.array(test_group.get("database_ids"))
test_run_indexes = np.array(test_group.get("run_indexes"))
test_matrix_indexes = np.array(test_group.get("matrix_indexes"))
test_target = np.array(test_group.get("target"))
test_matrix_dims = np.array(test_group.get("matrix_dims"))
# Close file
hf.close()

# Reshape epochs
train_features = train_features.reshape(train_features.shape[0], train_features.shape[1], train_features.shape[2], 1)
dev_features = dev_features.reshape(dev_features.shape[0], dev_features.shape[1], dev_features.shape[2], 1)
test_features = test_features.reshape(test_features.shape[0], test_features.shape[1], test_features.shape[2], 1)
    
train_erp_labels = utils.one_hot_labels(train_erp_labels)
dev_erp_labels = utils.one_hot_labels(dev_erp_labels)
test_erp_labels = utils.one_hot_labels(test_erp_labels)

#%%  TRAINING
# Print hyparams
print()
print(hyparams)
print()

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Create model
model = models.EEGInceptionV1(**hyparams)

# Print model summary
model.summary()

# Fit model
fit_hist = model.fit(train_features,
                     train_erp_labels,
                     epochs=500,
                     batch_size=2048,
                     validation_data=(dev_features, dev_erp_labels),
                     callbacks=models.get_training_callbacks())

# Save
model.save(model_path)
model.save_weights(weights_path)

# Clear session to avoid filling the RAM memory
K.clear_session()


