"""
Usage example of EEG-Inception with an ERP-based BCI dataset from:


Download the dataset from:
https://www.kaggle.com/esantamaria/gibuva-erpbci-dataset

"""

#%% IMPORT LIBRARIES
import numpy as np
import h5py, os
from EEGInception import EEGInception
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder

#%% PARAMETERS

dataset_path = 'PATH TO DATASET (hdf5 file)'

#%% HYPERPARAMETERS

input_time = 1000
fs = 128
n_cha = 8
filters_per_branch = 8
scales_time = (500, 250, 125)
dropout_rate = 0.25
activation = 'elu'
n_classes = 2
learning_rate = 0.001

#%% LOAD DATASET
hf = h5py.File(dataset_path, 'r')
features = np.array(hf.get("features"))
erp_labels = np.array(hf.get("erp_labels"))
codes = np.array(hf.get("codes"))
trials = np.array(hf.get("trials"))
sequences = np.array(hf.get("sequences"))
matrix_indexes = np.array(hf.get("matrix_indexes"))
run_indexes = np.array(hf.get("run_indexes"))
subjects = np.array(hf.get("subjects"))
database_ids = np.array(hf.get("database_ids"))
target = np.array(hf.get("target"))
matrix_dims = np.array(hf.get("matrix_dims"))
hf.close()

#%% PREPARE FEATURES AND LABELS
# Reshape epochs for EEG-Inception
features = features.reshape(
    (features.shape[0], features.shape[1],
     features.shape[2], 1)
)


# One hot encoding of labels
def one_hot_labels(caategorical_labels):
    enc = OneHotEncoder(handle_unknown='ignore')
    on_hot_labels = enc.fit_transform(
        caategorical_labels.reshape(-1, 1)).toarray()
    return on_hot_labels


train_erp_labels = one_hot_labels(erp_labels)

#%%  TRAINING
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Create model
model = EEGInception(
    input_time=1000, fs=128, ncha=8, filters_per_branch=8,
    scales_time=(500, 250, 125), dropout_rate=0.25,
    activation='elu', n_classes=2, learning_rate=0.001)

# Print model summary
model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0.0001,
    mode='min', patience=10, verbose=1,
    restore_best_weights=True)

# Fit model
fit_hist = model.fit(features,
                     erp_labels,
                     epochs=500,
                     batch_size=1024,
                     validation_split=0.2,
                     callbacks=[early_stopping])

# Save
model.save('model')


