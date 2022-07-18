import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from scipy.fftpack import fft
import numpy as np
import csv
from tensorflow import keras
# from tensorflow.keras.utils import np_utils
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# from vis.visualization import visualize_saliency

np.set_printoptions(threshold=sys.maxsize)

# Read trial timestamps
# set a direct path to Filedata.csv

filedata = np.ndarray((16, 5), dtype=int)
with open("C:/project lab/code/Filedata.csv", newline='') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    i_row = 0
    for row in r:
        for i_col in range(filedata.shape[1]):
            filedata[i_row, i_col] = row[i_col]
        i_row = i_row + 1

i_subject = 3

# set the direct path
ifn_config = "C:/project lab/code/ECoG OKITI/Configs/S" + str(i_subject).zfill(2) + ".csv"
config = np.ndarray((filedata[i_subject, 1], 5), dtype=int)
with open(ifn_config, newline='') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    i_row = 0
    for row in r:
        for i_col in range(config.shape[1]):
            config[i_row, i_col] = row[i_col]
        i_row = i_row + 1

# Read ECoG/EEG data

window_length = int(filedata[i_subject, 4] / 1)
offset = int(filedata[i_subject, 1] / 1)
n_chunk = 9
i_length = 38
bigdata = np.zeros((int(filedata[i_subject, 0] * n_chunk * i_length), window_length, filedata[i_subject, 1]),
                   dtype=float)
label = np.zeros((int(filedata[i_subject, 0] * n_chunk * i_length),), dtype=int)

for i_file in range(filedata[i_subject, 0]):

    # set a direct path
    ifn = "C:/project lab/code/ECoG OKITI/S" + str(i_subject).zfill(2) + "/S" + str(i_subject).zfill(2) + "R" + str(
        i_file)
    ifn_csv = ifn + ".csv"
    ifn_dat = ifn + ".dat"

    with open(ifn_csv, newline='') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        n_rows = 0
        for row in r:
            n_rows = n_rows + 1

    indices = np.ndarray((n_rows,), dtype=int)
    with open(ifn_csv, newline='') as csvfile:
        r = csv.reader(csvfile, delimiter=',')
        i_row = 0
        for row in r:
            indices[i_row] = row[0]
            i_row = i_row + 1

    filelength = int(os.path.getsize(ifn_dat) / (filedata[i_subject, 1] * (filedata[i_subject, 2] / 8)))
    indices_to_plot = np.zeros((filelength,))
    indices_to_plot[indices] = 1

    gates = np.zeros((filelength,))
    last_value = 0
    for i in range(filelength):
        gates[i] = last_value
        if indices_to_plot[i] == 1:
            if last_value == 0:
                last_value = 1
            else:
                last_value = 0
        gates[i] = last_value

    datatype = np.uint16
    if filedata[i_subject, 2] == 16 and filedata[i_subject, 3] == 1:
        datatype = np.int16
    if filedata[i_subject, 2] == 32 and filedata[i_subject, 3] == 0:
        datatype = np.uint32
    if filedata[i_subject, 2] == 32 and filedata[i_subject, 3] == 1:
        datatype = np.int32

    data = np.fromfile(ifn_dat, dtype=datatype)
    data = data.reshape((filelength, filedata[i_subject, 1])).astype(float)

    for i in range(i_length):
        for j in range(n_chunk):
            data_index = i_file * i_length * n_chunk + i * n_chunk + j
            bigdata[data_index, :, :] = data[
                                        indices[i] + (j + 1) * offset: indices[i] + (j + 1) * offset + window_length,
                                        :].reshape(1, window_length, filedata[i_subject, 1])
            label[data_index] = gates[indices[i] + 10] * (i_file + 1)

# Data preprocessing

# Delete channels contaminated with artefacts and EEG channels
notfeasible = np.where(config[:, 2] != 0)[0]
bigdata = np.delete(bigdata, notfeasible, axis=2)
bigdata = np.abs(fft(bigdata, axis=1))

# Filter 50 Hz harmonics
bigdata[:, int((window_length * 50 / filedata[i_subject, 4])), :] = 0
bigdata[:, int((window_length * 100 / filedata[i_subject, 4])), :] = 0
bigdata[:, int((window_length * 150 / filedata[i_subject, 4])), :] = 0

# Restrict data to the (0, 200] Hz range
bigdata = np.expand_dims(bigdata[:, 1:int((window_length * 200 / filedata[i_subject, 4])), :], axis=3)

# Standardize data
bigdata = (bigdata - np.mean(bigdata)) / (np.std(bigdata))

data_to_draw = bigdata.reshape((n_chunk, i_length, filedata[i_subject, 0], bigdata.shape[1], bigdata.shape[2]))
active_indices = 2 * np.arange(int(np.floor(i_length / 2)))
active_avg = np.average(data_to_draw[:, active_indices, :, :, :], axis=1)
passive_avg = np.squeeze(np.average(np.average(data_to_draw[:, active_indices + 1, :, :, :], axis=1), axis=1))
print(active_avg.shape)

shuffle_indices = np.arange(bigdata.shape[0])
np.random.shuffle(shuffle_indices)
bigdata = bigdata[shuffle_indices, :, :]
label = label[shuffle_indices]
label_cat = keras.utils.to_categorical(label, num_classes=filedata[i_subject, 0] + 1)
training_rate = 0.8

# 2D convnet
input_layer = keras.Input(shape=bigdata[0].shape, name="input_layer")
c1 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='valid')(input_layer)
do1 = keras.layers.Dropout(0.5)(c1)
bn1 = keras.layers.BatchNormalization()(do1)
c2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='valid')(bn1)
do2 = keras.layers.Dropout(0.5)(c2)
bn2 = keras.layers.BatchNormalization()(do2)
c3 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(bn2)
do3 = keras.layers.Dropout(0.5)(c3)
bn3 = keras.layers.BatchNormalization()(do3)
c4 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(bn3)
do4 = keras.layers.Dropout(0.5)(c4)
bn4 = keras.layers.BatchNormalization()(do4)

flatten = keras.layers.Flatten()(bn4)
dense1 = keras.layers.Dense(128, activation="relu")(flatten)
do0 = keras.layers.Dropout(0.5)(dense1)
bn0 = keras.layers.BatchNormalization()(do0)

out = keras.layers.Dense(label_cat.shape[1], activation="softmax", name="out")(bn0)
model = keras.Model(inputs=[input_layer], outputs=[out])
optimizer = keras.optimizers.RMSprop(lr=0.1, decay=0.001)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
callbacks = [keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=10),
             keras.callbacks.ModelCheckpoint(filepath='C:/project lab/code/best_model.h5', monitor='val_accuracy', save_best_only=True)]

# Training and test databases
train_input = bigdata[0:int(bigdata.shape[0] * training_rate), :, :]
test_input = bigdata[int(bigdata.shape[0] * training_rate):bigdata.shape[0], :, :]

# # Class weights for balanced training
# cw = np.sum(label_cat[0:int(bigdata.shape[0]*training_rate), :], axis=0)
# cw = cw[0]/cw
# print(cw)

# cw_dict = {}
# for i in range(cw.shape[0]):
#     cw_dict[i] = cw[i]
# print(cw_dict)

# Convnet training
model.fit(x=train_input, y={"out": label_cat[0:int(bigdata.shape[0] * training_rate), :]}, callbacks=callbacks,
          epochs=100, verbose=2,
          validation_split=0.25, shuffle=True, batch_size=5)
# move into fit function later "class_weight=cw_dict"


# We evaluate the best model
model = keras.models.load_model('C:/project lab/code/best_model.h5')

# Prediction+
label_predicted = model.predict(test_input)
label_predicted0 = np.zeros(bigdata.shape[0] - int(bigdata.shape[0] * training_rate))

# Confusion matrix
for i in range(bigdata.shape[0] - int(bigdata.shape[0] * training_rate)):
    label_predicted0[i] = np.argmax(label_predicted[i, :])
cm = confusion_matrix(label[int(bigdata.shape[0] * training_rate):bigdata.shape[0]], label_predicted0)
print(cm)

# Evaluate results
test_label = label_cat[int(bigdata.shape[0] * training_rate):bigdata.shape[0]]
score = model.evaluate(test_input, {"out": test_label})
print(score)

# # Produce saliency maps from test samples
# test_reconstructed = np.zeros((test_input.shape[0], test_input.shape[1], test_input.shape[2]))
# for i_sample in range(test_label.shape[0]):
#     class_id = np.argmax(test_label[i_sample, :])
#     print(class_id)
#     test_reconstructed[i_sample, :, :] = visualize_saliency(model=model, layer_idx=-1, filter_indices=class_id,
#                                                             seed_input=test_input[i_sample, :, :])

# # Fit a small dense network to the best features
# n_greatest_vec = [1, 10, 100, 1000]
# for n_greatest in n_greatest_vec:
#     sal_imp = np.zeros((test_input.shape[1], test_input.shape[2]))

#     # Best features of each category
#     for i in range(test_label.shape[1]):
#         x = np.where(test_label[:, i] == 1)[0]
#         y0 = np.squeeze(np.average(test_input[x, :, :], axis=0))
#         y = np.squeeze(np.average(test_reconstructed[x, :, :], axis=0))

#         greatest = np.sort(y, axis=None)
#         greatest = greatest[greatest.shape[0] - n_greatest:greatest.shape[0]]

#         for val in greatest:
#             sal_imp = sal_imp + np.where(y == val, 1, 0)

#         fig, ax = plt.subplots()
#         im = ax.imshow(y)
#         cbar = ax.figure.colorbar(im, ax=ax)
#         plt.show()

#     fig, ax = plt.subplots()
#     im = ax.imshow(sal_imp)
#     cbar = ax.figure.colorbar(im, ax=ax)
#     plt.show()

#     # Extract the best features from each input sample, create train and test databases
#     sal_imp = sal_imp.reshape((sal_imp.shape[0] * sal_imp.shape[1]))
#     mask = np.where(sal_imp > 0)
#     train_pruned = train_input.reshape((train_input.shape[0], train_input.shape[1] * train_input.shape[2]))
#     train_pruned = train_pruned[:, mask]
#     test_pruned = test_input.reshape((test_input.shape[0], test_input.shape[1] * test_input.shape[2]))
#     test_pruned = test_pruned[:, mask]

#     # Description of dense layer
#     m2_input_layer = keras.Input(shape=train_pruned[0].shape, name="m2_input_layer")
#     m2_flatten = keras.layers.Flatten()(m2_input_layer)
#     m2_dense0 = keras.layers.Dense(256, activation="relu")(m2_flatten)
#     m2_do0 = keras.layers.Dropout(0.5)(m2_dense0)
#     m2_bn0 = keras.layers.BatchNormalization()(m2_do0)
#     m2_out = keras.layers.Dense(label_cat.shape[1], activation="softmax", name="m2_out")(m2_bn0)

#     model2 = keras.Model(inputs=[m2_input_layer], outputs=[m2_out])
#     model2.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
#     callbacks = [keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10),
#                  keras.callbacks.ModelCheckpoint(filepath='best_model2.h5', monitor='val_acc', save_best_only=True)]

    # # Train model
    # model2.fit(train_pruned, {"m2_out": label_cat[0:int(bigdata.shape[0]*training_rate), :]}, callbacks=callbacks, epochs=100, verbose=2,
    #           validation_split=0.25, shuffle=True, class_weight=cw, batch_size=5)
    
    # # We use the best model for prediction
    # model2 = keras.models.load_model('best_model2.h5')
    
    # # Prediction and evaluation
    # label_predicted = model2.predict(test_pruned)
    # label_predicted0 = np.zeros(bigdata.shape[0] - int(bigdata.shape[0] * training_rate))
    # for i in range(bigdata.shape[0] - int(bigdata.shape[0] * training_rate)):
    #     label_predicted0[i] = np.argmax(label_predicted[i, :])
    # cm = confusion_matrix(label[int(bigdata.shape[0]*training_rate):bigdata.shape[0]], label_predicted0)
    # print(cm)
    # test_label = label_cat[int(bigdata.shape[0]*training_rate):bigdata.shape[0]]
    # score = model2.evaluate(test_pruned, {"m2_out": test_label})
    # print(score)
