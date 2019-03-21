import random
from collections import defaultdict, OrderedDict

from config import GLOBAL_MAX_LEN, N_LABELS
from time import time
__author__ = "Yinchong Yang"
__copyright__ = "Siemens AG, 2017"
__licencse__ = "MIT"
__version__ = "0.1"
import random
import tqdm
"""
MIT License
Copyright (c) 2017 Siemens AG
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import numpy as np
import pickle
import datetime

import cv2
import keras
from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout, Masking, BatchNormalization
from keras.models import Model
from keras.optimizers import *
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Custom Functions -----------------------------------------------------------------------------------------------------
from TTRNN import TT_GRU, TT_LSTM


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,npy_path, labels_file, n_labels=N_LABELS, max_len=GLOBAL_MAX_LEN, batch_size=20, shuffle=True):
        self.shuffle = shuffle
        self.npy_path = npy_path
        self.labels_file = labels_file
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_labels = n_labels

        with open(labels_file, 'r') as f:
            data = [line.rstrip() for line in f.readlines()]
        # Y = np.array([int(line.split()[1]) for line in data], dtype='int8')

        names = [line.split()[0] for line in data]
        labels = [int(line.split()[1]) for line in data]

        N = len(data)
        self.y_dict = OrderedDict()
        for i in range(N):
            # if names[i] not in y_dict:
                # continue
            if names[i] in self.y_dict:
                self.y_dict[names[i]][labels[i]] = 1
            else:
                self.y_dict[names[i]] = np.zeros(n_labels)
                self.y_dict[names[i]][labels[i]] = 1

        self.clip_names = list(self.y_dict.keys())#[:15]

        # X = []
        # y_dict = OrderedDict()
        # # y_dict = defaultdict(lambda: np.zeros(n_labels))
        # for i in range(N):
        #     if names[i] not in names_set:
        #         continue
        #     print(names[i])
        #     if names[i] in y_dict:
        #         y_dict[names[i]][labels[i]] = 1
        #     else:
        #         y_dict[names[i]] = np.zeros(n_labels)
        #         y_dict[names[i]][labels[i]] = 1

        #         frames = np.load(npy_path + '/' + names[i] + '.npy')
        #         if len(frames) > 800:
        #             frames = frames[::2]
        #         frames = frames.reshape(frames.shape[0], -1) # of shape (nb_frames, 240*320*3)
        #         frames = (frames - 128).astype('int8')   # this_clip.mean()
        #         X.append(pad_sequences([frames], maxlen=max_len, truncating='post', dtype='int8')[0])
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.clip_names) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        names = self.clip_names[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(names)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.clip_names = list(self.y_dict.keys())
        if self.shuffle == True:
            np.random.shuffle(self.clip_names)

    def __data_generation(self, names):

        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        X = []
        for name in names:
            frames = np.load(self.npy_path + '/' + name + '.npy')
            if len(frames) > 800:
                frames = frames[::2]
            frames = frames.reshape(frames.shape[0], -1) # of shape (nb_frames, 240*320*3)
            frames = (frames - 128).astype('int8')   # this_clip.mean()
            X.append(pad_sequences([frames], maxlen=self.max_len, truncating='post', dtype='int8')[0])

        X = np.array(X)
        Y = np.array([self.y_dict[name] for name in names])
        return X, Y
        # # Generate data
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #     X[i,] = np.load('data/' + ID + '.npy')

        #     # Store class
        #     y[i] = self.labels[ID]

        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
class ShotDataset(keras.utils.Sequence):
    def __init__(self, train_list_path, data_dir, resize_shape=(112, 112), batch_size=1):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.resize_shape = resize_shape
        with open(train_list_path, 'r') as train_list_f:
            self.train_strings = [i.rstrip().split("DeepSBD/")[-1] for i in train_list_f.readlines()]
            np.random.shuffle(self.train_strings)
        self.idxs = list(range(len(self.train_strings)))
        np.random.shuffle(self.idxs)

    def __len__(self):
        return len(self.train_strings) // self.batch_size


    def __data_generation(self, idxs):

        X = []
        Y = []
        Y = np.zeros((len(idxs), 3), dtype=np.float32)
        for k, idx in enumerate(idxs):
            path, _, label = self.train_strings[idx].split()
            label = int(label)
            # print(self.data_dir, path)
            pics = [i for i in os.listdir(self.data_dir + '/' + path) if i.endswith('.jpg')]
            # t1 = time()
            seq = []
            for pic_name in pics:
                # t1 = time()
                img = cv2.imread(self.data_dir + '/' + path + '/' + pic_name, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # print(time() - t1)
                img = cv2.resize(img, self.resize_shape)

                # imgs.append(np.array([b, g, r]))
                seq.append((img - 128) / 255)
            # qs = [i[0] for i in sorted(list(zip(seq, pics)), key=lambda x: x[1])]
            # qs = np.array(qs)
            seq = np.array(seq)
            seq = seq[np.argsort(pics)]
            # print((qs - seq).max())
            seq = seq.reshape(16, -1)
            # seq = np.swapaxes(seq, 0, 2)
            # print(seq.dtype)
            seq = seq.astype(np.float32)
            X.append(seq)
            Y[k][label] = 1
            # y = np.zeros(3)
            # y[label] = 1
            # Y.append(y)

        X = np.array(X)
        # Y = np.array(Y)
        # print(X.shape)
        return X, Y
        # return seq, np.float32(label)
        return seq, label


    def __getitem__(self, idx):

        idxs = self.idxs[idx*self.batch_size:(idx+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(idxs)

        return X, y


        # path, _, label = self.train_strings[idx].split()
        # label = int(label)
        # pics = [i for i in os.listdir(self.data_dir + '/' + path) if i.endswith('.jpg')]
        # # t1 = time()
        # seq = []
        # for pic_name in pics:
        #     # t1 = time()
        #     img = cv2.imread(self.data_dir + '/' + path + '/' + pic_name, cv2.IMREAD_COLOR)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     # print(time() - t1)
        #     img = cv2.resize(img, self.resize_shape)

        #     # imgs.append(np.array([b, g, r]))
        #     seq.append(img)
        # seq = [i[0] for i in sorted(list(zip(seq, pics)), key=lambda x: x[1])]
        # seq = np.array(seq)
        # # print(seq.dtype)
        # seq = seq.astype(np.float32)
        # # return seq, np.float32(label)
        # return seq, label


class DataGeneratorFromMP4(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,npy_path, labels_file, n_labels=N_LABELS, max_len=GLOBAL_MAX_LEN, batch_size=20, shuffle=True):
        self.shuffle = shuffle
        self.npy_path = npy_path
        self.labels_file = labels_file
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_labels = n_labels

        with open(labels_file, 'r') as f:
            data = [line.rstrip() for line in f.readlines()]
        # Y = np.array([int(line.split()[1]) for line in data], dtype='int8')

        names = [line.split()[0] for line in data]
        labels = [int(line.split()[1]) for line in data]

        N = len(data)
        self.y_dict = OrderedDict()
        for i in range(N):
            # if names[i] not in y_dict:
                # continue
            if names[i] in self.y_dict:
                self.y_dict[names[i]][labels[i]] = 1
            else:
                self.y_dict[names[i]] = np.zeros(n_labels)
                self.y_dict[names[i]][labels[i]] = 1

        self.clip_names = list(self.y_dict.keys())#[:15]

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.clip_names) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        names = self.clip_names[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(names)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.clip_names = list(self.y_dict.keys())
        if self.shuffle == True:
            np.random.shuffle(self.clip_names)

    def __data_generation(self, names):




        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = []
        for name in names:
            clip_path = self.npy_path + '/' + name + '.mp4'

            frames = []                     
            vidcap = cv2.VideoCapture(clip_path)
            success,image = vidcap.read()
            count = 0             
            success = True        
            while success:                
                frames.append(image)  
                #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
                success,image = vidcap.read()
                #print ('Read a new frame: ', success)
                count += 1            
            frames = np.array(frames)


            if len(frames) > 800:
                frames = frames[::2]
            frames = frames.reshape(frames.shape[0], -1) # of shape (nb_frames, 240*320*3)
            frames = (frames - 128).astype('int8')   # this_clip.mean()
            X.append(pad_sequences([frames], maxlen=self.max_len, truncating='post', dtype='int8')[0])

        X = np.array(X)
        Y = np.array([self.y_dict[name] for name in names])
        return X, Y
     

def load_npy_data(npy_path, labels_file, n_labels=N_LABELS, max_len=GLOBAL_MAX_LEN, max_examples='ALL'):
    with open(labels_file, 'r') as f:
        data = [line.rstrip() for line in f.readlines()]
    # Y = np.array([int(line.split()[1]) for line in data], dtype='int8')

    names = [line.split()[0] for line in data]
    labels = [int(line.split()[1]) for line in data]
    N = len(data)

    names_set = set(names)
    if max_examples != 'ALL':
        names_set = set(random.sample(names_set, max_examples))
    # X = np.zeros((N, max_len, 234*100*3), dtype='int8')
    # Y = np.zeros((N, n_labels), dtype='int8')


    X = []
    y_dict = OrderedDict()
    # y_dict = defaultdict(lambda: np.zeros(n_labels))
    for i in range(N):
        if names[i] not in names_set:
            continue
        print(names[i])
        if names[i] in y_dict:
            y_dict[names[i]][labels[i]] = 1
        else:
            y_dict[names[i]] = np.zeros(n_labels)
            y_dict[names[i]][labels[i]] = 1

            frames = np.load(npy_path + '/' + names[i] + '.npy')
            if len(frames) > 800:
                frames = frames[::2]
            frames = frames.reshape(frames.shape[0], -1) # of shape (nb_frames, 240*320*3)
            frames = (frames - 128).astype('int8')   # this_clip.mean()
            X.append(pad_sequences([frames], maxlen=max_len, truncating='post', dtype='int8')[0])
    X = np.array(X)
    Y = np.array(list(y_dict.values()))
    return X, Y



# Load the data --------------------------------------------------------------------------------------------------------
np.random.seed(11111986)

# Settings:
model_type = 1
use_TT = 1




classes = ['AnswerPhone', 'DriveCar', 'Eat', 'FightPerson', 'GetOutCar', 'HandShake',
           'HugPerson', 'Kiss', 'Run', 'SitDown', 'SitUp', 'StandUp']


# filter out labels that are not included because their samples are longer than 50 frames:
# tr_sample_filenames = os.listdir(data_path + 'actioncliptrain/')
# tr_sample_filenames.sort()
#
# tr_sample_ids = np.array( [x[-10:-5] for x in tr_sample_filenames] ).astype('int16')

# tr_label_filename = data_path + 'train_labels.txt'
# tr_labels = np.loadtxt(tr_label_filename, dtype='int16')
# with open(tr_label_filename, 'r') as f:
#     tr = [l.rstrip() for l in f.readlines()]
#
#     tr_label_ids = tr_labels[:, 0]
#     tr_labels = tr_labels[:, 1::]
#
#
# # filter out labels that are not included because their samples are longer than 50 frames:
# # te_sample_filenames = os.listdir(data_path + 'actioncliptest/')
# # te_sample_filenames.sort()
# #
# # te_sample_ids = np.array( [x[-10:-5] for x in te_sample_filenames] ).astype('int16')
#
# te_label_filename = data_path + 'test_labels.txt'
# te_labels = np.loadtxt(te_label_filename, dtype='int16')
# te_label_ids = te_labels[:, 0]
# te_labels = te_labels[:, 1::]
#
# n_tr = tr_labels.shape[0]
# n_te = te_labels.shape[0]
#
#
# # X_train, Y_train = load_data(np.arange(0, 128), mode='train')  # small set
# X_train, Y_train = load_data(np.arange(0, n_tr), mode='train')
# # X_test, Y_test = load_data(np.arange(0, 128), mode='test')  # small set
# X_test, Y_test = load_data(np.arange(0, n_te), mode='test')


data_path = '/mnt/ttrnn/Hollywood2/scaled_clips/AVIClips'
train_labels = '/mnt/ttrnn/Hollywood2/train_labels.txt'
test_labels = '/mnt/ttrnn/Hollywood2/test_labels.txt'


data_path = "/media/tva/721af2a0-5850-411f-b27a-515f87736c71/vic/projects/moana/data/DeepSBD"
train_list_path = "/media/tva/721af2a0-5850-411f-b27a-515f87736c71/vic/projects/moana/new_train_list.txt"
test_list_path = "/media/tva/721af2a0-5850-411f-b27a-515f87736c71/vic/projects/moana/new_test_list.txt"
# X_train, Y_train = load_npy_data(data_path, train_labels, n_labels=N_LABELS, max_len=GLOBAL_MAX_LEN, max_examples=300)
# X_test, Y_test = load_npy_data(data_path, test_labels, n_labels=N_LABELS, max_len=GLOBAL_MAX_LEN, max_examples=50)
# print(X_train.size)
# exit()

# Define the model -----------------------------------------------------------------------------------------------------
alpha = 1e-2

tt_input_shape = [10, 18, 13, 30]
tt_input_shape = [7, 16, 12, 28]
tt_output_shape = [4, 4, 4, 4]
tt_ranks = [1, 4, 4, 4, 1]

dropoutRate = .25

input = Input(shape=(16, 112*112*3))
if model_type == 0:
    if use_TT ==0:
        rnn_layer = GRU(np.prod(tt_output_shape),
                        return_sequences=False,
                        dropout=0.25, recurrent_dropout=0.25, activation='tanh')
    else:
        rnn_layer = TT_GRU(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                           tt_ranks=tt_ranks,
                           return_sequences=False,
                           dropout=0.25, recurrent_dropout=0.25, activation='tanh')
else:
    if use_TT ==0:
        rnn_layer = LSTM(np.prod(tt_output_shape),
                         return_sequences=False,
                         dropout=0.25, recurrent_dropout=0.25, activation='tanh')
    else:
        rnn_layer = TT_LSTM(tt_input_shape=tt_input_shape, tt_output_shape=tt_output_shape,
                            tt_ranks=tt_ranks,
                            return_sequences=False,
                            dropout=0.25, recurrent_dropout=0.25, activation='tanh')
h = rnn_layer(input)

output = Dense(units=3, activation='softmax', kernel_regularizer=l2(alpha))(h)
model = Model(input, output)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')
model.load_weights('/mnt/ttrnn/my_model49412.h5')

## ------------------
#dgen = ShotDataset(train_list_path, data_path, batch_size=600  )
dgen = ShotDataset(test_list_path, data_path, batch_size=300  )
from sklearn.metrics import precision_recall_fscore_support
for i in range(500):
    #model.fit_generator(dgen, nb_epoch=1, verbose=1, workers=4, use_multiprocessing=True)
    #model.save('/mnt/ttrnn/my_model%s.h5' % random.randint(1, 100000))
    #continue
   #for X, Y in tqdm.tqdm(dgen):
    _Y_pred = []
    _Y_test = []
    i = 0
    for X_test, Y_test in tqdm.tqdm(dgen):
        t1 = time()
        Y_pred = model.predict(X_test)
        t2 = time() - t1
        print(t2)
        _Y_pred.append(Y_pred)
        _Y_test.append(Y_test)
        #if i == 20:
        #    break
        i+=1

    
    _Y_pred = np.vstack(tuple(_Y_pred)).argmax(axis=1)
    _Y_test = np.vstack(tuple(_Y_test)).argmax(axis=1)
    print(_Y_pred.shape)
    #print(average_precision_score(_Y_test, _Y_pred))
    print(precision_recall_fscore_support(_Y_test, _Y_pred))
    model.fit_generator(dgen, nb_epoch=1, verbose=1, workers=4, use_multiprocessing=True)
    model.save('/mnt/ttrnn/my_model%s.h5' % random.randint(1, 100000))

## ------------------


# lstm = LSTM(np.prod(tt_output_shape),
#                          return_sequences=False,
#                          dropout=0.25, recurrent_dropout=0.25, activation='tanh')

# classic_h = lstm(input)
# classic_output =  Dense(units=12, activation='sigmoid', kernel_regularizer=l2(alpha))(classic_h)
# classic_model = Model(input, classic_output)
# classic_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')

# classic_model.load_weights('/mnt/ttrnn/my_model12061.h5')
model.load_weights('/mnt/ttrnn/my_model68326.h5')
# dgen = DataGeneratorFromMP4('/mnt/scaled-498-list-scenes/498-list-scenes', '/mnt/scaled-498-list-scenes/498-list-scenes/data_labels.txt', batch_size=9  )
# dgen = DataGeneratorFromMP4('/mnt/scaled-498-list-scenes/498-list-scenes', '/mnt/scaled-498-list-scenes/test_ls.txt', batch_size=1  )

represent_mp4 = False
if represent_mp4:
    rnn = rnn_layer(input)

    # index_output = Dense(units=12, activation='sigmoid', kernel_regularizer=l2(alpha))(rnn)
    index_model = Model(input, rnn)
    index_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')
    index_model.layers[0].set_weights( model.layers[0].get_weights())
    index_model.layers[1].set_weights( model.layers[1].get_weights())
    print('layers', model.layers)
    print([[w.shape for w in layer.weights] for layer in model.layers])
    print([[w.shape for w in layer.weights] for layer in index_model.layers])
    model = index_model
    import tqdm
    from scipy.spatial.distance import cosine
    from matplotlib import pyplot as plt
    preds = []
    for X, Y in tqdm.tqdm(dgen):
        # print('%s / %s' % (i, len(dgen)))
    # X, Y = dgen.__iter__().__next__()
        # print(X.shape)
        plt.imshow(X[0].reshape(-1, 100, 234, 3)[-1] + 128)
        plt.show()
        # print(X[0])
        print(Y)
        preds.append(model.predict(X)[0])
        # print('shape', pred.shape)
        print(preds[-1])
    # print(cosine(preds[0], preds[1]))
    print(( (preds[0] - preds[1]) @(preds[0] - preds[1]) ) )

# print([i.shape for i in model.get_weights()])
# print(classic_model.layers[2].weights)

# model.layers[0].set_weights(classic_model.layers[0].get_weights())
# model.layers[2].set_weights(classic_model.layers[2].get_weights())

# print(classic_model.layers)
# print('classic model loaded')
# for i, weights in enumerate():
#     model.layers[i].set_weights(weights)


dgen = DataGenerator(data_path, test_labels, batch_size=9  )
import tqdm
dgen = DataGenerator(data_path, train_labels, batch_size=9  )
# Start training -------------------------------------------------------------------------------------------------------
for l in range(501):
    print('iter ' + str(l))
    
    _Y_pred = []
    _Y_test = []
    i = 0
    for X_test, Y_test in tqdm.tqdm(dgen):
        Y_pred = model.predict(X_test)
        _Y_pred.append(Y_pred)
        _Y_test.append(Y_test)

        if i == 50:
            break
        i+=1


    _Y_pred = np.vstack(tuple(_Y_pred))
    _Y_test = np.vstack(tuple(_Y_test))
    _Y_pred[_Y_pred >=1] = 1
    _Y_test[_Y_test >=1] = 1
    print(average_precision_score(_Y_test, _Y_pred))
    # print(_Y_pred.shape)
        # print(Y_test.__repr__())
        # test_res = average_precision_score(Y_test, Y_pred)
        # print(Y_test[0])
        # print(Y_pred.__repr__())
    # print(X_train.shape, Y_train.shape)
    # model.fit(X_train, Y_train, nb_epoch=1, batch_size=2, verbose=1, validation_split=.0)
    model.fit_generator(dgen, nb_epoch=1, verbose=1, workers=3, use_multiprocessing=True)# validation_split=.0)
    moodel.predict_g
    model.save('/mnt/ttrnn/my_model%s.h5' % random.randint(1, 100000))

        # if l % 10 == 0:
        #     Y_hat = model.predict(X_train)
        #     Y_pred = model.predict(X_test)
        #     train_res = average_precision_score(Y_train, Y_hat)
        #     test_res = average_precision_score(Y_test, Y_pred)

        #     print('Training: ')
        #     print(train_res)
        #     print('Test: ')
        #     print(test_res)
