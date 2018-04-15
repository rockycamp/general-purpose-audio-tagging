import os, shutil
import numpy as np
import pandas as pd
import librosa
import scipy
import fileio
import aifc
import audio_processor as ap

from sklearn.cross_validation import StratifiedKFold

from my_classes import Config, DataGenerator
from create_dirs import create_dirs

from keras import backend as K
from keras import losses, layers, models, optimizers
from keras.models import Model
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Activation, BatchNormalization, concatenate, Convolution1D,
                          Convolution2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling1D, GlobalAveragePooling2D,
                          GlobalMaxPool1D, GlobalMaxPool2D, Input,
                          MaxPool1D, MaxPool2D, MaxPooling2D, ZeroPadding2D)

from keras.layers.advanced_activations import ELU
from keras.utils import to_categorical
from keras.utils.data_utils import get_file

# debug switch
DEBUG_MODEL = False

# audio normalisation
def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5

def prepare_data(df, config, data_dir):
    X = np.empty(shape=(df.shape[0], config.dim[0], config.dim[1], 1))
    input_length = config.audio_length
    for i, fname in enumerate(df.index):
        if i%500 == 0:
            print(fname, " (",i,"/",len(df.index))
        file_path = data_dir + fname
        data, _ = librosa.core.load(file_path, sr=config.sampling_rate, res_type="kaiser_fast")

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
        data = np.expand_dims(data, axis=-1)
        X[i,] = data
    return X

# dumming 1d model
def get_1d_dummy_model(config):

    nclass = config.n_classes
    input_length = config.audio_length

    inp = Input(shape=(input_length,1))
    x = GlobalMaxPool1D()(inp)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

    return model


# Conv1D model
def get_1d_conv_model(config):

    nclass = config.n_classes
    input_length = config.audio_length

    inp = Input(shape=(input_length,1))
    x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
    x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
    x = MaxPool1D(16)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    x = MaxPool1D(4)(x)
    x = Dropout(rate=0.1)(x)

    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(64, activation=relu)(x)
    x = Dense(1028, activation=relu)(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

def get_2d_dummy_model(config):

    nclass = config.n_classes

    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = GlobalMaxPool2D()(inp)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


def get_2d_conv_model(config):

    nclass = config.n_classes

    inp = Input(shape=(config.dim[0],config.dim[1],1))
    x = Convolution2D(32, (4,10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4,10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model


# Choi model
def get_choi_model(config):

    nclass = config.n_classes
    # Only tf dimension ordering
    channel_axis = 3
    freq_axis = 1
    time_axis = 2

    inp = Input(shape=(config.dim[0], config.dim[1], 1))

    # Input block
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(inp)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), dim_ordering="th", name='pool1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), dim_ordering="th", name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), dim_ordering="th", name='pool3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), dim_ordering="th", name='pool4')(x)

    # Conv block 5
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), dim_ordering="th", name='pool5')(x)

    # Output
    x = Flatten()(x)
    #x = Dense(50, activation='relu', name='hidden1')(x)
    out = Dense(nclass, activation='softmax', name='output')(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.RMSprop(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])

    return model


def conv1d_main():

    train = pd.read_csv("../../.kaggle/competitions/freesound-audio-tagging/train.csv")
    test = pd.read_csv("../../.kaggle/competitions/freesound-audio-tagging/sample_submission.csv")

    # convert raw labels to indices
    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname", inplace=True)
    test.set_index("fname", inplace=True)
    train["label_idx"] = train.label.apply(lambda x: label_idx[x])
    if DEBUG_MODEL:
        train = train[:1000]
        test = test[:1000]

    config = Config(sampling_rate=16000, audio_duration=2, n_folds=10, learning_rate=0.001)
    if DEBUG_MODEL:
        config = Config(sampling_rate=100, audio_duration=1, n_folds=3, max_epochs=1)

    # use from sklearn.cross_validation.StratifiedKFold for splitting the trainig data into 10 folds.
    PREDICTION_FOLDER = "predictions_1d_conv"
    #if not os.path.exists('/home/david/.kaggle/competitions/freesound-prediction-file'):
    #    os.mkdir('/home/david/.kaggle/competitions/freesound-prediction-file')
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
    if os.path.exists('logs/' + PREDICTION_FOLDER):
        shutil.rmtree('logs/' + PREDICTION_FOLDER)

    skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)

    for i, (train_split, val_split) in enumerate(skf):
        train_set = train.iloc[train_split]
        val_set = train.iloc[val_split]
        checkpoint = ModelCheckpoint('conv1d_best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%d'%i, write_graph=True)
        callbacks_list = [checkpoint, early, tb]
        print("Fold: ", i)
        print("#"*50)
        if not DEBUG_MODEL:
            model = get_1d_conv_model(config)
        else:
            model = get_1d_dummy_model(config)

        train_generator = DataGenerator(config, '../../.kaggle/competitions/freesound-audio-tagging/audio_train/', train_set.index,
                                        train_set.label_idx, batch_size=64,
                                        preprocessing_fn=audio_norm)
        val_generator = DataGenerator(config, '../../.kaggle/competitions/freesound-audio-tagging/audio_train/', val_set.index,
                                      val_set.label_idx, batch_size=64,
                                      preprocessing_fn=audio_norm)

        history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
                                      epochs=config.max_epochs, use_multiprocessing=True, workers=6, max_queue_size=20)

        model.load_weights('conv1d_best_%d.h5'%i)

        # Save train predictions
        train_generator = DataGenerator(config,
                                        '../../.kaggle/competitions/freesound-audio-tagging/audio_train/',
                                        train.index,
                                        batch_size=128,
                                        preprocessing_fn=audio_norm)
        predictions = model.predict_generator(train_generator,
                                              use_multiprocessing=True,
                                              workers=6,
                                              max_queue_size=20,
                                              verbose=1)
        np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy"%i, predictions)

        # Save test predictions
        test_generator = DataGenerator(config,
                                       '../../.kaggle/competitions/freesound-audio-tagging/audio_test/',
                                       test.index,
                                       batch_size=128,
                                       preprocessing_fn=audio_norm)
        predictions = model.predict_generator(test_generator,
                                              use_multiprocessing=True,
                                              workers=6,
                                              max_queue_size=20,
                                              verbose=1)
        np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)

        # Make a submission file
        top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]] #3
        predicted_labels = [' '.join(list(x)) for x in top_3]
        test['label'] = predicted_labels
        test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv"%i)

    # Average the predictions of the 10 (two in debug) folds
    pred_list = []
    for i in range(config.n_folds):
        pred_list.append(np.load(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i))
    prediction = np.ones_like(pred_list[0])
    for pred in pred_list:
        prediction = prediction*pred
    prediction = prediction**(1./len(pred_list))

    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]] #3
    predicted_labels = [' '.join(list(x)) for x in top_3]
    print(len(predicted_labels))

    if DEBUG_MODEL:
        test = pd.read_csv('../../.kaggle/competitions/freesound-audio-tagging/sample_submission.csv')
        test = test[:1000]
        test['label'] = predicted_labels
        test[['fname', 'label']].to_csv("1d_conv_ensembled_submission.csv", index=False)
    else:
        test = pd.read_csv('../../.kaggle/competitions/freesound-audio-tagging/sample_submission.csv')
        test['label'] = predicted_labels
        test[['fname', 'label']].to_csv("1d_conv_ensembled_submission.csv", index=False)

    print(model.summary())


def conv2d_main():
    train = pd.read_csv("../../.kaggle/competitions/freesound-audio-tagging/train.csv")
    test = pd.read_csv("../../.kaggle/competitions/freesound-audio-tagging/sample_submission.csv")

    # convert raw labels to indices
    LABELS = list(train.label.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    train.set_index("fname", inplace=True)
    test.set_index("fname", inplace=True)
    train["label_idx"] = train.label.apply(lambda x: label_idx[x])
    if DEBUG_MODEL:
        train = train[:1000]
        test = test[:1000]

    config = Config(sampling_rate=44100, audio_duration=3, n_folds=10,
                    learning_rate=0.0005, use_mfcc=True, n_mfcc=96)
    if DEBUG_MODEL:
        config = Config(sampling_rate=44100, audio_duration=2, n_folds=3,
                        max_epochs=1, use_mfcc=True, n_mfcc=40)

    if not DEBUG_MODEL:
        model = get_choi_model(config)
        print(model.summary())

    X_train = prepare_data(train, config, '../../.kaggle/competitions/freesound-audio-tagging/audio_train/')
    X_test = prepare_data(test, config, '../../.kaggle/competitions/freesound-audio-tagging/audio_test/')
    y_train = to_categorical(train.label_idx, num_classes=config.n_classes)

    # Normalisation
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std

    PREDICTION_FOLDER = "predictions_2d_conv"
    if not os.path.exists(PREDICTION_FOLDER):
        os.mkdir(PREDICTION_FOLDER)
    if os.path.exists('logs/' + PREDICTION_FOLDER):
        shutil.rmtree('logs/' + PREDICTION_FOLDER)

    skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)

    for i, (train_split, val_split) in enumerate(skf):
        K.clear_session()
        X, y, X_val, y_val = X_train[train_split], y_train[train_split], X_train[val_split], y_train[val_split]
        checkpoint = ModelCheckpoint('conv2d_best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%i'%i, write_graph=True)
        callbacks_list = [checkpoint, early, tb]
        print("#"*50)
        print("Fold: ", i)
        #model = get_2d_conv_model(config)
        if not DEBUG_MODEL:
            model = get_choi_model(config)
        else:
            model = get_2d_dummy_model(config)

        history = model.fit(X, y, validation_data=(X_val, y_val), callbacks=callbacks_list,
                            batch_size=64, epochs=config.max_epochs)
        model.load_weights('conv2d_best_%d.h5'%i)

        # Save train predictions
        predictions = model.predict(X_train, batch_size=64, verbose=1)
        np.save(PREDICTION_FOLDER + "/train_predictions_%d.npy"%i, predictions)

        # Save test predictions
        predictions = model.predict(X_test, batch_size=64, verbose=1)
        np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)

        # Make a submission file
        top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
        predicted_labels = [' '.join(list(x)) for x in top_3]
        test['label'] = predicted_labels
        test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv"%i)

    pred_list = []
    for i in range(config.n_folds):
        pred_list.append(np.load(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i))
    prediction = np.ones_like(pred_list[0])
    for pred in pred_list:
        prediction = prediction*pred
    prediction = prediction**(1./len(pred_list))
    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test = pd.read_csv('../../.kaggle/competitions/freesound-audio-tagging/sample_submission.csv')
    if DEBUG_MODEL:
        test = test[:1000]
    test['label'] = predicted_labels
    test[['fname', 'label']].to_csv("2d_conv_ensembled_submission.csv", index=False)

def ensemble_predictions():
    pred_list = []
    for i in range(10):
        pred_list.append(np.load("~/.kaggle/competitions/freesound-prediction-data-2d-conv-reduced-lr/test_predictions_%d.npy"%i))
    for i in range(10):
        pred_list.append(np.load("~/.kaggle/competitions/freesound-prediction-file/test_predictions_%d.npy"%i))
    prediction = np.ones_like(pred_list[0])
    for pred in pred_list:
        prediction = prediction*pred
    prediction = prediction**(1./len(pred_list))
    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-prediction, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test = pd.read_csv('~/.kaggle/competitions/freesound-audio-tagging/sample_submission.csv')
    test['label'] = predicted_labels
    test[['fname', 'label']].to_csv("1d_2d_ensembled_submission.csv", index=False)

if __name__ == '__main__':
    #conv1d_main()
    conv2d_main()
    #ensemble_predictions()
