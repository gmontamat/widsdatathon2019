import cv2
import numpy as np
import os
import pandas as pd
import pickle


data_path = os.path.join(os.getcwd(), '..', 'input')


height = 256
width = 256


with open(os.path.join(data_path, 'train_images_256x256.pkl'), 'rb') as fin:
    train_images = pickle.load(fin)
with open(os.path.join(data_path, 'train_responses.pkl'), 'rb') as fin:
    train_responses = pickle.load(fin)
with open(os.path.join(data_path, 'train_scores.pkl'), 'rb') as fin:
    train_scores = pickle.load(fin)
with open(os.path.join(data_path, 'augmented_images_256x256.pkl'), 'rb') as fin:
    augmented_images = pickle.load(fin)
with open(os.path.join(data_path, 'augmented_responses.pkl'), 'rb') as fin:
    augmented_responses = pickle.load(fin)
with open(os.path.join(data_path, 'augmented_scores.pkl'), 'rb') as fin:
    augmented_scores = pickle.load(fin)


train_images_90 = train_images[train_scores.squeeze() > .9, :, :, :]
train_responses_90 = train_responses[train_scores.squeeze() > .9, :]
train_images_70 = train_images[(train_scores.squeeze() < .9) & (train_scores.squeeze() > .7), :, :, :]
train_responses_70 = train_responses[(train_scores.squeeze() < .9) & (train_scores.squeeze() > .7), :]
train_images_50 = train_images[train_scores.squeeze() < .7, :, :, :]
train_responses_50 = train_responses[train_scores.squeeze() < .7, :]


del train_images
del train_responses
del train_scores


augmented_images_90 = augmented_images[augmented_scores.squeeze() > .9, :, :, :]
augmented_responses_90 = augmented_responses[augmented_scores.squeeze() > .9, :]
augmented_images_70 = augmented_images[(augmented_scores.squeeze() < .9) & (augmented_scores.squeeze() > .7), :, :, :]
augmented_responses_70 = augmented_responses[(augmented_scores.squeeze() < .9) & (augmented_scores.squeeze() > .7), :]
augmented_images_50 = augmented_images[augmented_scores.squeeze() < .7, :, :, :]
augmented_responses_50 = augmented_responses[augmented_scores.squeeze() < .7, :]


del augmented_images
del augmented_responses
del augmented_scores


train_images_90 = train_images_90 * 2. / 255. - 1.
train_images_70 = train_images_70 * 2. / 255. - 1.
train_images_50 = train_images_50 * 2. / 255. - 1.
augmented_images_90 = augmented_images_90 * 2. / 255. - 1.
augmented_images_70 = augmented_images_70 * 2. / 255. - 1.
augmented_images_50 = augmented_images_50 * 2. / 255. - 1.


from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


class roc_callback(Callback):
    """
    Define a callback which returns train ROC AUC after
    each epoch and stops early when validation AUC
    doesn't improve.
    """

    def __init__(self, training_data, validation_data, patience=3, baseline=0.998):
        super(Callback, self).__init__()
        self.best_roc_val = 0.
        self.consecutive_worse = 0
        self.patience = patience
        self.baseline = baseline
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: {} - roc-auc-val: {}'.format(round(roc, 6), round(roc_val, 6)), end=80 * ' ' + '\n')
        if roc_val > self.best_roc_val:
            self.consecutive_worse = 0
            self.best_roc_val = roc_val
            self.model.save_weights('best.h5')
        else:
            self.consecutive_worse += 1
            if self.consecutive_worse >= self.patience:
                if self.best_roc_val > self.baseline:
                    print("Epoch %05d: early stopping." % epoch)
                    self.model.stop_training = True
                    self.model.load_weights('best.h5')
                else:
                    print("Ran out of patience, resetting weights...")
                    self.model.load_weights('original.h5')
                    self.best_roc_val = 0.
                    self.consecutive_worse = 0
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def resnet50():
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(height, width, 3), pooling='avg')
    last = resnet.output
    # x = Flatten()(last)
    # x = Dropout(0.5)(last)
    # x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(last)
    return Model(inputs=[resnet.input], outputs=[x])

# model = resnet50()
# model.summary()


from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(n_splits=5)


k = 0
for index_90, index_70, index_50 in zip(
    skf.split(train_images_90, train_responses_90.squeeze()),
    skf.split(train_images_70, train_responses_70.squeeze()),
    skf.split(train_images_50, train_responses_50.squeeze())):
    k += 1
    train_index_90, test_index_90 = index_90
    train_index_70, test_index_70 = index_70
    train_index_50, test_index_50 = index_50
    images = np.concatenate([
        train_images_90[train_index_90, :, :, :],
        train_images_70[train_index_70, :, :, :],
        train_images_50[train_index_50, :, :, :]
    ], axis=0)
    responses = np.concatenate([
        train_responses_90[train_index_90, :],
        train_responses_70[train_index_70, :],
        train_responses_50[train_index_50, :]
    ], axis=0)
    permutation = np.random.permutation(images.shape[0])
    images = images[permutation, :, :, :]
    responses = responses[permutation, :]
    val_images = np.concatenate([
        train_images_90[test_index_90, :, :, :],
        train_images_70[test_index_70, :, :, :],
        train_images_50[test_index_50, :, :, :]
    ], axis=0)
    val_responses = np.concatenate([
        train_responses_90[test_index_90, :],
        train_responses_70[test_index_70, :],
        train_responses_50[test_index_50, :]
    ], axis=0)
    permutation = np.random.permutation(val_images.shape[0])
    val_images = val_images[permutation, :, :, :]
    val_responses = val_responses[permutation, :]
    # Train model until validation ROC AUC can't be improved
    model = resnet50()
    model.save_weights('original.h5')
    model.compile(
        loss='binary_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )
    model.fit(
        images, responses, batch_size=16, epochs=100,
        validation_data=(val_images, val_responses),
        callbacks=[
            roc_callback(training_data=(images, responses), validation_data=(val_images, val_responses))
        ]
    )
    model.save('resnet50_kfold{}.h5'.format(k))
