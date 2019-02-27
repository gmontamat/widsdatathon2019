import cv2
import numpy as np
import os
import pandas as pd
import pickle

from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.applications.vgg19 import VGG19
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model, load_model, Sequential
from keras.optimizers import SGD
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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

train_images_90 = train_images_90 * 2. - 255.
train_images_70 = train_images_70 * 2. - 255.
train_images_50 = train_images_50 * 2. - 255.
augmented_images_90 = augmented_images_90 * 2. - 255.
augmented_images_70 = augmented_images_70 * 2. - 255.
augmented_images_50 = augmented_images_50 * 2. - 255.
# train_images_90 = preprocess_input(train_images_90)
# train_images_70 = preprocess_input(train_images_70)
# train_images_50 = preprocess_input(train_images_50)
# augmented_images_90 = preprocess_input(augmented_images_90)
# augmented_images_70 = preprocess_input(augmented_images_70)
# augmented_images_50 = preprocess_input(augmented_images_50)
# train_images_90 = np.subtract(train_images_90, [103.939, 116.779, 123.68], dtype=np.float32)
# train_images_70 = np.subtract(train_images_70, [103.939, 116.779, 123.68], dtype=np.float32)
# train_images_50 = np.subtract(train_images_50, [103.939, 116.779, 123.68], dtype=np.float32)
# augmented_images_90 = np.subtract(augmented_images_90, [103.939, 116.779, 123.68], dtype=np.float32)
# augmented_images_70 = np.subtract(augmented_images_70, [103.939, 116.779, 123.68], dtype=np.float32)
# augmented_images_50 = np.subtract(augmented_images_50, [103.939, 116.779, 123.68], dtype=np.float32)


class RocCallback(Callback):
    """
    Define a callback which returns train ROC AUC after
    each epoch and stops early when validation AUC
    doesn't improve.
    """

    def __init__(self, training_data, validation_data):
        super(Callback, self).__init__()
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: {} - roc-auc-val: {}'.format(round(roc, 6), round(roc_val, 6)), end=80 * ' ' + '\n')
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

# Save ResNet50 weights
# resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(height, width, 3), pooling='max')
# x = resnet.output
# x = Dropout(0.75)(x)
# x = Dense(1, activation='sigmoid')(x)
# model = Model(inputs=[resnet.input], outputs=[x])
# for layer in resnet.layers:
#     layer.trainable = False
# model.save_weights("resnet50.h5")

def resnet50():
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(height, width, 3), pooling=None)
    x = resnet.output
    # for layer in resnet.layers:
    #     layer.trainable = False
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.25)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[resnet.input], outputs=[x])
    # for layer in resnet.layers[:10]:
    #     layer.trainable = False
    # model.load_weights("resnet50.h5")
    return model

# model = resnet50()
# model.summary()

kfold = 5
skf = StratifiedKFold(n_splits=kfold)

k = 0
# for index_90, index_70, index_50 in zip(
for index_90, index_70, index_50, index_aug in zip(
        skf.split(train_images_90, train_responses_90.squeeze()),
        skf.split(train_images_70, train_responses_70.squeeze()),
        skf.split(train_images_50, train_responses_50.squeeze()),
        skf.split(augmented_images_90, augmented_responses_90.squeeze())):
    k += 1
    train_index_90, test_index_90 = index_90
    train_index_70, test_index_70 = index_70
    train_index_50, test_index_50 = index_50
    train_index_aug, test_index_aug = index_aug
    images = np.concatenate([
        train_images_90[train_index_90, :, :, :],
        train_images_70[train_index_70, :, :, :],
        train_images_50[train_index_50, :, :, :],
        augmented_images_90[train_index_aug, :, :, :]
    ], axis=0)
    responses = np.concatenate([
        train_responses_90[train_index_90, :],
        train_responses_70[train_index_70, :],
        train_responses_50[train_index_50, :],
        augmented_responses_90[train_index_aug, :]
    ], axis=0)
    permutation = np.random.permutation(images.shape[0])
    images = images[permutation, :, :, :]
    responses = responses[permutation, :]
    val_images = np.concatenate([
        train_images_90[test_index_90, :, :, :],
        train_images_70[test_index_70, :, :, :],
        train_images_50[test_index_50, :, :, :],
        augmented_images_90[test_index_aug, :, :, :]
    ], axis=0)
    val_responses = np.concatenate([
        train_responses_90[test_index_90, :],
        train_responses_70[test_index_70, :],
        train_responses_50[test_index_50, :],
        augmented_responses_90[test_index_aug, :]
    ], axis=0)
    permutation = np.random.permutation(val_images.shape[0])
    val_images = val_images[permutation, :, :, :]
    val_responses = val_responses[permutation, :]
    # Train model until validation ROC AUC can't be improved
    model = resnet50()
    sgd = SGD(lr=1e-3) # , momentum=0.9, nesterov=True)
    # sgd = SGD(lr=1e-4, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(
        images, responses, batch_size=16, epochs=20,
        validation_data=(val_images, val_responses),
        callbacks=[
            RocCallback(training_data=(images, responses), validation_data=(val_images, val_responses))
        ]
    )
    model.save('resnet50_kfold{}.h5'.format(k))


def img_as_array(image_id, size=None, image_set='train_images'):
    image_path = os.path.join(data_path, image_set, image_id)
    img = cv2.imread(str(image_path))
    if size is None:
        return img
    return cv2.resize(img, size)


test_dir = 'leaderboard_test_data'
holdout_dir = 'leaderboard_holdout_data'

test_images = []
test_ids = []
for image_id in os.listdir(os.path.join(data_path, test_dir)):
    img = img_as_array(image_id, image_set=test_dir)
    test_images.append(img.reshape(1, height, width, 3))
    test_ids.append(image_id)
for image_id in os.listdir(os.path.join(data_path, holdout_dir)):
    img = img_as_array(image_id, image_set=holdout_dir)
    test_images.append(img.reshape(1, height, width, 3))
    test_ids.append(image_id)
test_images = np.concatenate(test_images, axis=0)

# test_images = test_images / 255.
test_images = test_images * 2. - 255.
# test_images = preprocess_input(test_images)
# test_images = np.subtract(test_images, [103.939, 116.779, 123.68], dtype=np.float32)

fold_predictions = []
for i in range(1, kfold + 1):
    model_name = 'resnet50_kfold{}.h5'.format(i)
    model = load_model(model_name)
    predictions = model.predict(test_images)
    predictions = predictions.squeeze().tolist()
    fold_predictions.append(predictions)

data_dict = {'has_oilpalm{}'.format(i + 1): fold_predictions[i] for i in range(kfold)}
data_dict['image_id'] = test_ids
submission = pd.DataFrame(data_dict).sort_values('image_id')
submission['has_oilpalm'] = submission[['has_oilpalm{}'.format(i + 1) for i in range(kfold)]].mean(axis=1)
submission = submission[['image_id', 'has_oilpalm']]

submission.to_csv('submission.csv', index=False)
