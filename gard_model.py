# ------------ Importing Libraries -------------- #

import json
import gzip
import gensim
import random
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from collections import defaultdict
from keras.layers.core import Dropout
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics import confusion_matrix
# from mealpy.swarm_based.BES import BaseBES as BES
from mealpy.swarm_based.BA import BaseBA as BA
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Multiply, Lambda, Reshape

"""## Preprocessing"""

# # Hyperparameter
# dataName = "Patio_Lawn_and_Garden"  # name of your product
# vector_size = 100  # vector size of reviewtext and item description
# epochs = 5


def get_max_users_max_items(filename):
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
        line = f.readline()
        # print(line)
        while line is not None and line != "":
            arr = line.split("  ")
            u, i = int(arr[0]), int(arr[1])
            num_users = max(num_users, u)
            num_items = max(num_items, i)
            line = f.readline()
            # print(line)

    return num_users, num_items
# Construct  sparse matrix with size of maxmuim of users and items


def load_rating_file_as_matrix(filename):
    num_users, num_items = get_max_users_max_items(filename)
    # print(num_users , num_items)
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            arr = line.split("  ")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if rating > 0:
                mat[user, item] = rating
            line = f.readline()
    return mat
# get the  text vector for  user_review and item_description


def load_review_feature(filename):
    dict = {}
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            fea = line.strip('\n').split(',')
            index = int(fea[0])
            if index not in dict:
                dict[index] = fea[1:]
            line = f.readline()
    return dict
# get read all files we have saved in previous stage


# path = 'Data/Gardan_recom/' + dataName
# trainMatrix = load_rating_file_as_matrix(path + ".train.rating")
# user_review_fea = load_review_feature(path + ".user")
# item_review_fea = load_review_feature(path + ".item")
# testRatings = load_rating_file_as_matrix(path + ".test.rating")
# num_users, num_items = trainMatrix.shape
#
# print(num_users, num_items)


# ----------------------- Optimization -------------------- #
def fitness_function1(solution, model, test_data, test_label):
    # try:
    opt_val = model.get_weights()
    opt_val_1 = opt_val[0]
    solution.reshape(opt_val_1.shape[0], opt_val_1.shape[1])
    opt_val[0] = solution.reshape(opt_val_1.shape[0], opt_val_1.shape[1])
    model.set_weights(opt_val)
    y_pred = model.predict(test_data)
    # Acc =sklearn.metrics.accuracy_score(np.argmax(test_label, axis = 1), np.argmax(y_pred, axis = 1))
    # Acc = perf_evalution_CM(test_label, y_pred)
    Acc = mean_squared_error(test_label, y_pred)
    return Acc


def update_weights_model_BA(model, test_data, test_lab, opt):
    # named_model = dict(model.named_modules())
    to_opt = model.get_weights()
    to_opt_1 = np.array(to_opt[0])
    to_opt_2 = to_opt_1.reshape(to_opt_1.shape[0], to_opt_1.shape[1])
    wei_to_train_1 = main_weight_updation_optimization(to_opt_2, model, test_data, test_lab, opt)
    to_opt[0] = wei_to_train_1.reshape(to_opt_1.shape[0], to_opt_1.shape[1])
    model.set_weights(to_opt)
    # model.predict(test_data)
    # print("MSE OF OPT : ", mean_squared_error(model.predict(test_data), test_lab))
    # model = model
    return model


def main_weight_updation_optimization(curr_wei, model, test_data, test_lab, opt):

    problem_dict1 = {
        "fit_func": fitness_function1,
        "lb": [curr_wei.min(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        "ub": [curr_wei.max(), ] * curr_wei.shape[0] * curr_wei.shape[1],
        "minmax": "min",
        "log_to": None,
        "save_population": True,
        "Curr_Weight": curr_wei,
        "Model_trained_Partial": model,
        "test_label": test_lab,
        "test_loader": test_data,
                    }
    if opt == 1:
        print(" BAT Optimization")
        model = BA(problem_dict1, epoch=5, pop_size=20)
    # best_position2, best_fitness2 = curr_wei, model.solve()
    best_position2, best_fitness2 = model.solve()
    print(best_position2)
    return best_position2

#
# """## Build Model"""
#
# # build model
# # Intiliaze the model input
# user_input = Input(shape=(1,), dtype='int32', name='user_input')
# user_sent = Input(shape=(vector_size,), dtype='float32', name='user_sentiment')
# item_input = Input(shape=(1,), dtype='int32', name='item_input')
# item_cont = Input(shape=(vector_size,), dtype='float32', name='item_content')
#
# # Embedding layer for user
# Embedding_User = Embedding(input_dim=num_users, input_length=1, output_dim=vector_size, name='user_embedding')
#
# # Embedding layer for item
# Embedding_Item = Embedding(input_dim=num_items, input_length=1, output_dim=vector_size, name='item_embedding')
#
#
# # User Sentiment Dense Network


def user_Sentiment(user_latent, user_sent):
    latent_size = user_latent.shape[1]
    inputs = user_sent
    layer = Dense(latent_size, activation='relu', name='user_attention_layer')(inputs)
    sent = Lambda(lambda x: K.softmax(x), name='user_Sentiment_softmax')(layer)
    output = Multiply()([user_latent, sent])
    return output

# Item Content Dense Network


def item_Content(item_latent, item_cont):
    latent_size = item_latent.shape[1]
    inputs = item_cont
    layer = Dense(latent_size, activation='relu', name='item_attention_layer')(inputs)
    cont = Lambda(lambda x: K.softmax(x), name='item_Content_softmax')(layer)
    output = Multiply()([item_latent, cont])
    return output
#
# # Crucial to flatten an embedding vector
#
#
# user_latent = Reshape((vector_size,))(Flatten()(Embedding_User(user_input)))
# item_latent = Reshape((vector_size,))(Flatten()(Embedding_Item(item_input)))
# user_latent_atten = user_Sentiment(user_latent, user_sent)
# item_latent_atten = item_Content(item_latent, item_cont)
#
# user_latent = Dense(vector_size, activation='relu')(user_latent_atten)
# item_latent = Dense(vector_size, activation='relu')(item_latent_atten)
#
# # review-based attention calculation
# vec = Multiply()([user_latent, item_latent])
# user_item_concat = Concatenate()([user_sent, item_cont, user_latent, item_latent])
# att = Dense(vector_size, kernel_initializer='random_uniform', activation='softmax')(user_item_concat)
#
# # Element-wise product of user and item embeddings
# predict_vec = Multiply()([vec, att])
# # Final prediction layer
# prediction = Dense(vector_size, activation='relu')(predict_vec)
# # for overfitting
# prediction = Dropout(0.5)(prediction)
# prediction = Dense(1, name='prediction')(prediction)
#
# model = Model(inputs=[user_input, user_sent, item_input, item_cont], outputs=prediction)
#
# model.summary()
#
# # compile our model
# model.compile(optimizer="adam", loss="mean_absolute_error", metrics=['mean_squared_error'])
#
# # using callbacks to avoid overfitting and save best model during training
# earlystopping = EarlyStopping(monitor='val_loss', patience=100)
# checkpoint = ModelCheckpoint('Data/Gardan_recom/Patio_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Make the data ready fro training
# convert it to numpy array as we extract the labels


def get_instances(trainMatrix, user_review_fea, item_review_fea):
    user_input, user_fea, item_input, item_fea, labels = [], [], [], [], []
    num_users = trainMatrix.shape[0]
    for (u, i) in trainMatrix.keys():
        if u in user_review_fea.keys() and i in item_review_fea.keys():
            user_input.append(u)
            user_fea.append(user_review_fea[u])
            item_input.append(i)
            item_fea.append(item_review_fea[i])
            label = trainMatrix[u, i]
            labels.append(label)
    return np.array(user_input), np.array(user_fea, dtype='float32'), np.array(item_input), \
           np.array(item_fea, dtype='float32'), np.array(labels)


# user_input = np.load("npy_dat/gardan_npy/user_input.npy")
# user_fea = np.load("npy_dat/gardan_npy/user_fea.npy")
# item_input = np.load("npy_dat/gardan_npy/item_input.npy")
# item_fea = np.load("npy_dat/gardan_npy/item_fea.npy")
# labels = np.load("npy_dat/gardan_npy/labels.npy")
#
# # fit our model
# hist = model.fit([user_input, user_fea, item_input, item_fea], labels, batch_size=64, epochs=epochs, validation_split=0.2,
#                  verbose=1, callbacks=[earlystopping, checkpoint])
# opt_model = update_weights_model_BA(model, [user_input, user_fea, item_input, item_fea], labels, 1)
#
# # Draw the progress during the training
# loss = hist.history['loss']
# val_loss = hist.history['val_loss']
# epochs = range(1, len(loss) + 1)
# # loss plot
# plt.plot(epochs, loss, color='pink', label='Training Loss')
# plt.plot(epochs, val_loss, color='red', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # get instance for testing data
# user_input_test = np.load("npy_dat/gardan_npy/user_input_test.npy")
# user_fea_test = np.load("npy_dat/gardan_npy/user_fea_test.npy")
# item_input_test = np.load("npy_dat/gardan_npy/item_input_test.npy")
# item_fea_test = np.load("npy_dat/gardan_npy/item_fea_test.npy")
# test_label = np.load("npy_dat/gardan_npy/test_label.npy")
#
# # --------------- evulate the test dataset ----------------- #
# print("MAE of UTER model : ", mean_absolute_error(model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]), test_label))
#
# print("MSE of UTER model : ", mean_squared_error(model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]), test_label))
#
# pred = model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
#
# # --------- evaluate the test dataset ------------ #
# print("MAE of Optimization model : ", mean_absolute_error(opt_model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]), test_label))
#
# print("MSE of Optimization model : ", mean_squared_error(opt_model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]), test_label))

# pred_opt = opt_model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
# print("Prediction of UTER : \n", pred)
# print("Prediction of Opt : \n", pred_opt)
