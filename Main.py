# --------------- Importing Libraries ------------ #
import math
import os
import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from baby_samp import user_recom_stat, item_recom_stat, hybrid_uter_model, A3NCF_model, load_rating_file_as_matrix, \
    load_review_feature
from plot import Comp_plot_Complete_final, Comp_plot_Rating_Complete_final, Perf_plot_Complete_final, \
    org_plot_Rating_Complete_final
import PySimpleGUI as sg


# --------------- Loading Data ------------------- #
def load_data(db):
    if db == 0:
        user_input = np.load("npy_dat/baby_npy/user_input.npy")
        user_fea = np.load("npy_dat/baby_npy/user_fea.npy")
        item_input = np.load("npy_dat/baby_npy/item_input.npy")
        item_fea = np.load("npy_dat/baby_npy/item_fea.npy")
        labels = np.load("npy_dat/baby_npy/labels.npy")

        user_input_test = np.load("npy_dat/baby_npy/user_input_test.npy")
        user_fea_test = np.load("npy_dat/baby_npy/user_fea_test.npy")
        item_input_test = np.load("npy_dat/baby_npy/item_input_test.npy")
        item_fea_test = np.load("npy_dat/baby_npy/item_fea_test.npy")
        test_label = np.load("npy_dat/baby_npy/test_label.npy")
    elif db == 1:
        user_input = np.load("npy_dat/gardan_npy/user_input.npy")
        user_fea = np.load("npy_dat/gardan_npy/user_fea.npy")
        item_input = np.load("npy_dat/gardan_npy/item_input.npy")
        item_fea = np.load("npy_dat/gardan_npy/item_fea.npy")
        labels = np.load("npy_dat/gardan_npy/labels.npy")

        user_input_test = np.load("npy_dat/gardan_npy/user_input_test.npy")
        user_fea_test = np.load("npy_dat/gardan_npy/user_fea_test.npy")
        item_input_test = np.load("npy_dat/gardan_npy/item_input_test.npy")
        item_fea_test = np.load("npy_dat/gardan_npy/item_fea_test.npy")
        test_label = np.load("npy_dat/gardan_npy/test_label.npy")

    elif db == 2:
        user_input = np.load("npy_dat/music_npy/user_input.npy")
        user_fea = np.load("npy_dat/music_npy/user_fea.npy")
        item_input = np.load("npy_dat/music_npy/item_input.npy")
        item_fea = np.load("npy_dat/music_npy/item_fea.npy")
        labels = np.load("npy_dat/music_npy/labels.npy")

        user_input_test = np.load("npy_dat/music_npy/user_input_test.npy")
        user_fea_test = np.load("npy_dat/music_npy/user_fea_test.npy")
        item_input_test = np.load("npy_dat/music_npy/item_input_test.npy")
        item_fea_test = np.load("npy_dat/music_npy/item_fea_test.npy")
        test_label = np.load("npy_dat/music_npy/test_label.npy")

    return user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, test_label


def pre_process(db):
    if db == 0:
        file_path = 'Data/baby_recom/'
        dataName = "Baby"  # name of your product
    elif db == 1:
        file_path = 'Data/Music_recom/'
        dataName = "Digital_Music"  # name of your product
    elif db == 2:
        file_path = 'Data/Gardan_recom/'
        dataName = "Patio_Lawn_and_Garden"  # name of your product

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

    path = file_path + dataName
    trainMatrix = load_rating_file_as_matrix(path + ".train.rating")
    user_review_fea = load_review_feature(path + ".user")
    item_review_fea = load_review_feature(path + ".item")
    testRatings = load_rating_file_as_matrix(path + ".test.rating")
    num_users, num_items = trainMatrix.shape

    print(num_users, num_items)
    return trainMatrix, user_review_fea, item_review_fea, testRatings, num_users, num_items


# ------------------ Metrics ----------------------- #
def error_metrics(pred, test_label):
    mae = mean_absolute_error(pred, test_label)
    mse = mean_squared_error(pred, test_label)
    rmse = math.sqrt(mse)
    return [mae, mse, rmse]


def org_rat(prediction, test_label):
    dis = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for i in prediction.tolist():
        if round(i[0]) in dis:
            dis[round(i[0])] += 1
        else:
            dis[1] += 1
    print(dis)

    for i in range(1, 6):
        dis[i] /= len(prediction)
    print(dis)

    label = {}
    for i in test_label:
        if i in label:
            label[i] += 1
        else:
            label[i] = 1
    for i in range(1, 6):
        label[i] /= len(test_label)
    print(label)
    return dis, label


def raing_data(user_input_test, user_fea_test, item_input_test, item_feat_test, test_label, ii):
            user_input_test1 = []
            user_fea_test1 = []
            item_input_test1 = []
            item_fea_test1 = []
            test_label1 = []
            for ind, i in enumerate(test_label):
                if i == ii:
                    user_input_test1.append(user_input_test[ind])
                    user_fea_test1.append(user_fea_test[ind])
                    item_input_test1.append(item_input_test[ind])
                    item_fea_test1.append(item_feat_test[ind])
                    test_label1.append(i)

            return np.array(user_input_test1), np.array(user_fea_test1),  np.array(item_input_test1),  np.array(item_fea_test1), np.array(test_label1)


def origl_Rating(vector_size, num_users, num_items, user_input1, user_fea1, item_input1, item_fea1, labels1, user_input_test, user_fea_test, item_input_test, item_fea_test, test_label, db):

    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []
    perf_F = []
    perf_G = []

    epochs = 50
    tr_per = 0.8
    # for i in range(5):
    user_input1 = user_input1[0: int(len(user_input1) * tr_per) - 1]
    user_fea1 = user_fea1[0: int(len(user_fea1) * tr_per) - 1]
    item_input1 = item_input1[0: int(len(item_input1) * tr_per) - 1]
    item_fea1 = item_fea1[0: int(len(item_fea1) * tr_per) - 1]
    labels1 = labels1[0: int(len(labels1) * tr_per) - 1]

    user_input_test = user_input_test[0: int(len(user_input_test) * tr_per) - 1]
    user_fea_test = user_fea_test[0: int(len(user_fea_test) * tr_per) - 1]
    item_input_test = item_input_test[0: int(len(item_input_test) * tr_per) - 1]
    item_fea_test = item_fea_test[0: int(len(item_fea_test) * tr_per) - 1]
    test_label = test_label[0: int(len(test_label) * tr_per) - 1]

    model1 = user_recom_stat(vector_size, num_users, user_input1, user_fea1, labels1, user_input_test,
                             user_fea_test, epochs)
    model2 = item_recom_stat(vector_size, num_items, item_input1, item_fea1, labels1, item_input_test,
                             item_fea_test, epochs)
    model3 = A3NCF_model(num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test,
                         user_fea_test, item_input_test, item_fea_test, vector_size, latent_dim=100, regs=[0, 0], epochs=epochs)
    model4 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels,
                               user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=0)
    model5 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels,
                               user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=1)
    model6 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels,
                               user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=2)
    model7 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels,
                               user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=3)

    for rating in range(1, 6):
        # ----------------- Varying the Rating  ---------------- #
        user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d, test_label_d = raing_data(user_input_test,
                                                                                                user_fea_test,
                                                                                                item_input_test,
                                                                                                item_fea_test,
                                                                                                test_label,
                                                                                                rating)
        pred_1 = model1.predict([user_input_test_d, user_fea_test_d])
        pred_2 = model2.predict([item_input_test_d, item_fea_test_d])
        pred_3 = model3.predict([user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d])
        pred_4 = model4.predict([user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d])
        pred_5 = model5.predict([user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d])
        pred_6 = model6.predict([user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d])
        pred_7 = model7.predict([user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d])

        per_1 = org_rat(pred_1, test_label)
        per_2 = org_rat(pred_2, test_label)
        per_3 = org_rat(pred_3, test_label)
        per_4 = org_rat(pred_4, test_label)
        per_5 = org_rat(pred_5, test_label)
        per_6 = org_rat(pred_6, test_label)
        per_7 = org_rat(pred_7, test_label)

        perf_A.append(per_1)
        perf_B.append(per_2)
        perf_C.append(per_3)
        perf_D.append(per_4)
        perf_E.append(per_5)
        perf_F.append(per_6)
        perf_G.append(per_7)

    if db == 0:
        np.save('{0}\\npy\\db_1\\pred_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_1\\pred_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_1\\pred_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_1\\pred_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_1\\pred_E'.format(os.getcwd()), perf_E)
        np.save('{0}\\npy\\db_1\\pred_F'.format(os.getcwd()), perf_F)
        np.save('{0}\\npy\\db_1\\pred_G'.format(os.getcwd()), perf_G)
    elif db == 1:
        np.save('{0}\\npy\\db_2\\pred_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_2\\pred_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_2\\pred_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_2\\pred_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_2\\pred_E'.format(os.getcwd()), perf_E)
        np.save('{0}\\npy\\db_2\\pred_F'.format(os.getcwd()), perf_F)
        np.save('{0}\\npy\\db_2\\pred_G'.format(os.getcwd()), perf_G)
    elif db == 2:
        np.save('{0}\\npy\\db_3\\pred_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_3\\pred_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_3\\pred_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_3\\pred_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_3\\pred_E'.format(os.getcwd()), perf_E)
        np.save('{0}\\npy\\db_3\\pred_F'.format(os.getcwd()), perf_F)
        np.save('{0}\\npy\\db_3\\pred_G'.format(os.getcwd()), perf_G)


def Comp_Analysis_Rating(vector_size, num_users, num_items, user_input1, user_fea1, item_input1, item_fea1, labels1, user_input_test, user_fea_test, item_input_test, item_fea_test, test_label, db):

    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []
    perf_F = []
    perf_G = []

    epochs = 50
    tr_per = 0.8
    # for i in range(5):
    user_input1 = user_input1[0: int(len(user_input1) * tr_per) - 1]
    user_fea1 = user_fea1[0: int(len(user_fea1) * tr_per) - 1]
    item_input1 = item_input1[0: int(len(item_input1) * tr_per) - 1]
    item_fea1 = item_fea1[0: int(len(item_fea1) * tr_per) - 1]
    labels1 = labels1[0: int(len(labels1) * tr_per) - 1]

    user_input_test = user_input_test[0: int(len(user_input_test) * tr_per) - 1]
    user_fea_test = user_fea_test[0: int(len(user_fea_test) * tr_per) - 1]
    item_input_test = item_input_test[0: int(len(item_input_test) * tr_per) - 1]
    item_fea_test = item_fea_test[0: int(len(item_fea_test) * tr_per) - 1]
    test_label = test_label[0: int(len(test_label) * tr_per) - 1]

    model1 = user_recom_stat(vector_size, num_users, user_input1, user_fea1, labels1, user_input_test,
                             user_fea_test, epochs)
    model2 = item_recom_stat(vector_size, num_items, item_input1, item_fea1, labels1, item_input_test,
                             item_fea_test, epochs)
    model3 = A3NCF_model(num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test,
                         user_fea_test, item_input_test, item_fea_test, vector_size, latent_dim=100, regs=[0, 0], epochs=epochs)
    model4 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels,
                               user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=0)
    model5 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels,
                               user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=1)
    model6 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels,
                               user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=2)
    model7 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels,
                               user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=3)

    for rating in range(1, 6):
        # ----------------- Varying the Rating  ---------------- #
        user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d, test_label_d = raing_data(user_input_test,
                                                                                                user_fea_test,
                                                                                                item_input_test,
                                                                                                item_fea_test,
                                                                                                test_label,
                                                                                                rating)
        pred_1 = model1.predict([user_input_test_d, user_fea_test_d])
        pred_2 = model2.predict([item_input_test_d, item_fea_test_d])
        pred_3 = model3.predict([user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d])
        pred_4 = model4.predict([user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d])
        pred_5 = model5.predict([user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d])
        pred_6 = model6.predict([user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d])
        pred_7 = model7.predict([user_input_test_d, user_fea_test_d, item_input_test_d, item_fea_test_d])

        [MAE1, MSE1, RMSE1] = error_metrics(pred_1, test_label_d)
        [MAE2, MSE2, RMSE2] = error_metrics(pred_2, test_label_d)
        [MAE3, MSE3, RMSE3] = error_metrics(pred_3, test_label_d)
        [MAE4, MSE4, RMSE4] = error_metrics(pred_4, test_label_d)
        [MAE5, MSE5, RMSE5] = error_metrics(pred_5, test_label_d)
        [MAE6, MSE6, RMSE6] = error_metrics(pred_6, test_label_d)
        [MAE7, MSE7, RMSE7] = error_metrics(pred_7, test_label_d)

        per_1 = [MAE1, MSE1, RMSE1]
        per_2 = [MAE2, MSE2, RMSE2]
        per_3 = [MAE3, MSE3, RMSE3]
        per_4 = [MAE4, MSE4, RMSE4]
        per_5 = [MAE5, MSE5, RMSE5]
        per_6 = [MAE6, MSE6, RMSE6]
        per_7 = [MAE7, MSE7, RMSE7]

        perf_A.append(per_1)
        perf_B.append(per_2)
        perf_C.append(per_3)
        perf_D.append(per_4)
        perf_E.append(per_5)
        perf_F.append(per_6)
        perf_G.append(per_7)

    if db == 0:
        np.save('{0}\\npy\\db_1\\recom_R_comp_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_1\\recom_R_comp_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_1\\recom_R_comp_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_1\\recom_R_comp_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_1\\recom_R_comp_E'.format(os.getcwd()), perf_E)
        np.save('{0}\\npy\\db_1\\recom_R_comp_F'.format(os.getcwd()), perf_F)
        np.save('{0}\\npy\\db_1\\recom_R_comp_G'.format(os.getcwd()), perf_G)
    elif db == 1:
        np.save('{0}\\npy\\db_2\\recom_R_comp_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_2\\recom_R_comp_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_2\\recom_R_comp_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_2\\recom_R_comp_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_2\\recom_R_comp_E'.format(os.getcwd()), perf_E)
        np.save('{0}\\npy\\db_2\\recom_R_comp_F'.format(os.getcwd()), perf_F)
        np.save('{0}\\npy\\db_2\\recom_R_comp_G'.format(os.getcwd()), perf_G)
    elif db == 2:
        np.save('{0}\\npy\\db_3\\recom_R_comp_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_3\\recom_R_comp_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_3\\recom_R_comp_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_3\\recom_R_comp_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_3\\recom_R_comp_E'.format(os.getcwd()), perf_E)
        np.save('{0}\\npy\\db_3\\recom_R_comp_F'.format(os.getcwd()), perf_F)
        np.save('{0}\\npy\\db_3\\recom_R_comp_G'.format(os.getcwd()), perf_G)


# -------------- Comp Analysis ------------------ #


def comp_analysis(vector_size, num_users, num_items, user_input1, user_fea1, item_input1, item_fea1, labels1, user_input_test, user_fea_test, item_input_test, item_fea_test, test_label, db):

    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []
    perf_F = []
    perf_G = []

    epochs = 50
    tr_per = 0.4
    for i in range(6):
        user_input1 = user_input1[0: int(len(user_input1) * tr_per) - 1]
        user_fea1 = user_fea1[0: int(len(user_fea1) * tr_per) - 1]
        item_input1 = item_input1[0: int(len(item_input1) * tr_per) - 1]
        item_fea1 = item_fea1[0: int(len(item_fea1) * tr_per) - 1]
        labels1 = labels1[0: int(len(labels1) * tr_per) - 1]

        user_input_test = user_input_test[0: int(len(user_input_test) * tr_per) - 1]
        user_fea_test = user_fea_test[0: int(len(user_fea_test) * tr_per) - 1]
        item_input_test = item_input_test[0: int(len(item_input_test) * tr_per) - 1]
        item_fea_test = item_fea_test[0: int(len(item_fea_test) * tr_per) - 1]
        test_label = test_label[0: int(len(test_label) * tr_per) - 1]

        model1 = user_recom_stat(vector_size, num_users, user_input1, user_fea1, labels1, user_input_test, user_fea_test, epochs)
        model2 = item_recom_stat(vector_size, num_items, item_input1, item_fea1, labels1, item_input_test, item_fea_test, epochs)
        model3 = A3NCF_model(num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, vector_size, latent_dim=100, regs=[0, 0], epochs=epochs)
        model4 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=0)
        model5 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=1)
        model6 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=2)
        model7 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, epochs, opt=3)

        pred_1 = model1.predict([user_input_test, user_fea_test])
        pred_2 = model2.predict([item_input_test, item_fea_test])
        pred_3 = model3.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
        pred_4 = model4.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
        pred_5 = model5.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
        pred_6 = model6.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
        pred_7 = model7.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])

        [MAE1, MSE1, RMSE1] = error_metrics(pred_1, test_label)
        [MAE2, MSE2, RMSE2] = error_metrics(pred_2, test_label)
        [MAE3, MSE3, RMSE3] = error_metrics(pred_3, test_label)
        [MAE4, MSE4, RMSE4] = error_metrics(pred_4, test_label)
        [MAE5, MSE5, RMSE5] = error_metrics(pred_5, test_label)
        [MAE6, MSE6, RMSE6] = error_metrics(pred_6, test_label)
        [MAE7, MSE7, RMSE7] = error_metrics(pred_7, test_label)

        per_1 = [MAE1, MSE1, RMSE1]
        per_2 = [MAE2, MSE2, RMSE2]
        per_3 = [MAE3, MSE3, RMSE3]
        per_4 = [MAE4, MSE4, RMSE4]
        per_5 = [MAE5, MSE5, RMSE5]
        per_6 = [MAE6, MSE6, RMSE6]
        per_7 = [MAE7, MSE7, RMSE7]

        perf_A.append(per_1)
        perf_B.append(per_2)
        perf_C.append(per_3)
        perf_D.append(per_4)
        perf_E.append(per_5)
        perf_F.append(per_6)
        perf_G.append(per_7)
        tr_per += 0.1
    if db == 0:
        np.save('{0}\\npy\\db_1\\recom_comp_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_1\\recom_comp_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_1\\recom_comp_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_1\\recom_comp_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_1\\recom_comp_E'.format(os.getcwd()), perf_E)
        np.save('{0}\\npy\\db_1\\recom_comp_F'.format(os.getcwd()), perf_F)
        np.save('{0}\\npy\\db_1\\recom_comp_G'.format(os.getcwd()), perf_G)
    elif db == 1:
        np.save('{0}\\npy\\db_2\\recom_comp_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_2\\recom_comp_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_2\\recom_comp_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_2\\recom_comp_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_2\\recom_comp_E'.format(os.getcwd()), perf_E)
        np.save('{0}\\npy\\db_2\\recom_comp_F'.format(os.getcwd()), perf_F)
        np.save('{0}\\npy\\db_2\\recom_comp_G'.format(os.getcwd()), perf_G)
    elif db == 2:
        np.save('{0}\\npy\\db_3\\recom_comp_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_3\\recom_comp_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_3\\recom_comp_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_3\\recom_comp_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_3\\recom_comp_E'.format(os.getcwd()), perf_E)
        np.save('{0}\\npy\\db_3\\recom_comp_F'.format(os.getcwd()), perf_F)
        np.save('{0}\\npy\\db_3\\recom_comp_G'.format(os.getcwd()), perf_G)


# -------------- Performance Analysis ---------------- #
def perf_analysis(vector_size, num_users, num_items, user_input1, user_fea1, item_input1, item_fea1, labels1, user_input_test, user_fea_test, item_input_test, item_fea_test, test_label, db):
    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []

    tr_per = 0.4
    for i in range(6):
        user_input1 = user_input1[0: int(len(user_input1) * tr_per) - 1]
        user_fea1 = user_fea1[0: int(len(user_fea1) * tr_per) - 1]
        item_input1 = item_input1[0: int(len(item_input1) * tr_per) - 1]
        item_fea1 = item_fea1[0: int(len(item_fea1) * tr_per) - 1]
        labels1 = labels1[0: int(len(labels1) * tr_per) - 1]

        user_input_test = user_input_test[0: int(len(user_input_test) * tr_per) - 1]
        user_fea_test = user_fea_test[0: int(len(user_fea_test) * tr_per) - 1]
        item_input_test = item_input_test[0: int(len(item_input_test) * tr_per) - 1]
        item_fea_test = item_fea_test[0: int(len(item_fea_test) * tr_per) - 1]
        test_label = test_label[0: int(len(test_label) * tr_per) - 1]

        model_1, pred_1 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, epochs=10, opt=3)
        model_2, pred_2 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, epochs=20, opt=3)
        model_3, pred_3 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, epochs=30, opt=3)
        model_4, pred_4 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, epochs=40, opt=3)
        model_5, pred_5 = hybrid_uter_model(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, epochs=50, opt=3)

        [MAE1, MSE1, RMSE1] = error_metrics(model_1, test_label)
        [MAE2, MSE2, RMSE2] = error_metrics(model_2, test_label)
        [MAE3, MSE3, RMSE3] = error_metrics(model_3, test_label)
        [MAE4, MSE4, RMSE4] = error_metrics(model_4, test_label)
        [MAE5, MSE5, RMSE5] = error_metrics(model_5, test_label)

        per_1 = [MAE1, MSE1, RMSE1]
        per_2 = [MAE2, MSE2, RMSE2]
        per_3 = [MAE3, MSE3, RMSE3]
        per_4 = [MAE4, MSE4, RMSE4]
        per_5 = [MAE5, MSE5, RMSE5]

        perf_A.append(per_1)
        perf_B.append(per_2)
        perf_C.append(per_3)
        perf_D.append(per_4)
        perf_E.append(per_5)
        tr_per += 0.1
    if db == 1:
        np.save('{0}\\npy\\db_1\\recom_perf_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_1\\recom_perf_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_1\\recom_perf_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_1\\recom_perf_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_1\\recom_perf_E'.format(os.getcwd()), perf_E)

    elif db == 2:
        np.save('{0}\\npy\\db_2\\recom_perf_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_2\\recom_perf_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_2\\recom_perf_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_2\\recom_perf_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_2\\recom_perf_E'.format(os.getcwd()), perf_E)

    elif db == 3:
        np.save('{0}\\npy\\db_3\\recom_perf_A'.format(os.getcwd()), perf_A)
        np.save('{0}\\npy\\db_3\\recom_perf_B'.format(os.getcwd()), perf_B)
        np.save('{0}\\npy\\db_3\\recom_perf_C'.format(os.getcwd()), perf_C)
        np.save('{0}\\npy\\db_3\\recom_perf_D'.format(os.getcwd()), perf_D)
        np.save('{0}\\npy\\db_3\\recom_perf_E'.format(os.getcwd()), perf_E)


def get_details(p_name, root_path):
    """## Preprocessing"""

    # Hyperparameter
    dataName = p_name  # name of your product

    # get read all files we have saved in previous stage
    path = 'Data/'+root_path+'/' + dataName
    trainMatrix = load_rating_file_as_matrix(path + ".train.rating")
    user_review_fea = load_review_feature(path + ".user")
    item_review_fea = load_review_feature(path + ".item")
    testRatings = load_rating_file_as_matrix(path + ".test.rating")
    num_users, num_items = trainMatrix.shape

    print(num_users, num_items)
    return trainMatrix, user_review_fea, item_review_fea, testRatings, num_users, num_items


VVV = sg.popup_yes_no('Do you Want the Complete Execution?')
if (VVV=='Yes'):
    global trainMatrix, user_review_fea, item_review_fea, testRatings, num_users, num_items
    p_name_all = ['Baby', 'Patio_Lawn_and_Garden', 'Digital_Music']
    root_path = ['baby_recom', 'Gardan_recom', 'Music_recom']
    vector_size = 100  # vector size of review text and item description
    epochs = 50
    for db in range(3):
        trainMatrix, user_review_fea, item_review_fea, testRatings, num_users, num_items = get_details(p_name_all[db],root_path[db])
        user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, test_label = load_data(db)
        origl_Rating(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, test_label, db)
        Comp_Analysis_Rating(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, test_label, db)
        comp_analysis(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, test_label, db)
        perf_analysis(vector_size, num_users, num_items, user_input, user_fea, item_input, item_fea, labels, user_input_test, user_fea_test, item_input_test, item_fea_test, test_label, db)

    for i in range(1, 4):
        Comp_plot_Rating_Complete_final(i, 1)
        Comp_plot_Complete_final(i, 1)
        org_plot_Rating_Complete_final(i, 1)
        Perf_plot_Complete_final(i, 1)
else:
    for i in range(1, 4):
        Comp_plot_Rating_Complete_final(i, 1)
        Comp_plot_Complete_final(i, 1)
        org_plot_Rating_Complete_final(i, 1)
        # Perf_plot_Complete_final(i, 1)

