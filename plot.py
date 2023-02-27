import os
import matplotlib.pyplot as plt
import numpy as np


# --------------- LOADING THE NPY OF COMP_ANALYSIS ---------------- #
def load_perf_value_saved_Algo_Analysis(tt):
    if tt == 1:
        perf_A = np.load('{0}\\npy\\db_1\\recom_comp_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_1\\recom_comp_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_1\\recom_comp_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_1\\recom_comp_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_1\\recom_comp_E.npy'.format(os.getcwd()))
        perf_F = np.load('{0}\\npy\\db_1\\recom_comp_F.npy'.format(os.getcwd()))
        perf_G = np.load('{0}\\npy\\db_1\\recom_comp_G.npy'.format(os.getcwd()))

    elif tt == 2:

        perf_A = np.load('{0}\\npy\\db_2\\recom_comp_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_2\\recom_comp_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_2\\recom_comp_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_2\\recom_comp_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_2\\recom_comp_E.npy'.format(os.getcwd()))
        perf_F = np.load('{0}\\npy\\db_2\\recom_comp_F.npy'.format(os.getcwd()))
        perf_G = np.load('{0}\\npy\\db_2\\recom_comp_G.npy'.format(os.getcwd()))

    elif tt == 3:

        perf_A = np.load('{0}\\npy\\db_3\\recom_comp_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_3\\recom_comp_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_3\\recom_comp_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_3\\recom_comp_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_3\\recom_comp_E.npy'.format(os.getcwd()))
        perf_F = np.load('{0}\\npy\\db_3\\recom_comp_F.npy'.format(os.getcwd()))
        perf_G = np.load('{0}\\npy\\db_3\\recom_comp_G.npy'.format(os.getcwd()))

    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])
    E = np.asarray(perf_E[:][:])
    F = np.asarray(perf_F[:][:])
    G = np.asarray(perf_G[:][:])

    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()
    EE = E[:][:].transpose()
    FF = F[:][:].transpose()
    GG = G[:][:].transpose()

    return [AA, BB, CC, DD, EE, FF, GG]


def Main_comp_val_acc_sen_spe(A, B, C, D, E, F, G, tt):
    VALLL = np.column_stack((A[0], B[0], C[0], D[0], E[0], F[0], G[0]))
    perf1 = VALLL
    # perf1 = Perf_est_all_final(VALLL)
    VALLL = np.column_stack((A[1], B[1], C[1], D[1], E[1], F[1], G[1]))
    perf2 = VALLL
    # # perf2 = Perf_est_all_final(VALLL)

    VALLL = np.column_stack((A[2], B[2], C[2], D[2], E[2], F[2], G[2]))
    perf3 = VALLL

    return [np.flipud(perf1), np.flipud(perf2), np.flipud(perf3)]


def load_perf_value_saved_Algo_Analysis1(tt):
    if tt == 1:
        perf_A = np.load('{0}\\npy\\db_1\\recom_R_comp_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_1\\recom_R_comp_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_1\\recom_R_comp_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_1\\recom_R_comp_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_1\\recom_R_comp_E.npy'.format(os.getcwd()))
        perf_F = np.load('{0}\\npy\\db_1\\recom_R_comp_F.npy'.format(os.getcwd()))
        perf_G = np.load('{0}\\npy\\db_1\\recom_R_comp_G.npy'.format(os.getcwd()))

    elif tt == 2:

        perf_A = np.load('{0}\\npy\\db_2\\recom_R_comp_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_2\\recom_R_comp_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_2\\recom_R_comp_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_2\\recom_R_comp_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_2\\recom_R_comp_E.npy'.format(os.getcwd()))
        perf_F = np.load('{0}\\npy\\db_2\\recom_R_comp_F.npy'.format(os.getcwd()))
        perf_G = np.load('{0}\\npy\\db_2\\recom_R_comp_G.npy'.format(os.getcwd()))

    elif tt == 3:

        perf_A = np.load('{0}\\npy\\db_3\\recom_R_comp_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_3\\recom_R_comp_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_3\\recom_R_comp_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_3\\recom_R_comp_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_3\\recom_R_comp_E.npy'.format(os.getcwd()))
        perf_F = np.load('{0}\\npy\\db_3\\recom_R_comp_F.npy'.format(os.getcwd()))
        perf_G = np.load('{0}\\npy\\db_3\\recom_R_comp_G.npy'.format(os.getcwd()))

    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])
    E = np.asarray(perf_E[:][:])
    F = np.asarray(perf_F[:][:])
    G = np.asarray(perf_G[:][:])

    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()
    EE = E[:][:].transpose()
    FF = F[:][:].transpose()
    GG = G[:][:].transpose()

    return [AA, BB, CC, DD, EE, FF, GG]






def load_perf_value_saved_Algo_Analysis2(tt):
    if tt == 1:
        perf_A = np.load('{0}\\npy\\db_1\\pred_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_1\\pred_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_1\\pred_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_1\\pred_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_1\\pred_E.npy'.format(os.getcwd()))
        perf_F = np.load('{0}\\npy\\db_1\\pred_F.npy'.format(os.getcwd()))
        perf_G = np.load('{0}\\npy\\db_1\\pred_G.npy'.format(os.getcwd()))

    elif tt == 2:

        perf_A = np.load('{0}\\npy\\db_2\\pred_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_2\\pred_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_2\\pred_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_2\\pred_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_2\\pred_E.npy'.format(os.getcwd()))
        perf_F = np.load('{0}\\npy\\db_2\\pred_F.npy'.format(os.getcwd()))
        perf_G = np.load('{0}\\npy\\db_2\\pred_G.npy'.format(os.getcwd()))

    elif tt == 3:

        perf_A = np.load('{0}\\npy\\db_3\\pred_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_3\\pred_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_3\\pred_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_3\\pred_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_3\\pred_E.npy'.format(os.getcwd()))
        perf_F = np.load('{0}\\npy\\db_3\\pred_F.npy'.format(os.getcwd()))
        perf_G = np.load('{0}\\npy\\db_3\\pred_G.npy'.format(os.getcwd()))

    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])
    E = np.asarray(perf_E[:][:])
    F = np.asarray(perf_F[:][:])
    G = np.asarray(perf_G[:][:])

    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()
    EE = E[:][:].transpose()
    FF = F[:][:].transpose()
    GG = G[:][:].transpose()

    return [AA, BB, CC, DD, EE, FF, GG]

# --------------- LOADING THE NPY OF PERF_ANALYSIS ----------------- #


def load_performance_value_saved_Algo_Analysis(tt):
    if tt == 1:
        perf_A = np.load('{0}\\npy\\db_1\\recom_perf_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_1\\recom_perf_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_1\\recom_perf_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_1\\recom_perf_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_1\\recom_perf_E.npy'.format(os.getcwd()))

    elif tt == 2:
        perf_A = np.load('{0}\\npy\\db_2\\recom_perf_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_2\\recom_perf_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_2\\recom_perf_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_2\\recom_perf_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_2\\recom_perf_E.npy'.format(os.getcwd()))

    elif tt == 3:
        perf_A = np.load('{0}\\npy\\db_3\\recom_perf_A.npy'.format(os.getcwd()))
        perf_B = np.load('{0}\\npy\\db_3\\recom_perf_B.npy'.format(os.getcwd()))
        perf_C = np.load('{0}\\npy\\db_3\\recom_perf_C.npy'.format(os.getcwd()))
        perf_D = np.load('{0}\\npy\\db_3\\recom_perf_D.npy'.format(os.getcwd()))
        perf_E = np.load('{0}\\npy\\db_3\\recom_perf_E.npy'.format(os.getcwd()))

    A = np.asarray(perf_A[:][:])
    B = np.asarray(perf_B[:][:])
    C = np.asarray(perf_C[:][:])
    D = np.asarray(perf_D[:][:])
    E = np.asarray(perf_E[:][:])

    AA = A[:][:].transpose()
    BB = B[:][:].transpose()
    CC = C[:][:].transpose()
    DD = D[:][:].transpose()
    EE = E[:][:].transpose()

    return [AA, BB, CC, DD, EE]


def Main_comp_val_acc_sen_spe1(A, B, C, D, E, F, G, tt):
    VALLL = np.column_stack((A[0], B[0], C[0], D[0], E[0], F[0], G[0]))
    perf1 = VALLL
    # perf1 = Perf_est_all_final(VALLL)
    VALLL = np.column_stack((A[1], B[1], C[1], D[1], E[1], F[1], G[1]))
    perf2 = VALLL
    # # perf2 = Perf_est_all_final(VALLL)

    VALLL = np.column_stack((A[2], B[2], C[2], D[2], E[2], F[2], G[2]))
    perf3 = VALLL

    return [perf1, perf2, perf3]


def Main_comp_val_acc_sen_spe2(A, B, C, D, E, F, G, tt):
    VALLL = np.column_stack((A[0], B[0], C[0], D[0], E[0], F[0], G[0]))
    perf1 = VALLL
    # perf1 = Perf_est_all_final(VALLL)
    VALLL = np.column_stack((A[1], B[1], C[1], D[1], E[1], F[1], G[1]))
    perf2 = VALLL
    # # perf2 = Perf_est_all_final(VALLL)

    VALLL = np.column_stack((A[2], B[2], C[2], D[2], E[2], F[2], G[2]))
    perf3 = VALLL

    return [perf1, perf2, perf3]


def Main_perf_val_acc_sen_spe(A, B, C, D, E, tt):
    VALLL = np.column_stack((A[0], B[0], C[0], D[0], E[0]))
    perf1 = VALLL
    # perf1 = Perf_est_all_final(VALLL)
    # perf1 = VALLL.transpose()
    VALLL = np.column_stack((A[1], B[1], C[1], D[1], E[1]))
    perf2 = VALLL
    # perf2 = Perf_est_all_final(VALLL)
    # perf2 = VALLL.transpose()
    VALLL = np.column_stack((A[2], B[2], C[2], D[2], E[2]))
    perf3 = VALLL
    # perf3 = Perf_est_all_final(VALLL)
    # perf3 = VALLL.transpose()
    # return [np.flipud(perf1), np.flipud(perf2), np.flipud(perf3)]
    # VALLL = np.column_stack((A[3], B[3], C[3], D[3], E[3]))
    # perf4 = VALLL
    # VALLL = np.column_stack((A[4], B[4], C[4], D[4], E[4]))
    # perf5 = VALLL
    # VALLL = np.column_stack((A[5], B[5], C[5], D[5], E[5]))
    # perf6 = VALLL

    return [perf1, perf2, perf3]
        #  perf4, perf5, perf6]


def Complete_Figure(perf, val, str_1, xlab, ylab, tt):
    perf = perf
    perf = (-np.sort(-perf)).T
    if tt==3:
        perf = np.fliplr(perf)
    else:
        perf = perf
    # perf[perf > 100] = 100
    # perf = np.array(perf)

    db_name = ""
    if tt == 1:
        db_name = 'db_1'
    elif tt == 2:
        db_name = 'db_2'
    elif tt == 3:
        db_name = 'db_3'

    # --------------------------------SAVE_CSV------------------------------------- #
    np.savetxt('Results\\' + db_name + '\\' + 'Comp_Analysisi\\' + str(val) + '_' + str(tt) + '_' + 'Graph.csv', perf, delimiter=",")
    # -------------------------------BAR_PLOT-------------------------------------- #
    n_groups = 6
    index = np.arange(n_groups)
    bar_width = 0.10
    opacity = 0.8
    plt.bar(index, perf[0][:], bar_width, alpha=opacity, edgecolor='black', color='b', label=str_1[0][:])
    plt.bar(index+bar_width, perf[1][:], bar_width, alpha=opacity, edgecolor='black', color='g', label=str_1[1][:])
    plt.bar(index+2*bar_width, perf[2][:], bar_width, alpha=opacity, edgecolor='black', color='black', label=str_1[2][:])
    plt.bar(index+3*bar_width, perf[3][:], bar_width, alpha=opacity, edgecolor='black', color='r', label=str_1[3][:])
    plt.bar(index+4*bar_width, perf[4][:], bar_width, alpha=opacity, edgecolor='black', color='yellow', label=str_1[4][:])
    plt.bar(index+5*bar_width, perf[5][:], bar_width, alpha=opacity, edgecolor='black', color='violet', label=str_1[5][:])
    plt.bar(index+6*bar_width, perf[6][:], bar_width, alpha=opacity, edgecolor='black', color='orange', label=str_1[6][:])

    plt.title("User Recommendation System")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(index + bar_width, ('40', '50', '60', '70', '80', '90'))
    plt.legend(loc='lower left')
    plt.savefig('Results\\' + db_name + '\\' + 'Comp_Analysisi\\' + str(val) + '_' + str(tt) + '_' + 'Graph.png', dpi=800)
    plt.show()
    plt.clf()


def Complete_Figure1(perf, val, str_1, xlab, ylab, tt):
    perf = perf
    perf = (-np.sort(-perf)).T

    db_name = ""
    if tt == 1:
        db_name = 'db_1'
    elif tt == 2:
        db_name = 'db_2'
    elif tt == 3:
        db_name = 'db_3'

    # --------------------------------SAVE_CSV------------------------------------- #
    np.savetxt('Results\\' + db_name + '\\' + 'Comp_Analysis_Rating\\' + str(val) + '_' + str(tt) + '_' + 'Graph.csv', perf, delimiter=",")
    # -------------------------------BAR_PLOT-------------------------------------- #
    n_groups = 5
    index = np.arange(n_groups)
    bar_width = 0.10
    opacity = 0.8
    plt.bar(index, perf[0][:], bar_width, alpha=opacity, edgecolor='black', color='b', label=str_1[0][:])
    plt.bar(index+bar_width, perf[1][:], bar_width, alpha=opacity, edgecolor='black', color='g', label=str_1[1][:])
    plt.bar(index+2*bar_width, perf[2][:], bar_width, alpha=opacity, edgecolor='black', color='black', label=str_1[2][:])
    plt.bar(index+3*bar_width, perf[3][:], bar_width, alpha=opacity, edgecolor='black', color='r', label=str_1[3][:])
    plt.bar(index+4*bar_width, perf[4][:], bar_width, alpha=opacity, edgecolor='black', color='yellow', label=str_1[4][:])
    plt.bar(index+5*bar_width, perf[5][:], bar_width, alpha=opacity, edgecolor='black', color='violet', label=str_1[5][:])
    plt.bar(index+6*bar_width, perf[6][:], bar_width, alpha=opacity, edgecolor='black', color='orange', label=str_1[6][:])

    plt.title("User Recommendation System")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(index + bar_width, ('Rating-(0-1)', 'Rating-(1-2)', 'Rating-(2-3)', 'Rating-(3-4)', 'Rating-(4-5)'))
    plt.legend(loc='lower left')
    plt.savefig('Results\\' + db_name + '\\' + 'Comp_Analysis_Rating\\' + str(val) + '_' + str(tt) + '_' + 'Graph.png', dpi=800)
    plt.show()
    plt.clf()


def Complete_Figure2(perf, val, str_1, xlab, ylab, tt):
    perf = perf
    perf = (-np.sort(-perf)).T

    db_name = ""
    if tt == 1:
        db_name = 'db_1'
    elif tt == 2:
        db_name = 'db_2'
    elif tt == 3:
        db_name = 'db_3'

    # --------------------------------SAVE_CSV------------------------------------- #
    np.savetxt('Results\\' + db_name + '\\' + 'Rating_Analysis\\' + str(val) + '_' + str(tt) + '_' + 'Graph.csv', perf, delimiter=",")
    # -------------------------------BAR_PLOT-------------------------------------- #
    n_groups = 5
    index = np.arange(n_groups)
    bar_width = 0.10
    opacity = 0.8
    plt.bar(index, perf[0][:], bar_width, alpha=opacity, edgecolor='black', color='b', label=str_1[0][:])
    plt.bar(index+bar_width, perf[1][:], bar_width, alpha=opacity, edgecolor='black', color='g', label=str_1[1][:])
    plt.bar(index+2*bar_width, perf[2][:], bar_width, alpha=opacity, edgecolor='black', color='black', label=str_1[2][:])
    plt.bar(index+3*bar_width, perf[3][:], bar_width, alpha=opacity, edgecolor='black', color='r', label=str_1[3][:])
    plt.bar(index+4*bar_width, perf[4][:], bar_width, alpha=opacity, edgecolor='black', color='yellow', label=str_1[4][:])
    plt.bar(index+5*bar_width, perf[5][:], bar_width, alpha=opacity, edgecolor='black', color='violet', label=str_1[5][:])
    plt.bar(index+6*bar_width, perf[6][:], bar_width, alpha=opacity, edgecolor='black', color='orange', label=str_1[6][:])

    plt.title("User Recommendation System")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(index + bar_width, ('Rating-(0-1)', 'Rating-(1-2)', 'Rating-(2-3)', 'Rating-(3-4)', 'Rating-(4-5)'))
    plt.legend(loc='lower left')
    plt.savefig('Results\\' + db_name + '\\' + 'Rating_Analysis\\' + str(val) + '_' + str(tt) + '_' + 'Graph.png', dpi=800)
    plt.show()
    plt.clf()

def Comp_plot_Complete_final(tt, ii):
    [A, B, C, D, E, F, G] = load_perf_value_saved_Algo_Analysis(tt)
    [Perf_1, Perf_2, Perf_3] = Main_comp_val_acc_sen_spe(A, B, C, D, E, F, G, tt)
    str_1 = ["User_based_model", "Item_based_model", "ACN3F Model", "Uter Model", "Uter, Bat optimization", "Uter Model, Cat optimization", "Proposed hybrid optimization based Uter"]

    xlab = "Training Percentage(%)"

    ylab = "Mean Absolute Error"
    Complete_Figure(Perf_1, ii, str_1, xlab, ylab, tt)
    ylab = "Mean Square Error"
    ii = ii + 1
    Complete_Figure(Perf_2, ii, str_1, xlab, ylab, tt)
    ylab = "Root Mean Square Error"
    ii = ii + 1
    Complete_Figure(Perf_3, ii, str_1, xlab, ylab, tt)

# ----------------------PERFORMANCE PLOT ------------------- #


def Complete_Figure3(perf, val, str_1, xlab, ylab, tt):
    perf = perf
    perf = np.sort(perf).T
    # perf[perf > 100] = 100
    # perf = np.array(perf)

    db_name = ""
    if tt == 1:
        db_name = 'db_1'
    elif tt == 2:
        db_name = 'db_2'
    elif tt == 3:
        db_name = 'db_3'

    # --------------------------------SAVE_CSV------------------------------------- #
    np.savetxt('Results\\' + db_name + '\\' + 'Perf_analysis\\' + str(val) + '_' + str(tt) + '_' + 'Graph.csv', perf, delimiter=",")
    # -------------------------------BAR_PLOT-------------------------------------- #
    n_groups = 6
    index = np.arange(n_groups)
    bar_width = 0.09
    opacity = 0.8
    plt.bar(index, perf[0][:], bar_width, alpha=opacity, edgecolor='black', color='b', label=str_1[0][:])
    plt.bar(index+bar_width, perf[1][:], bar_width, alpha=opacity, edgecolor='black', color='g', label=str_1[1][:])
    plt.bar(index+2*bar_width, perf[2][:], bar_width, alpha=opacity, edgecolor='black', color='black', label=str_1[2][:])
    plt.bar(index+3*bar_width, perf[3][:], bar_width, alpha=opacity, edgecolor='black', color='r', label=str_1[3][:])
    plt.bar(index+4*bar_width, perf[4][:], bar_width, alpha=opacity, edgecolor='black', color='orange', label=str_1[4][:])

    plt.title("User Recommendation System")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(index + bar_width, ('40', '50', '60', '70', '80', '90'))
    plt.legend(loc='lower left')
    plt.savefig('Results\\' + db_name + '\\' + 'Perf_analysis\\' + str(val) + '_' + str(tt) + '_' + 'Graph.png', dpi=800)
    plt.show()
    plt.clf()


def Perf_plot_Complete_final(tt, ii):
    [A, B, C, D, E] = load_performance_value_saved_Algo_Analysis(tt)
    [Perf_1, Perf_2, Perf_3] = Main_perf_val_acc_sen_spe(A, B, C, D, E, tt)
    str_1 = ["Proposed hybrid optimization based Uter at Epoch = 10", "Proposed hybrid optimization based Uter at Epoch = 20", "Proposed hybrid optimization based Uter at Epoch = 30", "Proposed hybrid optimization based Uter at Epoch = 40", "Proposed hybrid optimization based Uter at Epoch = 50"]

    xlab = "Training Percentage(%)"

    ylab = "Mean Absolute Error"
    Complete_Figure3(Perf_1, ii, str_1, xlab, ylab, tt)
    ylab = "Mean Square Error"
    ii = ii + 1
    Complete_Figure3(Perf_2, ii, str_1, xlab, ylab, tt)
    ylab = "Root Mean Square Error"
    ii = ii + 1
    Complete_Figure3(Perf_3, ii, str_1, xlab, ylab, tt)
    # ylab = "Precesion"
    # ii = ii + 1
    # Complete_Figure3(Perf_4, ii, str_1, xlab, ylab, tt)
    # ylab = "Sensitivity"
    # ii = ii + 1
    # Complete_Figure3(Perf_5, ii, str_1, xlab, ylab, tt)
    # ylab = "Specificity"
    # ii = ii + 1
    # Complete_Figure3(Perf_6, ii, str_1, xlab, ylab, tt)


def Comp_plot_Rating_Complete_final(tt,ii):
    [A, B, C, D, E, F, G] = load_perf_value_saved_Algo_Analysis1(tt)
    [Perf_1, Perf_2, Perf_3] = Main_comp_val_acc_sen_spe1(A, B, C, D, E, F, G, tt)
    str_1 = ["User_based_model", "Item_based_model", "ACN3F Model", "Uter Model", "Uter, Bat optimization",
             "Uter Model, Cat optimization", "Proposed hybrid optimization based Uter"]

    xlab = ""

    ylab = "Mean Absolute Error"
    Complete_Figure1(Perf_1, ii, str_1, xlab, ylab, tt)
    ylab = "Mean Square Error"
    ii = ii + 1
    Complete_Figure1(Perf_2, ii, str_1, xlab, ylab, tt)
    ylab = "Root Mean Square Error"
    ii = ii + 1
    Complete_Figure1(Perf_3, ii, str_1, xlab, ylab, tt)


def org_plot_Rating_Complete_final(tt,ii):
    [A, B, C, D, E, F, G] = load_perf_value_saved_Algo_Analysis2(tt)
    [Perf_1, Perf_2, Perf_3] = Main_comp_val_acc_sen_spe2(A, B, C, D, E, F, G, tt)
    str_1 = ["User_based_model", "Item_based_model", "ACN3F Model", "Uter Model", "Uter, Bat optimization",
             "Uter Model, Cat optimization", "Proposed hybrid optimization based Uter"]

    xlab = ""

    ylab = "Predicted Rating "
    Complete_Figure2(Perf_1, ii, str_1, xlab, ylab, tt)
    ylab = "Real Rating"
    ii = ii + 1
    Complete_Figure2(Perf_2, ii, str_1, xlab, ylab, tt)
    # ylab = "Root Mean Square Error"
    # ii = ii + 1
    # # Complete_Figure1(Perf_3, ii, str_1, xlab, ylab, tt)
