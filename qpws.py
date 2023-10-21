import copy
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import pls_da_new
import threading
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# thread lock
lock = threading.Lock()

THREADSIZE = 11
GLOBALWAVENUM = [[] for _ in range(THREADSIZE)]
GLOBALACC = [[] for _ in range(THREADSIZE)]
GLOBALBESTACC = -1
GLOBALBESTPID = -1
GLOBALBESTLENS = -1
GLOBALBESTWAVE = []


def PC_Validation(X, y, pc, cv):
    t = np.arange(0, pc)
    if len(t) > 10:
        t = t[2::5]

    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True)
    ACCURACY = []
    rindex = 0
    for i in t:
        ACC = []
        for train_index, test_index in stratified_kfold.split(X, y):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            pls = PLSRegression(n_components=i + 1)
            pls.fit(x_train, pd.get_dummies(y_train))
            y_predict = pls.predict(x_test)
            y_predict = np.array([np.argmax(i) for i in y_predict])
            y_predict_train = pls.predict(x_train)
            y_predict_train = np.array([np.argmax(i) for i in y_predict_train])
            y_predict = np.concatenate((y_predict, y_predict_train))
            y_test = np.concatenate((y_test, y_train))
            ACC.append(accuracy_score(y_test, y_predict))
        ACCMEARN = np.mean(ACC)
        if len(ACCURACY) == 0 or ACCMEARN > np.max(ACCURACY):
            rindex = i
        ACCURACY.append(ACCMEARN)
    return np.mean(ACCURACY), rindex


def listLastMaxData(arr):
    max_value = np.max(arr)
    last_max_index = len(arr) - np.where(arr[::-1] == max_value)[0][0] - 1
    last_max_value = arr[last_max_index]
    return last_max_index, last_max_value


def fit(pid, x, y, comp):
    card = []
    card_num = []
    X = copy.deepcopy(x)
    ranges = 0.7

    ACCLIST = []
    while True:
        if len(card_num) > 2:
            if len(card[-1]) <= 1 or ranges <= 0.09:
                break

            if card_num[-1] == card_num[-2]:
                ranges -= 0.05

        if len(card) != 0:
            X = X[:, card[-1][:card_num[-1]]]

        xcal = []
        ycal = []
        num_folds = 5
        stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True)
        for train_indices, val_indices in stratified_kfold.split(X, y):
            xcal, ycal = X[train_indices], y[train_indices]
            break

        m, n = xcal.shape
        if comp > n:
            comp = n
        pls = PLSRegression(n_components=comp)
        pls.fit(xcal, pd.get_dummies(ycal))
        beta = pls.coef_.T
        b = np.abs(beta)
        b = np.mean(b, axis=0)
        b2 = np.argsort(-b, axis=0)
        b3 = np.sum(b)
        b4 = np.sort(b, axis=0)[::-1]
        card.append(b2)
        sum = 0
        t = 0
        for i in range(len(b4)):
            sum += b4[i]
            if 1.0 * sum / b3 * 1.0 > ranges:
                t = i + 1
                break
        card_num.append(t)
        acc, PCIDX = PC_Validation(xcal, ycal, comp, 5)
        ACCLIST.append(acc)

    t = np.arange(0, x.shape[1])
    card.insert(0, t)
    card = card[:-1]
    card_num.insert(0, x.shape[1])
    card_num = card_num[:-1]

    card_swap = []
    for j, element in enumerate(card):
        if j == 0:
            card_swap = card[j]
            continue
        new_list1 = [card_swap[i] for i in card[j]]

        unindexed_elements = [element for i, element in enumerate(card_swap) if i not in card[j]]
        new_list1.extend(unindexed_elements)
        card_swap = new_list1

    lock.acquire()
    try:
        global GLOBALACC, GLOBALWAVENUM, GLOBALBESTACC, GLOBALBESTPID, GLOBALBESTLENS, GLOBALBESTWAVE
        GLOBALWAVENUM[pid] = card_num
        GLOBALACC[pid] = ACCLIST
        accMaxIdx, accMaxValue = listLastMaxData(ACCLIST)
        if accMaxValue >= GLOBALBESTACC and GLOBALBESTLENS > card_num[accMaxIdx] or GLOBALBESTLENS == -1:
            GLOBALBESTACC = accMaxValue
            GLOBALBESTLENS = card_num[accMaxIdx]
            GLOBALBESTPID = pid
            GLOBALBESTWAVE = card_swap
    finally:
        lock.release()


def worker(i, x, y):
    fit(i, x, y, 35)


def Algorithm(x, y):
    threads = []
    for i in range(THREADSIZE):
        t = threading.Thread(target=worker, args=(i, x, y))
        t.start()
        threads.append(t)

    for i in threads:
        i.join()
    global plt
    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(GLOBALACC)):
        t = np.arange(0, len(GLOBALWAVENUM[i]))
        ax.plot(t, GLOBALWAVENUM[i], GLOBALACC[i], label=f'Thread{i + 1}')

    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Variables')
    ax.set_zlabel('Accuracy')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.legend()
    plt.show()

    plt.subplot(211)
    mx = 0
    for i in range(len(GLOBALWAVENUM)):
        t = np.arange(0, len(GLOBALWAVENUM[i]))
        mx = max(mx, t[-1])
        plt.plot(t, GLOBALWAVENUM[i], label=f'Thread{i + 1}', linewidth=1.5)

    plt = pltSets(plt, 'Number of sampling runs', 'Number of\nsampled variables', mx)

    plt.subplot(212)
    for i in range(len(GLOBALACC)):
        t = np.arange(0, len(GLOBALACC[i]))
        plt.plot(t, GLOBALACC[i], label='Thread {}'.format(i + 1), linewidth=1.5)
    plt = pltSets(plt, 'Number of sampling runs', 'Accuracy', mx)
    plt.show()

    mins = 100000
    for i in range(len(GLOBALACC)):
        mins = min(mins, len(GLOBALACC[i]))

    for i in range(len(GLOBALACC)):
        GLOBALACC[i] = GLOBALACC[i][:mins]
        print(len(GLOBALACC[i]))

    sum_result = np.sum(GLOBALACC, axis=0)
    new_arr = sum_result.flatten()
    new_arr /= THREADSIZE
    t = np.arange(0, len(GLOBALACC[0]))
    plt.plot(t, new_arr, linewidth=1.5)
    plt = pltSets(plt, 'Number of sampling runs', 'Thread Average Accuracy', t[-1])
    plt.show()

    print("best variables num:", GLOBALBESTLENS)
    print("best pid:", GLOBALBESTPID)
    print("best accuracy:", GLOBALBESTACC)
    print("best wave:", GLOBALBESTWAVE[:GLOBALBESTLENS])
    print("best all wave:", GLOBALBESTWAVE)
    # print(GLOBALACC)

    return


def pltSets(plt, xlabel, ylabel, mx):
    fonts = 20
    plt.xticks(fontsize=fonts)
    plt.yticks(fontsize=fonts)
    plt.ylabel(ylabel, fontsize=fonts)
    plt.xlabel(xlabel, fontsize=fonts)
    plt.gca().spines['left'].set_linewidth(2.0)
    plt.gca().spines['bottom'].set_linewidth(2.0)
    plt.gca().spines['right'].set_linewidth(2.0)
    plt.gca().spines['top'].set_linewidth(2.0)
    plt.xlim(left=0, right=mx)
    plt.ylim(bottom=0)

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticklabels([])
    ax3 = ax1.twiny()
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_xticklabels([])

    ax1.tick_params(axis='both', width=2.0, which='both', direction='in', length=8)
    ax2.tick_params(axis='both', width=2.0, which='both', direction='in', length=8)
    ax3.tick_params(axis='both', width=2.0, which='both', direction='in', length=8)

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend()
    return plt


if __name__ == '__main__':
    x, y = pls_da_new.deal_data_all_delete_testing()
    Algorithm(x, y)
