from data_parser import parseData
import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC, NuSVC
from itertools import chain
from itertools import combinations


def main():
    
    data, res, merlins, mostCorrects, percivals, vts, human_shots = parseData()
    res = np.array(res)
    merlins = np.array(merlins)
    print(res[0])
    print(merlins[0])
    print(percivals[0])
    print(vts[0])
    D = data
    N = len(D)
    num_features = data[0].shape[1]
    for feat_idxs in powerset(range(num_features)):
        if 0 not in feat_idxs or 1 not in feat_idxs or 2 not in feat_idxs:
            continue
        print(feat_idxs)

        data = []
        for dat in D:
            data.append( dat[:, feat_idxs] )
        data = np.array(data)
        
        K = 10
        train_accs = []
        test_accs = []
        for k in range(K):
            verbose = 0
            if k == K-1 and len(feat_idxs) == num_features:
                verbose = 1
            
            test_idxs = list(range(int(1. * k * N / 10), int( (1. + k) * N / 10 ) ))
            train_idxs = []
            for i in range(N):
                if i not in test_idxs:
                    train_idxs.append(i)
            #print(test_idxs)
            #print(train_idxs)

            #num_train = int(len(merlins) * .9)

            res_train = res[train_idxs]
            res_test = res[test_idxs]
            
            data_train = np.array(data[train_idxs])
            data_test = np.array(data[test_idxs])
            X = data_train.reshape( (data_train.shape[0], -1) )
            X_test = data_test.reshape( (data_test.shape[0], -1) )
            
            merlins_test = merlins[test_idxs]
            Y = merlins[train_idxs]
            Y_test = merlins_test
            merlin_clf, train_acc, test_acc = train_svm(X,Y,X_test,Y_test,res_train,res_test,human_shots,verbose)
            
            train_accs.append(train_acc)
            test_accs.append(test_acc)

        avg_train = sum(train_accs) / K
        avg_test= sum(test_accs) / K
        print("training accuracy: " + str(avg_train))
        print("test accuracy: " + str(avg_test))
        print("--------------------------")

        #percivals_train = percivals[:num_train]
        #percivals_test = percivals[num_train:]
        #percy_clf = train_svm(X,percivals_train,X_test,percivals_test,res_train,res_test)

        #vts_train = vts[:num_train]
        #vts_test = vts[num_train:]
        #vt_clf = train_svm(X,vts_train,X_test,vts_test,res_train,res_test)



    print("correct votes: ")
    acc = getAccuracy(mostCorrects, merlins, res, human_shots)
    print(acc)
    print("-------")

    print("human accuracy: ")
    print(1. * sum(human_shots) / len(human_shots))
    print("-------")

def train_svm(X,Y,X_test,Y_test,res_train,res_test, human_labels, verbose=0):
    lin_clf = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=200000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.00001,
                   verbose=0)
    #lin_clf = NuSVC(gamma='auto')
    lin_clf.fit(X, Y) 

    #print(X_test.shape)
    #print(len(Y))
    dec = lin_clf.decision_function(X_test)
    dec_train = lin_clf.decision_function(X)
    #print("test accuracy:")
    test_acc = getAccuracy(dec, Y_test, res_test, human_labels, X_test, verbose)
    #print("------")
    #print("train accuracy:")
    train_acc = getAccuracy(dec_train, Y, res_train, human_labels, X)
    #print("------")

    return lin_clf, train_acc, test_acc

def getAccuracy(predict, actual, possible, human_labels, X=None, verbose=0):
    correct = 0
    spy_shot = 0
    human_not_us = 0
    us_not_human = 0
    human_and_us = 0
    neither = 0
    for i in range(len(actual)):
        y = actual[i]
        v = argmaxRes(predict[i], possible[i])
        v_raw = np.argmax(predict[i])
        if v_raw not in possible[i]:
            spy_shot += 1
        if y == v:
            correct += 1
            if human_labels[i] == 1:
                human_and_us += 1
            else:
                us_not_human += 1
        else:
            if verbose > 1:
                print(X[i].reshape( (5, -1)))
                print(predict[i])
                print(actual[i])
            if human_labels[i] == 1:
                human_not_us += 1
            else:
                neither += 1
    acc = 1. * correct / len(actual)
    if verbose==1:
        print("num spy shots: " + str(spy_shot))
        print("num correct: " + str(correct))
        print("total: " + str(len(actual)))
        print("accuracy: " + str(acc))
        print("num human and clf correct: " + str(human_and_us))
        print("num human and not clf correct: " + str(human_not_us))
        print("num not human and clf correct: " + str(us_not_human))
        print("num not human and not clf correct: " + str(neither))
    return acc

def getAccuracyAll(predict, actual, possible, percivals, vts):
    correct = 0
    spy_shot = 0
    percy_shot = 0
    vt_shot = 0
    for i in range(len(actual)):
        y = actual[i]
        v = argmaxRes(predict[i], possible[i])
        v_raw = np.argmax(predict[i])
        if v_raw not in possible[i]:
            spy_shot += 1
        if y == v:
            correct += 1
        if v_raw == percivals[i]:
            percy_shot += 1
        if v_raw == vts[i]:
            vt_shot += 1
    acc = 1. * correct / len(actual)
    print("num spy shots: " + str(spy_shot))
    print("num percy shots: " + str(percy_shot))
    print("num vt shots: " + str(vt_shot))
    print("num correct: " + str(correct))
    print("total: " + str(len(actual)))
    print("accuracy: " + str(acc))
    return acc

def argmaxRes(vec, poss):
    max_idx = 0 
    max_v = -999999
    for i,v in enumerate(vec):
        if i in poss and v > max_v:
            max_idx = i
            max_v = v
    return max_idx

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

if __name__ == "__main__":
    main()
