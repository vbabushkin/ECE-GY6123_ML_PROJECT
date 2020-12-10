#  for plotting some figures
from __future__ import print_function

import pickle

import mat73
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 16})

import matplotlib.pyplot as plt

# data format:

# 80 freq. bins X 160 time X 59 channels X trials
#
# Trials: 34 participants X 20 pictures = 680 trials
# 182   + 225  +  273 = 680
# 181   + 236  +  263 = 680
# 295   + 206  +  179 = 680
# 48    +  76  +  556 = 680


# Importing the data files
class1 = mat73.loadmat('../DATA/PSD_ALL1.mat')
class2 = mat73.loadmat('../DATA/PSD_ALL2.mat')
class3 = mat73.loadmat('../DATA/PSD_ALL3.mat')
class4 = mat73.loadmat('../DATA/PSD_ALL4.mat')
# Taking absolute value of the data (since it is complex)
class1 = abs(class1['PSD_ALL_1'])
class2 = abs(class2['PSD_ALL_2'])
class3 = abs(class3['PSD_ALL_3'])
class4 = abs(class4['PSD_ALL_4'])

print(class1.shape)
print(class2.shape)
print(class3.shape)
print(class4.shape)

# Combine data from all classes and reshape such that trials are in the first dimension
# Take the mean over the time dimension
X = np.concatenate((class1, class2, class3, class4), axis=3)
X = np.transpose(X, [3, 0, 1, 2])
X = np.mean(X, axis=2)

X1 = np.expand_dims(X, 3)

xMat = np.mean(X1, axis=0)
xMat = xMat.squeeze()

print(X1.shape)

## plot spectrogram
hvha = class1
hvha = np.mean(hvha, axis=1)
hvha = np.mean(hvha, axis=2)

hvla = class2
hvla = np.mean(hvla, axis=1)
hvla = np.mean(hvla, axis=2)

lvha = class3
lvha = np.mean(lvha, axis=1)
lvha = np.mean(lvha, axis=2)

lvla = class4
lvla = np.mean(lvla, axis=1)
lvla = np.mean(lvla, axis=2)

fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
axs[0, 0].imshow(hvha, cmap='plasma', aspect=0.5)
axs[0, 0].invert_yaxis()
axs[0, 0].grid()
axs[0, 0].set_ylim([0, 80])
axs[0, 0].set_xlim([0, 59])
axs[0, 0].set_xticks(np.arange(0, hvha.shape[1], 5))
axs[0, 0].set_title('HVHA')

axs[0, 1].imshow(hvla, cmap='plasma', aspect=0.5)
axs[0, 1].invert_yaxis()
axs[0, 1].grid()
axs[0, 1].set_ylim([0, 80])
axs[0, 1].set_xlim([0, 59])
axs[0, 1].set_xticks(np.arange(0, hvla.shape[1], 5))
axs[0, 1].set_title('HVLA')

axs[1, 0].imshow(lvha, cmap='plasma', aspect=0.5)
axs[1, 0].invert_yaxis()
axs[1, 0].grid()
axs[1, 0].set_ylim([0, 80])
axs[1, 0].set_xlim([0, 59])
axs[1, 0].set_xticks(np.arange(0, lvha.shape[1], 5))
axs[1, 0].set_title('LVHA')

im = axs[1, 1].imshow(lvla, cmap='plasma', aspect=0.5)
axs[1, 1].invert_yaxis()
axs[1, 1].grid()
axs[1, 1].set_ylim([0, 80])
axs[1, 1].set_xlim([0, 59])
axs[1, 1].set_xticks(np.arange(0, lvla.shape[1], 5))
axs[1, 1].set_title('LVLA')
fig.text(0.5, 0.03, 'EEG channels', ha='center')
fig.text(0.05, 0.5, 'Frequency, Hz', va='center', rotation='vertical')
cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.75])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('FIGURES/spectrogram_4classes.pdf')

# plot accuracy for SVM
accSVM_4_fn = ('accuracySVM_4.p')
with open(accSVM_4_fn, 'rb') as fp:
    accSVM_4 = pickle.load(fp)

accSVM_9_fn = ('accuracySVM_9.p')
with open(accSVM_9_fn, 'rb') as fp:
    accSVM_9 = pickle.load(fp)

accSVM_12_fn = ('accuracySVM_12.p')
with open(accSVM_12_fn, 'rb') as fp:
    accSVM_12 = pickle.load(fp)

with plt.style.context("default"):
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(1, 10, 10), accSVM_4, linewidth=2, marker='o', label='4 classes')
    plt.plot(np.linspace(1, 10, 10), accSVM_9, linewidth=2, marker='v', label='9 classes')
    plt.plot(np.linspace(1, 10, 10), accSVM_12, linewidth=2, marker='s', label='12 classes')
    plt.xlabel('Folds', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend()
    plt.xlim([0.5, 10.5])
    plt.ylim([0.4, 1.0])
    plt.xticks(np.arange(1, 10.5, step=1))
    plt.grid()
    plt.savefig('FIGURES/accuracy_svm_10fold_all_classes.pdf')

## CNN
## plot train/test accuracy per fold per epoch
# Load accuracy per fold per epoch
accTrain_fn = ('accuracyTrain_4.p')
with open(accTrain_fn, 'rb') as fp:
    accTrain_4 = pickle.load(fp)

accVal_fn = ('accuracyVal_4.p')
with open(accVal_fn, 'rb') as fp:
    accVal_4 = pickle.load(fp)

# Load accuracy per fold per epoch
accTrain_fn = ('accuracyTrain_9.p')
with open(accTrain_fn, 'rb') as fp:
    accTrain_9 = pickle.load(fp)

accVal_fn = ('accuracyVal_9.p')
with open(accVal_fn, 'rb') as fp:
    accVal_9 = pickle.load(fp)

# Load accuracy per fold per epoch
accTrain_fn = ('accuracyTrain_12.p')
with open(accTrain_fn, 'rb') as fp:
    accTrain_12 = pickle.load(fp)

accVal_fn = ('accuracyVal_12.p')
with open(accVal_fn, 'rb') as fp:
    accVal_12 = pickle.load(fp)

with plt.style.context("default"):
    plt.figure(figsize=(10, 5))
    for iplt in range(2):

        plt.subplot(1, 2, iplt + 1)

        if iplt == 0:
            acc4 = np.mean(accTrain_4, axis=0)
            acc9 = np.mean(accTrain_9, axis=0)
            acc12 = np.mean(accTrain_12, axis=0)
        else:
            acc4 = np.mean(accVal_4, axis=0)
            acc9 = np.mean(accVal_9, axis=0)
            acc12 = np.mean(accVal_12, axis=0)

        plt.plot(acc4, '-', linewidth=2, label='4 classes')
        plt.plot(acc9, '-', linewidth=2, label='9 classes')
        plt.plot(acc12, '-', linewidth=2, label='12 classes')

        n = len(acc4)
        nepochs = len(acc4)
        plt.grid()
        plt.xlim([0, nepochs])

        plt.xlabel('Epoch')
        if iplt == 0:
            plt.ylabel('Train accuracy')
        else:
            plt.ylabel('Test accuracy')
        plt.legend(loc='lower left')
        plt.ylim([-0.02, 1.02])
    plt.tight_layout()
    plt.savefig('FIGURES/accuracyTrainVal_cnn_10fold_all_classes.pdf')

# plot mean accuracy for CNN
# Load mean accuracy
accTotal_4_fn = ('accuracyTotal_4.p')
with open(accTotal_4_fn, 'rb') as fp:
    accTotal_4 = pickle.load(fp)

accTotal_9_fn = ('accuracyTotal_9.p')
with open(accTotal_9_fn, 'rb') as fp:
    accTotal_9 = pickle.load(fp)

accTotal_12_fn = ('accuracyTotal_12.p')
with open(accTotal_12_fn, 'rb') as fp:
    accTotal_12 = pickle.load(fp)

with plt.style.context("default"):
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(1, 10, 10), accTotal_4, linewidth=2, marker='o', label='4 classes')
    plt.plot(np.linspace(1, 10, 10), accTotal_9, linewidth=2, marker='v', label='9 classes')
    plt.plot(np.linspace(1, 10, 10), accTotal_12, linewidth=2, marker='s', label='12 classes')
    plt.xlabel('Folds', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend()
    plt.xlim([0.5, 10.5])
    plt.ylim([0.4, 1.0])
    plt.xticks(np.arange(1, 10.5, step=1))
    plt.grid()
    plt.savefig('FIGURES/accuracy_cnn_10fold_all_classes.pdf')

# read the recordings
reportsCNN_4_fn = ('reportsCNN_4.p')
with open(reportsCNN_4_fn, 'rb') as fp:
    reportsCNN_4 = pickle.load(fp)

reportsCNN_9_fn = ('reportsCNN_9.p')
with open(reportsCNN_9_fn, 'rb') as fp:
    reportsCNN_9 = pickle.load(fp)

reportsCNN_12_fn = ('reportsCNN_12.p')
with open(reportsCNN_12_fn, 'rb') as fp:
    reportsCNN_12 = pickle.load(fp)

reportsSVM_4_fn = ('reportsSVM_4.p')
with open(reportsSVM_4_fn, 'rb') as fp:
    reportsSVM_4 = pickle.load(fp)

reportsSVM_9_fn = ('reportsSVM_9.p')
with open(reportsSVM_9_fn, 'rb') as fp:
    reportsSVM_9 = pickle.load(fp)

reportsSVM_12_fn = ('reportsSVM_12.p')
with open(reportsSVM_12_fn, 'rb') as fp:
    reportsSVM_12 = pickle.load(fp)

# print to fill the table in Latex template
reportsDict = reportsCNN_4
numFolds = 10
numClasses = 4
avgPrecision = np.zeros(numClasses)
avgRecall = np.zeros(numClasses)
avgF1 = np.zeros(numClasses)
avgSupport = np.zeros(numClasses)
classes = list(reportsDict[0].keys())[0:numClasses]
for clIdx in range(len(classes)):
    tmpPrecision = []
    tmpRecall = []
    tmpF1 = []
    tmpSupport = []
    for f in range(numFolds):
        tmpPrecision.append(reportsDict[f][classes[clIdx]]['precision'])
        tmpRecall.append(reportsDict[f][classes[clIdx]]['recall'])
        tmpF1.append(reportsDict[f][classes[clIdx]]['f1-score'])
        tmpSupport.append(reportsDict[f][classes[clIdx]]['support'])
    avgPrecision[clIdx] = np.mean(tmpPrecision)
    avgRecall[clIdx] = np.mean(tmpRecall)
    avgF1[clIdx] = np.mean(tmpF1)
    avgSupport[clIdx] = np.mean(tmpSupport)

for i in range(numClasses):
    print('{0:2.2f} '.format(avgF1[i]), end="&")
print()
