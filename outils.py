from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def data_split(X,y,test_size):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

    return X_train, X_test, y_train, y_test


def train_model(model,x,y,epochs=50):

    if isinstance(model, Sequential):

        model.fit(x,y,epochs=epochs)

    else:
        model.fit(x,y)

    return model



def validation(model,x,y):

    if isinstance(model, Sequential):

        y_pred = model.predict(x)

        y_pred = np.array([y_pred>0.5]).astype(np.int)

        y_pred = y_pred.reshape(len(y))

        return accuracy_score(y, y_pred)

    else:

        return model.score(x, y)


def plot_confusion_matrix(model, x, y_true, model_string, cm_labels, train_flag):

    y_pred = model.predict(x)

    if isinstance(model, Sequential):

        y_pred = np.array([y_pred >= 0.5]).astype(np.int)

        y_pred = y_pred.reshape(len(y_true))

    cm = confusion_matrix(y_true, y_pred)

    # we normalize the confusion matrix

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax = plt.subplot()
    sns.heatmap(np.array(cm), annot=True, ax=ax)

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')


    ax.xaxis.set_ticklabels(cm_labels)
    ax.yaxis.set_ticklabels(cm_labels)

    if train_flag:
        ax.set_title(model_string + ": confusion matrix for train set")
    else:
        ax.set_title(model_string + ": confusion matrix for test set")

    plt.show()


def comparation():
    pass


def optimize_hyperparam(model,min_accuracy):
    pass

models_string_dic = {
    'decision_tree_model': 'Decision tree',
    'kmeans_model': 'K-means',
    'svm_model': 'SVM',
    'neural_network': 'Neural network',
}