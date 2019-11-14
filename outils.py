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


def plot_confusion_matrix(model, x, y_true, model_string, cm_labels, train_flag, image_folders="Images"):

    y_pred = model.predict(x)

    if isinstance(model, Sequential):

        y_pred = np.array([y_pred >= 0.5]).astype(np.int)

        y_pred = y_pred.reshape(len(y_true))

    cm = confusion_matrix(y_true, y_pred)

    # we normalize the confusion matrix

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()

    # we plot

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    fmt = '.2f'
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j],fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    if train_flag:
        title = model_string + ": confusion matrix for train set"
        fig_name = model_string + '-cm-train.png'
    else:
        title = model_string + ": confusion matrix for test set"
        fig_name = model_string + '-cm-test.png'

    labels = ['', '', cm_labels[0], '', '', '', cm_labels[1]]

    ax.set(#xticks=np.arange(cm.shape[1]),
           #yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.tight_layout()

    plt.savefig(image_folders + "/" + fig_name)
    plt.close()


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