from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np

def data_preprocessing():
    pass


def data_split(X,y,test_size):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 42)

    return X_train, X_test, y_train, y_test


def train_model(model,x,y,epochs=50):

    if isinstance(model, Sequential):

        model.fit(x,y,epochs=epochs)

    else:
        model.fit(x,y)

    return model


def optimize_hyperparam(model,min_accuracy):
    pass

def validation(model,x,y):

    if isinstance(model, Sequential):

        y_pred = model.predict(x)

        y_pred = np.array([y_pred>0.5]).astype(np.int)

        y_pred = y_pred.reshape(len(y))

        return accuracy_score(y,y_pred)

    else:

        return model.score(x,y)

def plot_confusion_matrix(model,x,y_true,train_flag):

    y_pred = model.predict(x)

    if isinstance(model, Sequential):

        y_pred = np.array([y_pred >= 0.5]).astype(np.int)

        y_pred = y_pred.reshape(len(y_true))

    M = confusion_matrix(y_true, y_pred)

    # we normalize the confusion matrix

    M = M.astype('float') / M.sum(axis=1)[:, np.newaxis]

    plt.imshow(M)
    plt.colorbar()

    if train_flag:
        plt.title("Confusion matrix for train dataset")
    else:
        plt.title("Confusion matrix for test dataset")

    plt.show()


def comparation():
    pass
