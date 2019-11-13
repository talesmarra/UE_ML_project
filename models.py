import keras
import sklearn
from keras.layers import Dense
from sklearn import tree, svm


def svm_model():
    """
    :return: instance of the model
    """
    return svm.SVC(gamma='scale')


def kmeans_model():
    """
    :return: instance of the model
    """
    return sklearn.cluster.KMeans(n_clusters=2, random_state=0)


def decision_tree_model():
    """
    :return: instance of the model
    """
    return tree.DecisionTreeClassifier()


def neural_network(n_neurons_per_layer, n_layers, input_dim):
    """
    :param n_layers : (int) number of layers for the neural network
    :param n_neurons_per_layer: (list[int]) how many neurons per layer of the network
    :param input_dim: (int) dimension of input
    :return: instance of the model
    """
    if len(n_neurons_per_layer) != n_layers:
        print('number of layers of network not match number of neurons per layer')
    else:
        classifier = keras.Sequential()
        # Input layer
        classifier.add(
            Dense(n_neurons_per_layer[0], activation='relu', kernel_initializer='random_normal',
                  input_dim=input_dim))
        for n in n_neurons_per_layer[1:]:
            # Hidden Layers
            classifier.add(Dense(n, activation='relu', kernel_initializer='random_normal'))

            # Output Layer
        classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

        # Compiling the neural network
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return classifier


def call_models(list_models):
    """
    :param: list_models: the names of the models you want to use
    :type list_models: list
    """
    list_of_models = list()
    for model in list_models:
        if type(model) == str:
            if model == 'svm_model':
                list_of_models.append(svm_model())
            elif model == 'kmeans_model':
                list_of_models.append(kmeans_model())
            elif model == 'decision_tree_model':
                list_of_models.append(decision_tree_model())
        else:
            n_neurons_per_layer = model[1]
            n_layers = model[2]
            input_dim = model[3]
            try:
                list_of_models.append(neural_network(n_neurons_per_layer, n_layers, input_dim))
            except:
                print('Not loaded neural network')
                continue
    return list_of_models
