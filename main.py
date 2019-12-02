from models import *
from outils import *
from prepro import *
from sklearn.datasets import load_breast_cancer
import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, help='the list of models you want to use',
                        default='svm_model')

    parser.add_argument('--dataset', type=str, default='kidney-disease',
                        help='Dataset to use: either kidney-disease or '
                             'bank-note ')
    test_size = 0.33
    args = parser.parse_args()

    models_string = [item for item in args.models.split(',')]
    if args.dataset == 'kidney-disease':
        path = "data_classification/kidney_disease.csv"
        X, y, label = load_preprocessing_data(path, index_col=0, binar=True)  # for kidney_disease
    elif args.dataset == 'bank-note':
        path = "data_classification/data_banknote_authentication.txt"
        X, y, label = load_preprocessing_data(path, header=None, binar=True)  # for banknote
    else:
        print('dataset not available or misspelled')
        sys.exit(1)
    # models_string.append(['neural_network', [5, 15, 5, 5, 5], 5, X.shape[1]])

    y_labels = [label[0], label[1]]

    # load preprocessed data

    X_train, X_test, y_train, y_test = data_split(X, y, test_size)

    models = call_models(models_string)

    for i, model in enumerate(models):
        train_model(model, X_train, np.ravel(y_train))

        accuracy = validation(model, X_test, y_test)

        print(models_string[i], ' accuracy: ', accuracy)

        # we plot the confusion matrix for both the train and test datasets

        plot_confusion_matrix(model, X_train, y_train, models_string_dic[models_string[i]], y_labels, train_flag=True)

        plot_confusion_matrix(model, X_test, y_test, models_string_dic[models_string[i]], y_labels, train_flag=False)
