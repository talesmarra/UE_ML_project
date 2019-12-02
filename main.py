from models import *
from utilities import *
from prepro import *
from sklearn.datasets import load_breast_cancer
import argparse
import sys
import os

if __name__ == "__main__":

    # General parameters

    test_size = 0.33

    output_folder = 'Output'

    accs_file_path = "Output/accuracies_file.txt"

    # Parse some parameters (dataset and models to use)

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, help='the list of models you want to use',
                        default='svm_model')

    parser.add_argument('--dataset', type=str, default='kidney-disease',
                        help='Dataset to use: either kidney-disease or '
                             'bank-note ')
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

    y_labels = [label[0], label[1]]

    # load preprocessed data

    X_train, X_test, y_train, y_test = data_split(X, y, test_size)

    models = call_models(models_string)

    # create the output files

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(output_folder+'/Images')

    accs_file = create_accs_file(accs_file_path, args.dataset)

    for i, model in enumerate(models):

        train_model(model, X_train, np.ravel(y_train))

        accuracy = validation(model, X_test, y_test)

        print(models_string[i], ' accuracy: ', accuracy)

        print_acc_2_file(accs_file, models_string[i], accuracy)

        # we plot the confusion matrix for both the train and test datasets

        plot_confusion_matrix(model, X_train, y_train, models_string_dic[models_string[i]], y_labels, train_flag=True)

        plot_confusion_matrix(model, X_test, y_test, models_string_dic[models_string[i]], y_labels, train_flag=False)

    accs_file.close()