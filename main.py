
from models import *
from outils import *
from prepro import *
from sklearn.datasets import load_breast_cancer

if __name__ == "__main__":

    test_size = 0.33

    models_string = ['decision_tree_model','kmeans_model','svm_model']
    #['neural_network', [5, 15, 5, 5, 5], 5, X.shape[1]]]

    path = "data_classification/kidney_disease.csv"

    X,y,label = load_preprocessing_data(path,index_col=0,binar = True) # for kidney_disease
    #X,y = load_preprocessing_data(path,header=None,binar = True) #for banknote

    # load preprocessed data

    X_train, X_test, y_train, y_test = data_split(X,y,test_size)

    models = call_models(models_string)

    for i, model in enumerate(models):

        train_model(model, X_train, y_train)

        accuracy = validation(model, X_test, y_test)

        print(models_string[i], ' accuracy: ', accuracy)

        # we plot the confusion matrix for both the train and test datasets

        y_labels = ['0', '1']

        plot_confusion_matrix(model, X_train, y_train, models_string_dic[models_string[i]], y_labels, train_flag=True)

        plot_confusion_matrix(model, X_test, y_test, models_string_dic[models_string[i]], y_labels, train_flag=False)