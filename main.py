
from models import *
from utils import *
from sklearn.datasets import load_breast_cancer

if __name__ == "__main__":


    # , ['neural_network',[5,15,5,5,5],5]

    test_size = 0.33

    # load data

    X = load_breast_cancer()['data']
    y = load_breast_cancer()['target']

    print(X.shape)

    models_string = ['decision_tree_model','kmeans_model']
    #['neural_network', [5, 15, 5, 5, 5], 5, X.shape[1]]]

    data_preprocessing()

    # load preprocessed data

    X_train, X_test, y_train, y_test = data_split(X,y,test_size)

    models = call_models(models_string)

    for i,model in enumerate(models):

        train_model(model,X_train,y_train)

        accuracy = validation(model,X_test,y_test)

        print(models_string[i], accuracy)

        # we plot the confusion matrix for both the train and test datasets

        plot_confusion_matrix(model,X_train,y_train, train_flag=True)

        plot_confusion_matrix(model, X_test, y_test, train_flag=False)
