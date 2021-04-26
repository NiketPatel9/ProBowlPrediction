"""
This module is responsible for creating various machine learning model workflows, including model
creation, fitting, and evaluation.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean
from numpy import std
import itertools
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pickle


RANDOM_SEED = 42
CLASSES = ["Non-Pro Bowler", "Pro Bowler"]


def plot_roc(y_true, y_preds, title):
    """
    This function plots a simple ROC curve based on the baseline predictions from the training data,
    which should be perfect, and the new probabilities.

    :param y_true: The true test labels
    :param y_preds: The predicted probabilities of the testing data
    :param title: The title of the plotted ROC curve
    :return: void
    """
    model_fpr, model_tpr, _ = roc_curve(y_true, y_preds)
    roc_auc = auc(model_fpr, model_tpr)
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot([0, 1], [0, 1], 'k--', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label=f'Model ROC (area = {roc_auc:0.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{title} ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"visualizations/{title.replace(' ', '')}_roc.png")
    plt.clf()


def plot_matrix(true, preds, title):
    """
    This function takes in two vectors and plots a confusion matrix based on them.

    :param true: These are the true labels.
    :param preds: These are the predicted labels
    :param title: This is the title of the confusion matrix
    :return: void
    """
    cm = confusion_matrix(true, preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{title} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES)
    plt.yticks(tick_marks, CLASSES)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f"visualizations/{title.replace(' ', '')}_confusion_matrix.png")
    plt.clf()


def random_forest_experiment(X_train, X_test, y_train, y_test):
    """
    This function runs the Random Forest Binary Classification and evaluates model performance.

    :param X_train: The training features
    :param X_test: The testing features
    :param y_train: The training labels
    :param y_test: The testing labels
    :return: void
    """
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, max_features="sqrt",
                                   n_jobs=-1, verbose=1)

    model.fit(X_train, y_train)

    pred_probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("Classification Report and Metrics for Random Forest Classifier")
    print(classification_report(y_test, preds))

    plot_roc(y_test, pred_probs, "Random Forest")
    plot_matrix(y_test, preds, "Random Forest")


def logistic_regression_experiment(X_train, X_test, y_train, y_test):
    """
    This function runs the logistic regression workflow, including evaluation via metrics, ROC, and
    confusion matrix.
    :param X_train: The training features
    :param X_test: The testing features
    :param y_train: The training labels
    :param y_test: The testing labels
    :return: void
    """
    model = LogisticRegression(solver="liblinear", random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    pred_probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("Classification Report and Metrics for Logistic Regression Model")
    print(classification_report(y_test, preds))

    plot_roc(y_test, pred_probs, "Logistic Regression")
    plot_matrix(y_test, preds, "Logistic Regression")
    pickle.dump(model, open("lrm.sav", 'wb'))


def neural_network_experiment(X_train, X_test, y_train, y_test):
    """
    This function runs the neural network workflow, including evaluation via metrics, ROC, and
    confusion matrix.
    :param X_train: The training features
    :param X_test: The testing features
    :param y_train: The training labels
    :param y_test: The testing labels
    :return: void
    """

    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=len(X_train.columns)))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, validation_split=0.1, epochs=35, batch_size=10, verbose=1)

    plt.plot(model.history.history['accuracy'])
    plt.plot(model.history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f"visualizations/neuralnet_accuracy.png")
    plt.clf()

    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f"visualizations/neuralnet_loss.png")
    plt.clf()

    # pred_probs = model.predict_proba(X_test)[:, 1]
    pred_probs = model.predict(X_test)
    preds = np.round(pred_probs, 0)

    print("Classification Report and Metrics for Neural Network Model")
    print(classification_report(y_test, preds))

    plot_roc(y_test, pred_probs, "Neural Network")
    plot_matrix(y_test, preds, "Neural Network")

    model.save("neural_net.h5")


def k_fold_experiment(X, y):
    """
    This function runs the k-fold cross validation using the given dataset on three different classification models.

    :param X: This is the full set of predictors
    :param y: This is the full set of outcome variables
    :return: void
    """
    # Build models
    forest_classifier = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, max_features="sqrt",
                                               n_jobs=-1, verbose=1)
    regression_model = LogisticRegression(solver="liblinear", random_state=RANDOM_SEED)

    neural_network = Sequential()
    neural_network.add(Dense(256, activation='relu', input_dim=len(X.columns)))
    neural_network.add(Dropout(0.4))
    neural_network.add(Dense(128, activation='relu'))
    neural_network.add(Dropout(0.3))
    neural_network.add(Dense(64, activation='relu'))
    neural_network.add(Dropout(0.2))
    neural_network.add(Dense(32, activation='relu'))
    neural_network.add(Dropout(0.1))
    neural_network.add(Dense(1, activation='sigmoid'))
    neural_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    cv = KFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)

    # Random Forest and Regression k-fold is trivial
    scores = cross_val_score(forest_classifier, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print(f'Random Forest Accuracy: {mean(scores):.3f} with standard deviation of {std(scores):.3f}')

    scores = cross_val_score(regression_model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print(f'Logistic Regression Accuracy: {mean(scores):.3f} with standard deviation of {std(scores):.3f}')

    # K-fold with neural network needs extra work
    cvscores = []
    for train, test in cv.split(X, y):
        neural_network.fit(X.iloc[train], y.iloc[train], validation_split=0.1, epochs=35, batch_size=10, verbose=1)
        scores = neural_network.evaluate(X.iloc[test], y.iloc[test], verbose=0)
        cvscores.append(scores[1])

    print(f'Neural Network  Accuracy: {mean(cvscores):.3f} with standard deviation of {std(cvscores):.3f}')


def improved_random_forest(X_train, X_test, y_train, y_test):
    """
    This function will pick the best hyperparameters for the random forest classifier and fit
    the training data based on this new model. It will then perform similar evaluation as before.

    :param X_train: The training features
    :param X_test: The testing features
    :param y_train: The training labels
    :param y_test: The testing labels
    :return: void
    """
    # Hyperparameter grid
    param_grid = {
        'n_estimators': np.linspace(10, 200).astype(int),
        'max_depth': [None] + list(np.linspace(3, 20).astype(int)),
        'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
        'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False]
    }

    estimator = RandomForestClassifier(random_state=RANDOM_SEED)

    rs = RandomizedSearchCV(estimator, param_grid, n_jobs=-1,
                            scoring='roc_auc', cv=3,
                            n_iter=10, verbose=1, random_state=RANDOM_SEED)

    rs.fit(X_train, y_train)

    print(rs.best_params_)

    better_forest = rs.best_estimator_

    pred_probs = better_forest.predict_proba(X_test)[:, 1]
    preds = better_forest.predict(X_test)

    print("Classification Report and Metrics for Random Forest Classifier")
    print(classification_report(y_test, preds))

    plot_roc(y_test, pred_probs, "Improved Random Forest")
    plot_matrix(y_test, preds, "Improved Random Forest")

    pickle.dump(better_forest, open("rf.sav", 'wb'))


def bagging_experiment(X, y):
    """
    This function performs a bagging ensemble model binary classification

    :param X: This is the full set of predictors
    :param y: This is the full set of outcome variables
    :return: void
    """
    model = BaggingClassifier()
    cv = KFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f'Bagging Ensemble Accuracy: {mean(scores):.3f} with standard deviation of {std(scores):.3f}')
