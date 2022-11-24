# PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/junhea/OSS-project-2/blob/main/template.py

import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_dataset(dataset_path):
    # To-Do: Implement this function
    return pd.read_csv(dataset_path)


def dataset_stat(dataset_df):
    # To-Do: Implement this function
    cnt = dataset_df.groupby('target').size()
    return ((dataset_df.columns != 'target').sum(), cnt[0], cnt[1])


def split_dataset(dataset_df, testset_size):
    # To-Do: Implement this function
    X = dataset_df.drop(columns='target', axis=1)
    y = dataset_df['target']
    return train_test_split(X, y, test_size=testset_size)


def decision_tree_train_test(x_train, x_test, y_train, y_test):
    # To-Do: Implement this function
    dt_cls = DecisionTreeClassifier()
    dt_cls.fit(x_train, y_train)
    y_pred = dt_cls.predict(x_test)
    return (accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred))


def random_forest_train_test(x_train, x_test, y_train, y_test):
    # To-Do: Implement this function
    rf_cls = RandomForestClassifier()
    rf_cls.fit(x_train, y_train)
    y_pred = rf_cls.predict(x_test)
    return (accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred))


def svm_train_test(x_train, x_test, y_train, y_test):
    # To-Do: Implement this function
    svm_pipe = make_pipeline(
        StandardScaler(),
        SVC()
    )
    svm_pipe.fit(x_train, y_train)
    y_pred = svm_pipe.predict(x_test)
    return (accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred))


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
