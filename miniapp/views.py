from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_blobs


def title(request):
	return render(request, 'templates/home.html') 


def nb(request):
	dataset = pd.read_csv('static/excel/fifadataset.csv')
	dataset.fillna(0)
	X = dataset.iloc[:, 1:31].values
	y = dataset.iloc[:, -1].values
	X = X.astype(np.int64)
	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
	# Feature Scaling
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	# Fitting Naive Bayes to the Training set
	classifier = GaussianNB()
	classifier.fit(X_train, y_train)
	# Predicting the Test set results
	y_pred = classifier.predict(X_test)
	# Making the Confusion Matrix
	cm = confusion_matrix(y_test, y_pred)
	accuracy = accuracy_score(y_pred,y_test)
	accuracy = 100*accuracy
	X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu');
	return render(request, 'nb.html', {'accuracy': round(accuracy, 2)})

def rf(request):
	dataset = pd.read_csv('static/excel/fdata.csv')
	dataset.fillna(0)
	X = dataset.iloc[:, 24:50].values
	y = dataset.iloc[:, 7].values
	X = X.astype(np.int64)
	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	# Fitting Random Forest Regression to the dataset
	regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
	regressor.fit(X_train, y_train)
	# Predicting a new result
	y_pred = regressor.predict(X_test)
	rmse=0
	for i in range(len(y_test)):
	    rmse+=(y_test[i]-y_pred[i])**2
	rmse=rmse/len(y_test)
	rmse=(rmse)**(0.5)

	return render(request, 'rf.html' , {'rmse': round(rmse, 2)})

def lr(request):
	dataset = pd.read_csv('static/excel/fifadataset.csv')
	dataset.fillna(0)
	X = dataset.iloc[:, 1:31].values
	y = dataset.iloc[:, -1].values
	X = X.astype(np.int64)
	# Splitting the dataset into the Training set and Test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	# Fitting Simple Linear Regression to the Training set
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)
	# Predicting the Test set results
	y_pred = regressor.predict(X_test)
	y_pred = y_pred.astype(np.float64)
	rmse=0
	for i in range(len(y_test)):
	    rmse+=(y_test[i]-y_pred[i])**2
	rmse=rmse/len(y_test)
	rmse=(rmse)**(0.5)
	return render(request, 'lr.html', {'rmse': round(rmse, 2)})

