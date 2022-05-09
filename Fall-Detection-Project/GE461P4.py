from lib2to3.pytree import convert
from kiwisolver import Solver
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import metrics
#loading the data
data = pd.read_csv('falldetection_dataset.csv', header=None, sep=',')
data.head()

#dividing the data to features and labels as x and y respectively 
datax = data.dropna()
datax = data.drop(labels = [0,1], axis = 1)
data.head()
datay= data[1]
datay.head()

#splitting the data into training and testing sets
from sklearn.model_selection import train_test_split
#At firs I tried with train_test_split but code would not work and logic was problematic so I switched manual
X_train=datax.values
y_train=datay.values

#Apply PCA and find variances
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train_transform = pca.fit_transform(X_train)
variances = pca.explained_variance_ratio_

#calculating the variances for each principal component
for i in range(len(variances)):
    print("Variance of PC", i+1, "is", variances[i])
print("Variance captured by top 2 PCs:", sum(variances))

#plotting the data as x being features and y being labels of Fall and Non-Fall

plt.scatter(X_train_transform[y_train == 'F', 0], X_train_transform[y_train == 'F', 1], label="Fall", c='red')
plt.scatter(X_train_transform[y_train == 'NF', 0], X_train_transform[y_train == 'NF', 1], label="Non-Fall" ,c='blue')
plt.title('Graph of Training Data for PCA with 2 components')
plt.legend(loc="upper right", shadow=True, fontsize='x-large', ncol=2, fancybox=True)
plt.savefig('PCA_2_components_training.png')
plt.show()
#clustering the data

#Clustering the data with K-Mean Clustering
from sklearn.cluster import KMeans
N = [2,3,4,5,6,7,8]
for i in N:
    k = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    k.fit(X_train_transform)
    pred = k.predict(X_train_transform)

    #plotting the data for each cluster number separetly

    for j in range(i):
        plt.scatter(X_train_transform[pred == j, 0], X_train_transform[pred == j, 1], label="Cluster" + str(j+1))
    plt.legend(loc="lower right", shadow=True, fontsize='x-small', ncol=2, fancybox=True, bbox_to_anchor=(1.3, 1))
    plt.title(str(i) + " Clustering Prediction Projections")
    plt.savefig('K-Means_' + str(i) + '_clusters.png')
    plt.show()


#finding the optimal number of clusters
from sklearn.metrics import silhouette_score
for i in N:
    k = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    k.fit(X_train_transform)
    pred = k.predict(X_train_transform)
    score = silhouette_score(X_train_transform, pred)
    print("For number of clusters =", i, "the silhouette score is", score)
print("The optimal number of clusters is", N[np.argmax(silhouette_score(X_train_transform, pred))])

#when N=2, check the degree of percentage overlap/consistency between the cluster mem- berships and the action labels originally provided.
k = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
k.fit(X_train_transform)
pred = k.predict(X_train_transform)
val = [1 if i == "F" else 0 for i in y_train]
acc1 = metrics.accuracy_score(val, pred)
acc2= metrics.accuracy_score(val, 1-pred)
print("Consistency of Train Data:", max(acc1, acc2))

#since the graph is not very clear because of 2 outlier points, we will remove them and try again for better visualization
#we have 2 outlier points, if we calculate from Y axis, we can see that the outlier points are at the top and bottom of the graph
#so we will use min and max of Y axis to remove the outlier points
X_train_transform1 = X_train_transform[X_train_transform[:,1] < np.max(X_train_transform[:,1])]
X_train_transform2= X_train_transform1[X_train_transform1[:,1] > np.min(X_train_transform1[:,1])]
y_train1=y_train[X_train_transform[:,1] < np.max(X_train_transform[:,1])]
y_train2=y_train1[X_train_transform1[:,1] > np.min(X_train_transform1[:,1])]
for i in N:
    k = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    k.fit(X_train_transform2)
    pred = k.predict(X_train_transform2)
    score = silhouette_score(X_train_transform2, pred)
    print("For number of clusters =", i, "the silhouette score is", score)
print("The optimal number of clusters is", N[np.argmax(silhouette_score(X_train_transform2, pred))])
k = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
k.fit(X_train_transform2)
pred = k.predict(X_train_transform2)
val = [1 if i == "F" else 0 for i in y_train2]
acc1 = metrics.accuracy_score(val, pred)
acc2= metrics.accuracy_score(val, 1-pred)
print("Consistency of Train Data:", max(acc1, acc2))


#After removing the outlier points, we can see that the graph is much more clear now
plt.scatter(X_train_transform2[y_train2 == 'F', 0], X_train_transform2[y_train2 == 'F', 1], label="Fall", c='red')
plt.scatter(X_train_transform2[y_train2 == 'NF', 0], X_train_transform2[y_train2 == 'NF', 1], label="Non-Fall" ,c='blue')
plt.title('Graph of Training Data for PCA with 2 components-outlier points removed')
plt.legend(loc="upper right", shadow=True, fontsize='x-large', ncol=2, fancybox=True)
plt.savefig('PCA_2_components_training_outlier_removed.png')
plt.show()


#Again clustering the data and plotting
for i in N:
    k = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    k.fit(X_train_transform2)
    pred = k.predict(X_train_transform2)
    
    for j in range(i):
        plt.scatter(X_train_transform2[pred == j, 0], X_train_transform2[pred == j, 1], label="Cluster" + str(j+1))
    plt.legend(loc="lower right", shadow=True, fontsize='x-small', ncol=2, fancybox=True,bbox_to_anchor=(1, 1))
    plt.title(str(i) + " Clustering Prediction Projections without outliers")
    plt.savefig('K-Means_' + str(i) + '_clusters_without_outliers.png')
    plt.show()






#PART B OF THE PROJECT STARTS FROM HERE
#PART B OF THE PROJECT STARTS FROM HERE
#PART B OF THE PROJECT STARTS FROM HERE

#First we start with splitting the data to train , test and validation sets
from sklearn.model_selection import train_test_split
train1,test1,train2,test2=train_test_split(X_train,y_train,test_size=0.3,random_state=0)
test1,valid1,test2,valid2=train_test_split(test1,test2,test_size=0.5,random_state=0)

#then we prepare the necessary parameters for SVM
c_val =[0.0001, 0.001, 0.01, 0.1,1 ,10,100]
kernel = ['linear', 'sigmoid', 'poly', 'rbf']
gval =['scale', 'auto']
degree = [1,2,3,4,5,6,7]

#then we run the grid search to find the best parameters
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
results1 =[]
for i in c_val:
    for j in kernel:
        for k in gval:
            for l in degree:
                svm = SVC(C=i, kernel=j, gamma=k, degree=l,max_iter=10000)
                svm.fit(train1,train2)
                pred = svm.predict(valid1)
                acc = metrics.accuracy_score(valid2, pred)
                print("For C =", i, "kernel =", j, "gamma =", k, "degree =", l, "the accuracy is", acc)
                results1.append([i,j,k,l,acc])
                
find_bests=pd.DataFrame(results1,columns=['C','kernel','gamma','degree','accuracy'])
find_bests.sort_values(by=['accuracy'],ascending=False,inplace=True)
use_best=find_bests.iloc[0]
#then we run the model with the best parameters

svm = SVC(C=use_best[0], kernel=use_best[1], gamma=use_best[2], degree=use_best[3])
svm.fit(train1,train2)
pred = svm.predict(test1)
acc = metrics.accuracy_score(test2, pred)
print("The accuracy of the model is", acc)

#then we plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test2, pred)
print(cm)

#Saving the results and best parameters in a csv file
find_bests.to_csv('SVM_results.csv')


#MLP PART START FROM HERE
#Arranging the Parameters for MLP
from sklearn.neural_network import MLPClassifier
layer =[(8,8),(16,16),(32,32),(64,64),(128,128)]
act_func = ['relu', 'tanh', 'logistic']
solver_parameter = [ 'sgd', 'adam']
l_rate = [0.0001,0.001, 0.01, 0.1]
alpha=[0.0001,0.001, 0.01, 0.1, 1]
results2=[]

#Implementing the MLP based on Parameters
for i in layer:
    for j in act_func:
        for k in solver_parameter:
            for l in l_rate:
                for m in alpha:
                    mlp = MLPClassifier(hidden_layer_sizes=i, activation=j, solver=k, learning_rate_init=l, alpha=m,max_iter=10000)
                    mlp.fit(train1,train2)
                    pred = mlp.predict(valid1)
                    acc = metrics.accuracy_score(valid2, pred)
                    print("For layer=", i, "activation=", j, "solver=", k, "learning_rate=", l, "alpha=", m, "the accuracy is", acc)
                    results2.append([i,j,k,l,m,acc])

#Finding best parameters
find_bests2=pd.DataFrame(results2,columns=['layer','activation','solver','learning_rate','alpha','accuracy'])
find_bests2.sort_values(by=['accuracy'],ascending=False,inplace=True)
use_best2=find_bests2.iloc[0]
print("The best parameters are according to the validation data", use_best2)

#then we run the model with the best parameters
mlp = MLPClassifier(hidden_layer_sizes=use_best2[0], activation=use_best2[1], solver=use_best2[2], learning_rate_init=use_best2[3], alpha=use_best2[4],max_iter=10000)
mlp.fit(train1,train2)
pred = mlp.predict(test1)
acc = metrics.accuracy_score(test2, pred)
print("The accuracy of the model is", acc)

#then we plot the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test2, pred)
print(cm)

#Saving the results and best parameters in a csv file
find_bests2.to_csv('MLP_results.csv')

         
