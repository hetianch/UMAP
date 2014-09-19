import numpy as np
# import data and labels using python array handling package numpy
data = np.loadtxt("/Users/071cht/Desktop/programming_language_tutorial/Python/scikit/Ye_thesis/data_scikit/data.txt",delimiter= ',') 
labels = np.loadtxt("/Users/071cht/Desktop/programming_language_tutorial/Python/scikit/Ye_thesis/data_scikit/labels.txt")
intLabels=labels.astype(int)
# scikit has a imputer class which provide basic strategies for imputing missing values, using mean/median/or the most frequent values of the row or column
# in which the missing values are located. 

#the following code retures an np.array, data_nonmissing. It is a non-missing value version of data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data)
data_nonmissing=imp.transform(data)

#rename data and intLables to chase_X and chase_y for model fitting convenience
X=data_nonmissing
y=intLabels

#set training and testing? data for 7-fold cross validation
seperateIdx= len(X)*60/100
X_train = X[0:seperateIdx]
y_train = y[0:seperateIdx] 
X_test  = X[seperateIdx:]
y_test  = y[seperateIdx:]

print X_train.shape, X_test.shape


#train a knn classifier
from sklearn import neighbors
from sklearn import cross_validation

n_neighbors = 15
h = .02  # step size in the mesh

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    knn_classifier=clf.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    
    
#ACC
ACC=knn_classifier.score(X_test,y_test)
print 'knn_ACC:' + ' '+ str(ACC) 

#RMSE
from sklearn.metrics import mean_squared_error
y_pred_prob=knn_classifier.predict_proba(X_test)
RMSE = mean_squared_error(y_test, y_pred_prob[:,1])
print 'knn_RMSE:' + ' '+ str(RMSE) 

##Prc,Rec,F
from sklearn.metrics import precision_recall_fscore_support
P_R_F=precision_recall_fscore_support(y_test, y_pred, average='weighted') # 'weighted' take into acount for label imbalance
print 'knn_Precision:'+ ' '+ str(P_R_F[0])
print 'knn_Recall and sensitivity of the model:'+ ' '+ str(P_R_F[1])
print 'knn_F:'+ ' '+ str(P_R_F[2])
print 'number of 1 in y_test' + ' ' +str(P_R_F[3]) # (TP+FN)

## FPR
from sklearn.metrics import accuracy_score
TP=P_R_F[1] * P_R_F[3]
TN = accuracy_score(y_test, y_pred, normalize=False) - TP
#FN = P_R_F[3]-TP
FP= TP/P_R_F[0] - TP
FPR = FP/ (FP+TN)
sepecificity = 1-FPR
print 'knn_FPR:'+ ' '+ str(FPR)
print 'specificity of the model:' + ' ' + str(sepecificity)

#ROC
from sklearn.metrics import roc_curve,auc
import pylab as pl
y_pred_prob = knn_classifier.predict_proba(X_test) 
# return an n (number of samples) by 2 (number of classes) array with probability of the sample for 
# that class. The columns correspond to the classes in sorted order (0,1...)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

## Plot ROC curve
pl.figure(0)
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0.5, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic for svm')
pl.legend(loc="lower right")

pl.show()




