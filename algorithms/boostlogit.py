import numpy 
# import data and labels using python array handling package numpy
data = numpy.loadtxt("/Users/071cht/Desktop/programming_language_tutorial/Python/scikit/Ye_thesis/data.txt",delimiter=',') 
labels = numpy.loadtxt("/Users/071cht/Desktop/programming_language_tutorial/Python/scikit/Ye_thesis/labels.txt")

# check dataset
#print data,labels
#print type(data), type(labels)
#print data.shape,labels.shape,data.dtype,labels.dtype
"""somehow labels are loaded as float so we need to change it back to int in order to implement classifier"""
intLabels=labels.astype(int)
#print intLabels.dtype
#assert not numpy.any(numpy.isnan(data) | numpy.isinf(data))
#print numpy.isinf(data),numpy.isnan(data)
#print 1 in numpy.isinf(data) # if returns true, means there're inf in data
#print 1 in numpy.isnan(data) # if returns true, means there're nan in data
"""We must pay special attention to missing values, because scikit cann't use dataset containing missing values
Solutions:
1.discard entire rows and columns containing missing values
2.imputation"""

# scikit has a imputer class which provide basic strategies for imputing missing values, using mean/median/or the most frequent values of the row or column
# in which the missing values are located. 

#the following code retures an np.array, data_nonmissing. It is a non-missing value version of data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data)
data_nonmissing=imp.transform(data)

###HOWEVER:
## From my aspect, it doesn't make much sense to impute the value according to the mean of the entire colomn
## So I decide to adopt the first solution in the future.

""" I need to figure out how to work with numpy array"""


#rename data and intLables to chase_X and chase_y for model fitting convenience
X=data_nonmissing
y=intLabels

"""NOTE!!! in order to make it run faster, I choose every 10th colomn of in entire dataset
X_subset=numpy.zeros((X.shape[0], 1+numpy.floor((X.shape[1]-1)/100)))
COL=X.shape[1]
i=0
for value in range(COL):
	if value %100 == 0:
		 X_subset[:,i] = X[:,value]
		 i=i+1"""
### feature reduction by PCA. We choose 500 components to put into feature selection algorithm
from sklearn.decomposition import PCA
pca = PCA(n_components=100)
reduction=pca.fit(X)
X_subset = reduction.transform(X)
print X.shape, X_subset.shape

# Plot the PCA spectrum
import pylab as pl
pl.figure(0)
#pl.axes([.2, .2, .7, .7])
pl.plot(pca.explained_variance_, linewidth=2)
pl.axis('tight')
pl.xlim([0.0, 80])
pl.ylim([0.0, 30000])
pl.xlabel('n_components')
pl.ylabel('explained_variance_')
#sebset data for training and testing by random seed and use 60% to train and 40% to test
numpy.random.seed(0) #Seed the random number generator with a fixed value, so that it return the same array each time.
indices = numpy.random.permutation(len(X_subset))
seperateIdx= len(X_subset)*60/100
X_train = X_subset[indices[0:seperateIdx]] 
y_train = y[indices[0:seperateIdx]] 
X_test  = X_subset[indices[seperateIdx:]]
y_test  = y[indices[seperateIdx:]] 


"""turn off feature selection for now
from sklearn.logistic import SVC
from sklearn.feature_selection import RFECV
params = {'kernel': 'linear', 'probability': 1}
svc = SVC(**params)
rfecv = RFECV(estimator=svc, step=1, cv=3, scoring='accuracy')
logistic_classifier=rfecv.fit(X_train, y_train)
print logistic_classifier
error = numpy.divide (rfecv.grid_scores_ , len(y_train))
print("Optimal number of features : %d" % rfecv.n_features_)
y_pred = logistic_classifier.predict(X_test)

# Plot number of features VS. cross-validation scores
import pylab as pl
pl.figure(1)
pl.xlabel("Number of features selected")
pl.ylabel("Cross validation score (error rate)")
pl.plot(range(1, len(rfecv.grid_scores_) + 1), error)
print("Optimal number of features : %d" % rfecv.n_features_)

"""

#train a logistic classifier
"""hardcode all the paremeters here"""
from sklearn import linear_model
params = {'C': 1e5}
logistic = linear_model.LogisticRegression(**params)
logistic_classifier=logistic.fit(X_train, y_train)
y_pred = logistic_classifier.predict(X_test)


""" Note:Random seed return a score around 0.9, which is much higher than KFold crossvalidation (around 0.5). I DON'T KNOW WHY!!!!
The only difference is that the sample from K-fold crossvalidation is consecutive in time and highly correlated with adjecent data points. 
But random seed shuffle the data. Will this make difference? If so, we need to shuffle the data before crossvalidate"""


# Try cross-validation(We did a 3 fold crossvalidation and arbitaryly assigning the following paremeters)
"""bug: the following code arise an warning: overflow encountered in exp. Check out the discussion about it later http://comments.gmane.org/gmane.comp.python.scikit-learn/3730"""

# 3-fold cross validation means:
# Ex: >>X = np.array([2, 3, 1, 0,12,10,22,11,22,111,23,12]),
#	  >>kfold = cross_validation.KFold(len(X), n_folds=3)  3-fold means 2/3 train, 1/3 test
#     ###ytrain [ 12  10  22  11  22 111  23  12] ytest [2 3 1 0]
	  ###ytrain [  2   3   1   0  22 111  23  12]  ytest [12 10 22 11]
	  ###ytrain [ 2  3  1  0 12 10 22 11]          ytest [ 22 111  23  12]

#from sklearn import cross_validation
#kfold = cross_validation.KFold(len(X), n_folds=3)
#gbc= GradientlogisticClassifier(n_estimators=100, max_depth=1, learning_rate=1.0,random_state=0)
#crossScores=[gbc.fit(X[train], y[train]).score(X[test], y[test]) for train, test in kfold]
#print crossScores, 'average crossvalidation score ='+str(sum(crossScores)/3)   ##return : 0.55



"""Algorithm evaluation starts here!!!!"""
"""All the following results are based on random seeds"""
#ACC
ACC=logistic_classifier.score(X_test,y_test)
print 'logistic_ACC:' + ' '+ str(ACC) 
	#Another way to calcuate Accuracy:
		# from sklearn.metrics import accuracy_score
		# ACC=accuracy_score(y_test, y_pred, normalize=True) 
		# normalize==False returns number of correctly classified samiples (TP+TN); otherwise, return the fraction


#RMSE
from sklearn.metrics import mean_squared_error
y_pred_prob=logistic_classifier.predict_proba(X_test)
RMSE = mean_squared_error(y_test, y_pred_prob[:,1])
print 'logistic_RMSE:' + ' '+ str(RMSE) 

##Prc,Rec,F
from sklearn.metrics import precision_recall_fscore_support
P_R_F=precision_recall_fscore_support(y_test, y_pred, average='weighted') # 'weighted' take into acount for label imbalance
print 'logistic_Precision:'+ ' '+ str(P_R_F[0])
print 'logistic_Recall and sensitivity of the model:'+ ' '+ str(P_R_F[1])
print 'logistic_F:'+ ' '+ str(P_R_F[2])
print 'number of 1 in y_test' + ' ' +str(P_R_F[3]) # (TP+FN)

## FPR
from sklearn.metrics import accuracy_score
TP=P_R_F[1] * P_R_F[3]
TN = accuracy_score(y_test, y_pred, normalize=False) - TP
#FN = P_R_F[3]-TP
FP= TP/P_R_F[0] - TP
FPR = FP/ (FP+TN)
sepecificity = 1-FPR
print 'logistic_FPR:'+ ' '+ str(FPR)
print 'specificity of the model:' + ' ' + str(sepecificity)

#ROC
from sklearn.metrics import roc_curve,auc
import pylab as pl
y_pred_prob = logistic_classifier.predict_proba(X_test) 
# return an n (number of samples) by 2 (number of classes) array with probability of the sample for 
# that class. The columns correspond to the classes in sorted order (0,1...)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

## Plot ROC curve

pl.figure(1)
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic for logistic regression')
pl.legend(loc="lower right")

"""
Haven't figure out how to do this
#Training and testing error by number of estimators or (number of iterations)
from sklearn.metrics import zero_one_loss

boost_test_err=numpy.zeros((params['n_estimators'],), dtype=numpy.float64)
for i, y_pred_stage in enumerate(logistic_classifier.staged_predict(X_test)):
    boost_test_err[i] = zero_one_loss(y_test, y_pred_stage, normalize=True) 
# if normalize = False it returns the number rather than the fraction of misclassification

boost_train_err = numpy.zeros((params['n_estimators'],), dtype=numpy.float64)
for i, y_pred_stage in enumerate(logistic_classifier.staged_predict(X_train)):
    boost_train_err[i] = zero_one_loss(y_train, y_pred_stage,normalize=True) 																  

fig = pl.figure()
ax = fig.add_subplot(111)
ax.plot(numpy.arange(params['n_estimators']) + 1, boost_test_err , label='Boost Test Error', color='red')
ax.plot(numpy.arange(params['n_estimators']) + 1, boost_train_err, label='Boost Train Error', color='blue')
ax.set_xlabel('number of estimators')
ax.set_ylabel('error rate')
pl.title('training and testing error by number of estimators')
leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(1.0)

pl.figure(1)
pl.subplot(1,2,1)
pl.title('training and testing error by number of estimators')
pl.plot(numpy.arange(params['n_estimators']) + 1, boost_test_err , label='Boost Test Error', color='red')
pl.plot(numpy.arange(params['n_estimators']) + 1, boost_train_err, label='Boost Train Error', color='blue')
pl.xlabel('number of estimators')
pl.ylabel('error rate')
pl.legend(loc='upper right', fancybox=True)
#leg.get_frame().set_alpha(1.0)"""


#Plot feature importance

"""the following are based on rfecv  feature selection result, let's comment out for now
feature_importance = rfecv.ranking_.max()-rfecv.ranking_+1
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = numpy.argsort(feature_importance)
sorted_idx_toplot=sorted_idx[-10:]
pos = numpy.arange(sorted_idx_toplot.shape[0]) + .5
pl.figure(3)
pl.barh(pos, feature_importance[sorted_idx_toplot], align='center')
pl.yticks(pos, sorted_idx_toplot[-10:]) # gives the feature_names in numpy.ndarray: ['f1' 'f2' 'f3' 'f4' ....]
pl.xlabel('Relative Importance')
pl.title('Variable Importance')"""



#plot confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split     We don't use this function because we have subseted train and test by random seed
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
labels= ['not chasing', 'chasing']
cm=confusion_matrix(y_test,y_pred)
#plt.clf()
fig = plt.figure(2)
ax=fig.add_subplot(111)
cax = ax.matshow(cm)
fig.colorbar(cax)

##Add annotations in the 2 by 2 table 
width=len(cm)
height=len(cm[0])
for x in xrange(width):
	for y in xrange(height):
		ax.annotate(str(cm[x][y]),xy=(y,x), horizontalalignment = 'center', verticalalignment = 'center')

ax.set_xticklabels(['']+labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('predicted')
plt.ylabel('True')
plt.title ('confusion matrix for logistic regression')

pl.show()

