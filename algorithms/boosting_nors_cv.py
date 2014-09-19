import pylab as pl
import numpy as np 
from scipy import interp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


# import data and labels using python array handling package np
data = np.loadtxt("/Users/071cht/Desktop/programming_language_tutorial/Python/scikit/Ye_thesis/data_scikit/from_Ye/testdata2_rescale.txt",delimiter= ',') 
labels = np.loadtxt("/Users/071cht/Desktop/programming_language_tutorial/Python/scikit/Ye_thesis/data_scikit/from_Ye/testlabel2.txt")

# check dataset
#print data,labels
#print type(data), type(labels)
#print data.shape,labels.shape,data.dtype,labels.dtype
"""somehow labels are loaded as float so we need to change it back to int in order to implement classifier"""
intLabels=labels.astype(int)
#print intLabels.dtype
#assert not np.any(np.isnan(data) | np.isinf(data))
#print np.isinf(data),np.isnan(data)
#print 1 in np.isinf(data) # if returns true, means there're inf in data
#print 1 in np.isnan(data) # if returns true, means there're nan in data
"""We must pay special attention to missing values, because scikit cann't use dataset containing missing values
Solutions:
1.discard entire rows and columns containing missing values
2.imputation"""

# scikit has a imputer class which provide basic strategies for imputing missing values, using mean/median/or the most frequent values of the row or column
# in which the missing values are located. 

#the following code retures an np.array, data_nonmissing. It is a non-missing value version of data

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data)
data_nonmissing=imp.transform(data)

#rename data and intLables for model fitting convenience
X_data=data_nonmissing
Y=intLabels

pca = PCA(n_components=50)
reduction=pca.fit(X_data)
X = reduction.transform(X_data)



"""explained_variance=list()
components=np.linspace(10,500,50)
for i in components:
	pca = PCA(n_components=i.astype(int))
	reduction=pca.fit(X_data)
	X = reduction.transform(X_data)
	explained_variance_elm=pca.explained_variance_[-1]
	explained_variance.append(explained_variance_elm)



#plot the PCA spectrum
pl.figure(0)
pl.plot(components,explained_variance, linewidth=2)
pl.xlim([0.0, 500])
pl.ylim([0.0, 1])
pl.xlabel('n_components')
pl.ylabel('explained_variance increment by adding components')"""


# setting cross validation and learing algorithms
cvfold=2
kfold = cross_validation.KFold(len(X), n_folds=cvfold)

#set up params
boost_params = {'n_estimators': 100, 'max_depth': 1,'learning_rate': 1.0, 'random_state': 0}
logit_params={'C':1e5}
svm_linear_params={'kernel': 'linear','probability':1}
svm_gaussian_params={'kernel': 'rbf', 'probability': 1}
knn_params={'n_neighbors':15, 'weights': 'distance'}
params=dict([('boost', boost_params),('logit',logit_params),('svm_linear',svm_linear_params),('svm_gaussian',svm_gaussian_params),('knn',knn_params)])

#set up classifiers
boost= GradientBoostingClassifier(**params['boost'])
logit= linear_model.LogisticRegression(**params['logit'])
svm_linear=SVC(**params['svm_linear'])
svm_gaussian=SVC(**params['svm_gaussian'])
knn= neighbors.KNeighborsClassifier(**params['knn'])

algorithms=dict([('boosting',boost),('logistic regression',logit),('support vector machine with linear kernel',svm_linear),('support vector machine with gaussian kernel',svm_gaussian),('k-nearest neighbors',knn)])
algorithms_name=['boosting','logistic regression','support vector machine with linear kernel','support vector machine with gaussian kernel','k-nearest neighbors']

for name in algorithms_name:

	ACC=list()
	RMSE=list()
	Precision=list()
	Recall=list()
	F=list()
	Num_1_inytest=list()
	FPR=list()
	Specificity=list()
	ROC_AUC=list()
	mean_tpr=0.0
	mean_fpr=np.linspace(0,1,100)
	all_tpr=[]
	cm=np.array([[0,0],[0,0]])
	algorighm=algorithms[name]
	
	for i,(train, test) in  enumerate (kfold):
		curr_classifier=algorighm.fit(X[train],Y[train])
		#print 'curr_classifier' ,curr_classifier
		y_pred = curr_classifier.predict(X[test])
		y_pred_prob=curr_classifier.predict_proba(X[test])
		# return an n (number of samples) by 2 (number of classes) array with probability of the sample for 
	    # that class. The columns correspond to the classes in sorted order (0,1...)

		#ACC
		ACC_elm=curr_classifier.score(X[test], Y[test])
		ACC.append(ACC_elm)
      
	
		#RMSE
		RMSE_elm = mean_squared_error(Y[test], y_pred_prob[:,1])
		RMSE.append(RMSE_elm)
		
		##Prc,Rec,F
		P_R_F=precision_recall_fscore_support(Y[test], y_pred, average='weighted') # 'weighted' take into acount for label imbalance
		Precision_elm=P_R_F[0]
		Recall_elm=P_R_F[1]
		F_elm=P_R_F[2]
		Num_1_inytest_elm=P_R_F[3]
		Precision.append(Precision_elm)
		Recall.append(Recall_elm)
		F.append(F_elm)
		Num_1_inytest.append(Num_1_inytest_elm)

		## FPR
		TP=P_R_F[1] * P_R_F[3]
		TN = accuracy_score(Y[test], y_pred, normalize=False) - TP
		#FN = P_R_F[3]-TP
		FP= TP/P_R_F[0] - TP
		FPR_elm = FP/ (FP+TN)
		Specificity_elm = 1-FPR_elm
		FPR.append(FPR_elm)
		Specificity.append(Specificity_elm)

		#compute for roc
		fpr, tpr, thresholds = roc_curve(Y[test], y_pred_prob[:, 1])
		mean_tpr += interp(mean_fpr, fpr, tpr)
		mean_tpr[0] = 0.0
		roc_auc = auc(fpr, tpr)
		ROC_AUC_elm = roc_auc
		ROC_AUC.append(ROC_AUC_elm)
		pl.figure(1)
		pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.4f)' % (i, roc_auc))	

		#compute for confusion matrix
		labels=['not chase','chase']
		cm_elm=confusion_matrix(Y[test],y_pred)
		cm =cm+cm_elm
	#plot ROC mean curve
	mean_tpr /= cvfold
	mean_tpr[-1]=1.0
	mean_auc = auc(mean_fpr,mean_tpr)
	pl.plot(mean_fpr,mean_tpr, 'k--', label='Mean ROC (area = % 0.4f)' % mean_auc,lw=2)   
	
	#plot ROC luck curve
	pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
	
	#basic setting of plot
	pl.xlim([-0.05, 1.05])
	pl.ylim([-0.05, 1.05])
	pl.xlabel('False Positive Rate')
	pl.ylabel('True Positive Rate')
	pl.title('Receiver operating characteristic for %s' % name)
	pl.legend(loc="lower right")

	#plot confusion matrix
	#from sklearn.cross_validation import train_test_split     We don't use this function because we have subseted train and test by random seed
	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	fig = pl.figure(2)
	ax=fig.add_subplot(111)
	cax = ax.matshow(cm)
	fig.colorbar(cax)
	##Add annotations in the 2 by 2 table 
	width=len(cm)
	height=len(cm[0])
	for x in xrange(width):
		for y in xrange(height):
			ax.annotate(str(cm[x][y]),xy=(y,x), horizontalalignment = 'center', verticalalignment = 'center', weight= 'extra bold',size = 'large')

	ax.set_xticklabels(['']+labels)
	ax.set_yticklabels([''] + labels)
	pl.xlabel('predicted')
	pl.ylabel('True')
	pl.title('confusion matrix for %s' % name)


	#output model evaluation measures 
	print name+' '+'ACC='+str(sum(ACC)/cvfold)   ##return : 0.55
	print name+' '+'RMSE='+str(sum(RMSE)/cvfold)  
	print name+' '+'Precision='+str(sum(Precision)/cvfold)  
	print name+' '+'Recall='+str(sum(Recall)/cvfold)  
	print name+' '+'F='+str(sum(F)/cvfold)  
	print name+' '+'Num_1_inytest='+str(sum(Num_1_inytest)/cvfold)  
	print name+' '+'FPR='+str(sum(FPR)/cvfold)
	print name+' '+'Specificity='+str(sum(Specificity)/cvfold)  
	print name+' '+'AUC='+str(sum(ROC_AUC)/cvfold)   
	pl.show()
	



