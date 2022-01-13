import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV 
from sklearn.model_selection import StratifiedKFold ,cross_val_score
from sklearn.svm import SVC 
from sklearn.metrics import roc_auc_score,accuracy_score,log_loss 
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df=pd.read_csv(r'C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/Vehicle Silhouettes/Vehicle.csv')

lbl_encode=LabelEncoder()
lbl_y=lbl_encode.fit_transform(df.iloc[:,-1])

X=df.iloc[:,:-1]
y=lbl_y


x_train, x_test,y_train, y_test = train_test_split(X,y, 
                                                   random_state=2021, 
                                                   test_size=0.3,stratify=y) 

lda=LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
y_pred=lda.predict(x_test)
y_pred_prob=lda.predict_proba(x_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test,y_pred))
print(log_loss(y_test,y_pred_prob))


kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2021) 
lda = LinearDiscriminantAnalysis() 
results = cross_val_score(estimator=lda,X=x,y=y, 
                          scoring='neg_log_loss',cv=kfold) 
print(results.mean())