import pandas as pd  
from sklearn.ensemble import StackingClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,roc_auc_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
 
df=pd.read_csv(r'C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/bank/bank.csv',sep=';')
dum_df = pd.get_dummies(df,drop_first=True) 
 
X= dum_df.iloc[:,:-1] 
y = dum_df.iloc[:,-1] 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,  
                                                    random_state=2018) 
 
 
logreg = LogisticRegression() 
svc =SVC() 
gauss = GaussianNB() 
dt=DecisionTreeClassifier(random_state=2021)
lda = XGBClassifier(random_state=2021) 
est = [("Logistic",logreg),("SVC",svc),("Gaussian",gauss),("DecisionTree",dt),("LinearDiscriminantAnalysis",lda)] 
clf = StackingClassifier(estimators=est, 
                         final_estimator=lda, 
                         passthrough=True) 
clf.fit(X_train,y_train) 
y_pred = clf.predict(X_test) 
print(accuracy_score(y_test, y_pred)) 
y_pred_prob = clf.predict_proba(X_test)[:,1] 
print(roc_auc_score(y_test, y_pred_prob))