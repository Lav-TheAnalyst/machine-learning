import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv(r"C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/Kyphosis/Kyphosis.csv") 
dum_df = pd.get_dummies(df,drop_first=True) 
 
X = dum_df.iloc[:,1:-1] 
y = dum_df.iloc[:,-1] 
 
X_train,X_test,y_train,y_test = train_test_split(X,y, 
                                                 test_size=0.3, 
                                                 random_state=2021, 
                                                 stratify=y) 


#==============================================

lr=LogisticRegression()
model_bg=BaggingClassifier(random_state=2021,base_estimator=lr,
                           n_estimators=15,
                           max_features=X_train.shape[1],
                           max_samples=X_train.shape[0],oob_score=True)

model_bg.fit(X_train,y_train)
print(model_bg.oob_score_)
y_pred=model_bg.predict(X_test)
print(accuracy_score(y_test,y_pred))
y_pred_prob=model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))



#===================================
model= GaussianNB()
model1=BaggingClassifier(random_state=2021,base_estimator=model,
                         n_estimators=15,
                           max_features=X_train.shape[1],
                           max_samples=X_train.shape[0],oob_score=True)
model1.fit(X_train,y_train)
print(model1.oob_score_)
y_pred=model1.predict(X_test)
print(accuracy_score(y_test,y_pred))
y_pred_prob=model1.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

#========================================
model_bg=BaggingClassifier(random_state=2021,
                           max_features=X_train.shape[1],
                           n_estimators=15,
                           max_samples=X_train.shape[0],oob_score=True)
model_bg.fit(X_train,y_train)
print(model1.oob_score_)
y_pred=model_bg.predict(X_test)
print(accuracy_score(y_test,y_pred))
y_pred_prob=model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))


#========================================
ld=LinearDiscriminantAnalysis()
model_bg=BaggingClassifier(random_state=2021,base_estimator=ld,
                           n_estimators=15,
                           max_features=X_train.shape[1],
                           max_samples=X_train.shape[0],oob_score=True)
model_bg.fit(X_train,y_train)
print(model1.oob_score_)
y_pred=model_bg.predict(X_test)
print(accuracy_score(y_test,y_pred))
y_pred_prob=model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

