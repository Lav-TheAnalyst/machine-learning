import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score

df= pd.read_csv(r"C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/Kyphosis/Kyphosis.csv")
dum_df = pd.get_dummies(df,drop_first=True)

x=dum_df.drop('Kyphosis_present',axis=1)
y=dum_df['Kyphosis_present']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,stratify=y,
                                                    random_state=2021)


#--------------------------------------------------
# LogisticRegression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
model_bg = BaggingClassifier(random_state=2021,
                             base_estimator=lr,
                             n_estimators=15,
                             max_features=X_train.shape[1],
                             max_samples=X_train.shape[0],
                             oob_score=True)
                             
model_bg.fit(X_train,y_train)
print(model_bg.oob_score_)
y_pred = model_bg.predict(X_test)
print(accuracy_score(y_test,y_pred))
y_pred_prob = model_bg.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,y_pred_prob))

#---------------------------------------------------
# GaussianNB
from sklearn.naive_bayes import GaussianNB

print("Gaussian")
model = GaussianNB()
model.fit(X_train,y_train)


y_pred_prob = model.predict_proba(X_test)[:,1]
print("roc_auc_score")
print(roc_auc_score(y_test,y_pred_prob))
print()
#Gaussian roc_auc_score : 0.81
#---------------------------------------------------
# LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda=LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)

y_pred_prob = lda.predict_proba(X_test)[:,1]
print("roc_auc_score")
print(roc_auc_score(y_test,y_pred_prob))
print()

#LinearDiscriminantAnalysis roc_auc_score: 0.84
#---------------------------------------------------
# LogisticRegression
from sklearn.linear_model import LogisticRegression
print("Logistic")

log=LogisticRegression()
log.fit(X_train,y_train)

y_pred_prob = log.predict_proba(X_test)[:,1]
print("roc_auc_score")
print(roc_auc_score(y_test,y_pred_prob))