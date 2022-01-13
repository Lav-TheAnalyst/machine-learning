import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score
import numpy as np


df=pd.read_csv(r'C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/human-resources-analytics/HR_comma_sep.csv')
dum_df=pd.get_dummies(df,drop_first=True)
X=dum_df.drop('left',axis=1)
y=dum_df['left']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2021,stratify=y)


clf = RandomForestClassifier(random_state=2021)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print(accuracy_score(y_test,y_pred))


#====================================Grid Search CV

params={'max_features':[2,3,4,5,6]}
kfold= StratifiedKFold(n_splits=5,random_state=2021,shuffle=True)
clf = RandomForestClassifier(random_state=2021)
CV = GridSearchCV(estimator=clf,param_grid=params,cv=kfold,scoring='roc_auc',verbose=2)

CV.fit(X,y)
print(CV.best_params_)
print(CV.best_score_)

#=============================================================

best_model=CV.best_estimator_

import matplotlib.pyplot as plt

print(best_model.feature_importances_)

ind=np.arange(X.shape[1])
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=90)
plt.title('Feature Importance')
plt.xlabel('Variables')
plt.show()
