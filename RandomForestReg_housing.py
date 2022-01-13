import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import numpy as np


df=pd.read_csv(r'C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/Real Estate/Housing.csv')
dum_df=pd.get_dummies(df,drop_first=True)
X=dum_df.drop('price',axis=1)
y=dum_df['price']

#=====================================================================







#=======================================================================

params={'max_features':[2,3,4,5,6]}
kfold=KFold(n_splits=5,random_state=2021,shuffle=True)
clf = RandomForestRegressor(random_state=2021)
CV = GridSearchCV(estimator=clf,param_grid=params,cv=kfold,scoring='r2')

CV.fit(X,y)
print(CV.best_params_)
print(CV.best_score_)

#========================================================================

best_model=CV.best_estimator_

import matplotlib.pyplot as plt

print(best_model.feature_importances_)

ind=np.arange(X.shape[1])
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=90)
plt.title('Feature Importance')
plt.xlabel('Variables')
plt.show()
