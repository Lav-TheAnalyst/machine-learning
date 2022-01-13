from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import graphviz
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
from sklearn.model_selection import cross_val_score
from sklearn import tree
import matplotlib.pyplot as plt
df=pd.read_csv(r'C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/Concrete Strength/Concrete_Data.csv')
dum_df=pd.get_dummies(df,drop_first=True)

X = dum_df.drop('Strength',axis=1)
y = dum_df['Strength']
#----------------------------------------------------------------------
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=2020,)
dtr=DecisionTreeRegressor(random_state=2021,max_depth=2)
dtr.fit(X_train,y_train)


dot_data=tree.export_graphviz(dtr,out_file=None,
                             feature_names=X_train.columns,
                             filled=True,
                             rounded=True,
                             special_characters=True)


graph=graphviz.Source(dot_data)
graph

#------------------------------------------------
#best parameters
y_pred=dtr.predict(X_test)

#GridSearchCV

paramters={
'max_depth':[5,6,7,8,9,10] ,
'min_samples_split':[3,4,5,6,7],
'min_samples_leaf':[3,4,5,6,7]
}

kfold=KFold(n_splits=5,random_state=2021,shuffle=True)
dtr=DecisionTreeRegressor(random_state=2021)
cv=GridSearchCV(estimator=dtr,param_grid=paramters,cv=kfold,scoring='r2')

cv.fit(X,y)

print(cv.best_params_)
print(cv.best_score_)

#-------------------------------
#best model tree

best_model=cv.best_estimator_

dot_data=tree.export_graphviz(best_model,out_file=None,
                             feature_names=X_train.columns,
                             filled=True,
                             rounded=True,
                             special_characters=True)

graph=graphviz.Source(dot_data)
graph


#--------------------------------
#best_model.feature_importances_
best_model.feature_importances_

ind = np.arange(X.shape[1])
plt.bar(ind,best_model.feature_importances_)
plt.xticks(ind,(X.columns),rotation=45)
plt.title('Feature Importance')
plt.xlabel("Variables")
plt.show()