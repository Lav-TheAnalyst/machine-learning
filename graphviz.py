import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score
import graphviz
from sklearn import tree
df=pd.read_csv(r'C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/Sonar/Sonar.csv')

dum_df=pd.get_dummies(df,drop_first=True)

x=dum_df.iloc[:,1:-1]
y=dum_df.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2021,stratify=y)


dtc=DecisionTreeClassifier(random_state=2021)
# =============================================================================
# dtc.fit(X_train,y_train)
# 
# dot_data=tree.export_graphviz(dtc,out_file=None,
#                               feature_names=X_train.columns,
#                               class_names=['Benign','Maliganat'],
#                               filled=True,rounded=True,
#                               special_characters=True)
# graph=graphviz.Source(dot_data)
# graph
# 
# 
# =============================================================================

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2021)
parametrs={'max_depth':[3,4,5,6],
           'min_samples_split':[2,3,4,5],
           'min_samples_leaf':[1,2,3,4,5]}

# =============================================================================
# cv=GridSearchCV(estimator=dtc,param_grid=parametrs,
#                 scoring='roc_auc',cv=kfold,verbose=3)
# =============================================================================



cv = GridSearchCV(estimator=dtc, param_grid=parametrs,  
                          scoring='roc_auc', cv=kfold)  
 
cv.fit(x,y) 
 
pd_cv = pd.DataFrame(cv.cv_results_) 
 
print(cv.best_params_) 
print(cv.best_score_)