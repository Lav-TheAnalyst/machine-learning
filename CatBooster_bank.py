
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np 
 
df=pd.read_csv(r'C:/Users/RAJPUT/Desktop/CDAC/Advance Analytics/Cases/bank/bank.csv',sep=';')
dum_df=pd.get_dummies(df,drop_first=True)

X = dum_df.iloc[:,:-1] 
y = dum_df.iloc[:,-1]

clf = CatBoostClassifier(learning_rate=0.03,random_seed= 2021)
clf.fit(X_train,y_train,verbose=False,plot=True)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test,y_pred))


clf_grid = CatBoostClassifier(random_state=2021,iterations=250)


grid = {
    'learning_rate': [0.03, 0.1],
    'depth':[4, 6, 10],
    'l2_leaf_reg': [1, 3, 5]
}

grid_search_results = clf_grid.grid_search(grid, X,y, verbose=3, plot=True,stratified=True)


grid_search_results['params']

grid_search_results['cv_results'].keys()

pd.DataFrame(grid_search_results['cv_results'])

