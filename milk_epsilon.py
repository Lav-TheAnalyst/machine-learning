import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

milk =pd.read_csv("milk.csv",index_col=0)
scaler=StandardScaler()
milkscaled=scaler.fit_transform(milk)

clust_DB=DBSCAN(eps=0.4,min_samples=2)
clust_DB.fit(milkscaled)

labels = clust_DB.labels_

clust_data=pd.DataFrame({'ClustID':labels},index=milk.index)
clust_data=pd.concat([milk,clust_data],axis='columns')

 
silhouette_score(milkscaled,clust_DB.labels_) 
 
####################################################### 
 
eps_range = [0.2,0.4,0.6,1,1.1,1.5] 
mp_range = [2,3,4,5,6,7] 
cnt =0 
a = [] 
for e in eps_range: 
    for m in mp_range: 
        clust_DB = DBSCAN(eps=e,min_samples=m) 
        clust_DB.fit(milkscaled) 
        if len(set(clust_DB.labels_))  >=2: 
            cnt = cnt + 1 
            sil_sc = silhouette_score(milkscaled,clust_DB.labels_) 
            a.append([cnt,e,m,sil_sc])
            print(e,m,sil_sc)
             
a= np.array(a) 
pa = pd.DataFrame(a,columns=['sr','eps','min_pt','sil']) 
print("Best Parameters:") 
pa[pa['sil']== pa['sil'].max()]