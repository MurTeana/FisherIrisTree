import numpy as np
import numpy as npy

x=np.array([
    [1,1,1,1],
    [1,1,1,2],
    [2,1,1,1],
    [3,2,1,1],
    [3,3,2,1],
    [3,3,2,2],
    [2,3,2,2],
    [1,2,1,1],
    [1,3,2,1],
    [3,2,2,1],
    [1,2,2,2],
    [2,2,1,2],     
    [2,1,2,1], 
    [3,2,1,2]
    ])
y=npy.array([1,1,2,2,2,1,2,1,2,2,2,2,2,1])
from sklearn.tree import DesisionTreeClassifier
tree = DesisionTreeClassifier(random_state=0)
tree.fit(x,y)
print('--Правильность на обучающем наборе:{:.3f}'.format(tree.score(x,y)))
print('Важность признаков: \n{}'.format(tree.feauture_importance_))