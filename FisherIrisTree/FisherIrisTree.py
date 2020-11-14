import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

datafile=pd.read_csv("C:/Users/St.Eva/Desktop/Projects Python/FisherIrisTree/IrisFisher.csv")
print (datafile.head())

x=datafile.iloc[:,:1]
y=datafile.iloc[:,1:]

#print (x.head())
#print (y.head())

#построим модель с помощью sklearn
from sklearn import tree
model=tree.DecisionTreeClassifier(criterion="entropy")
print(model.fit(x,y))
#оценка модели
print(model.score(x,y))
print('Правильность на обучающем наборе:{:.3f}'.format(model.score(x,y)))
print('Важность признаков: \n{}'.format(model.feature_importances_))
#прогнозирование с помощью модели
print('С данной длиной ирис будет отнесен к классу ')
print(model.predict([[12]]))
print('С данной длиной ирис будет отнесен к классу ')
print(model.predict([[29]]))
print('С данной длиной ирис будет отнесен к классу ')
print(model.predict([[49]]))
#
#визуализация модели
#print(tree.plot_tree(model))
