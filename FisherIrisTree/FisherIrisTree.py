import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

df = pd.read_csv("C:/Users/St.Eva/Desktop/Projects Python/FisherIrisTree/IrisFisher.csv")

df.head()

df.drop('Id',axis=1,inplace=True)
df_X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
df_y = df.Species

df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y,test_size=0.2)

tree = DecisionTreeClassifier(max_depth=4)

tree.fit(df_X_train, df_y_train)

tree.score(df_X_test, df_y_test)

export_graphviz(tree, out_file='C:/Users/St.Eva/Desktop/Projects Python/FisherIrisTree/pics/tree_iris.dot', feature_names=df_X.columns, filled=True)
