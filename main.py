from model.DecisionTree import DecisionTree
import numpy as np
import pandas as pd 

if __name__ == "__main__":

    df = pd.read_csv('./students.csv')
    feature_types = ['categorical' if type(i) == str else "real" for i in np.array(df.columns)[:-1] ]
    feature_types = np.array(feature_types)

    clf_custom = DecisionTree(feature_types=feature_types, criterion='gini')

    X = np.array(df[['STG', 'SCG', 'STR', 'LPR', 'PEG']])
    y = np.array(df[' UNS'])

    clf_custom.fit(X, y)
    output = clf_custom.predict(X)

    print(f"Точность реализованной модели на тренировочной борке: {np.sum(y == output) / output.shape[0] * 100: 0.1f}%")  