import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Carregar dataset
df = pd.read_csv('Iris.csv')

# Separar features e labels
df.drop("Id", axis=1, inplace=True)
df["Species"] = df["Species"].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
X = df.drop(["Species"], axis=1).to_numpy()
y = df.loc[:, "Species"].to_numpy()

print(X)
print(y)

# Separar dataset em teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Criando modelo
clf = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)

# treino do modelo
clf.fit(X_train, y_train)

# Predição 
clf_pred = clf.predict(X_test)
aa = clf.predict(X)
print(aa)

# Métricas
clf_accuracy = accuracy_score(y_test, clf_pred)
print(classification_report(y_test, clf_pred))

# pickle model (binário)
pickle.dump(clf, open('model.pkl', 'wb'))



