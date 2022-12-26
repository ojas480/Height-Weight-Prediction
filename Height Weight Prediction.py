import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('SOCR-HeightWeight.csv')

X = df[['Height(Inches)']]
y = df['Weight(Pounds)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

model = DecisionTreeRegressor()

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

# print(f'Test score: {score:.2f}')


# [70] can be replaced with any value of a height 
y_pred = model.predict([[70]])
print(f'Predicted weight: {y_pred[0]:.2f} pounds')