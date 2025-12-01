import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('https://tinyurl.com/y6r7qjrp', delimiter=',')

inputs = df.values[:, :-1]
output = df.values[:, -1]

x_train, x_test, y_train, y_test = train_test_split(inputs, output, test_size=1/5)

nn = MLPClassifier(
    solver='sgd',
    hidden_layer_sizes=(10, ),
    activation='logistic',
    max_iter=1000_000,
    learning_rate_init=.05
)

nn.fit(x_train, y_train)

print(f"Accuracy train: {nn.score(x_train, y_train)}")
print(f"Accuracy test: {nn.score(x_test, y_test)}")

print('Matrix:')
print(confusion_matrix(y_test, nn.predict(x_test)))
