from numpy import array
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
dados = array([[0,0],[0,1],[1,0],[1,1]])
resultado_esperado = array([
    0, 1, 1, 0
])
mlp = MLPClassifier(
    activation='logistic',
    max_iter=100,
    random_state=42,
    solver='adam'
)
mlp.fit(dados,resultado_esperado)
train_test_split(dados,resultado_esperado,test_size=0.2,random_state=42)
resultado_estimado = mlp.predict(dados)
print(resultado_estimado)
print(f'Precisão: {accuracy_score(resultado_esperado,resultado_estimado)}')