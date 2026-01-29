from numpy import array

#Recomendação de treino com base em idade e horas de sono

dados = [[20,8],
         [25,6],
         [30,7],
         [22,9],
         [35,5],
         [40,6],
         [28,7],
         [21,8]]
resultado_esperado = [
    1, 0, 0, 1, 0, 0, 1, 1
]
dados,resultado_esperado = array(dados),array(resultado_esperado)
peso1, peso2, eta = 0.5, -0.5, 0.8

def formula(peso1,peso2,item1,item2):
    resultado = (peso1 * item1) + (peso2 * item2) + eta
    if resultado > 0:
        resultado = 1
    elif resultado < 0:
        resultado = 0
    else:
        resultado = 0.5
    return resultado

def resultados(item1,item2,d):
    resultado_estimado = formula(peso1,peso2,item1,item2)
    print(f'Entrada: {int(item1),int(item2)}')
    print(f'Resultado Esperado: {d}')
    print(f'Saída Estimada: {resultado_estimado}')
    print('Status:','Correto' if d == resultado_estimado else 'Errado')
    print('---'*12)
    return resultado_estimado

def prever(dados, peso1, peso2):
    previsoes = []
    for item1, item2 in dados:
        previsoes.append(formula(peso1, peso2, item1, item2))
    return previsoes

def acuracia(dados, resultados_esperados, peso1, peso2, eta):
    previsoes = prever(dados, peso1, peso2, eta)
    acertos = sum(1 for y_true, y_pred in zip(resultados_esperados, previsoes) if y_true == y_pred)
    return acertos / len(resultados_esperados)

treinando = True
while treinando:
    treinando = False
    for (item1,item2),d in zip(dados,resultado_esperado):
        resultado_estimado = resultados(item1,item2,d)
        erro = d - resultado_estimado
        if erro != 0:
            treinando = True
            peso1 = peso1 + eta * erro * item1
            peso2 = peso2 + eta * erro * item2
print('Fim do treino')


dados_novos = [
    [26, 7],
    [33, 6],
    [19, 9],
]
resultados_novos = [1, 0, 1]

print("Acurácia nos dados novos:", acuracia(dados_novos, resultados_novos, peso1, peso2, eta))
