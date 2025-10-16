import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def calcula_A(y, x):
    m = sum((x - np.mean(x)) * (y - np.mean(y))) / sum((x - np.mean(x)) ** 2)
    return m

def calcula_b(y, x):
    b = np.mean(y) - calcula_A(y, x) * np.mean(x)
    return b

def calcula_erro(a,b,data_x,data_y):
    erro = 0
    for i in range(len(data_y)):
        erro += (a*data_x[i]+b - data_y[i])**2
    return erro

A = np.linspace(-10,10,100)
B = np.linspace(-5,5,100)

data = pd.read_excel('data.xlsx')
data_x = data['x']
data_y = data['y']
erros = np.zeros(shape = (100,100))
menor = calcula_erro(A[0],B[0],data_x, data_y)
valores = [A[0],B[0],menor]
for i,a in enumerate(A):
    for j,b in enumerate(B):
        erro = calcula_erro(a,b,data_x, data_y)
        erros[i][j] = erro
        if erro < menor:
            menor = erro
            valores = [a,b,menor]
print(valores)
a,b = valores[0], valores[1]
minimos_a, minimos_b = calcula_A(data_y,data_x), calcula_b(data_y,data_x)

plt.plot(data_x,data_y,'bo')
plt.plot(data_x,minimos_a*data_x+minimos_b,'r',label='Derivadas')
plt.plot(data_x,a*data_x + b,'g',label = 'Arrays A e B')
plt.legend()
plt.figure()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(A, B)
ax.plot_surface(X, Y, erros, cmap='viridis')
plt.plot(1.9191919191919187,4.3939393939393945,433.7149430915442,'ro')
plt.show()
print(valores)