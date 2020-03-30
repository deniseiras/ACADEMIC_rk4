import numpy as np
# import math
import matplotlib.pyplot as plt


def f(x_n, y_n):
    # y_this = -y_n + x_n + 2
    # Exercicio 10.2.1 - livro-py.pdf
    y_this = -0.5*y_n + x_n + 2
    # derivada da solução exata
    # y_this = 2 - 8 * np.exp(-x_n/2)
    return y_this


def euler(n_max, h, x, y):

    for n in range(0, n_max):
        y[n + 1] = y[n] + h * f(x[n], y[n])
        x[n + 1] = x[n] + h

    return x, y

def rk4(n_max, h, x, y):

    for n in range(0, n_max):
        k1 = h * f(x[n], y[n])
        k2 = h * f(x[n] + h / 2, y[n] + k1 / 2)
        k3 = h * f(x[n] + h / 2, y[n] + k2 / 2)
        k4 = h * f(x[n] + h, y[n] + k3)

        y[n + 1] = y[n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[n + 1] = x[n] + h

    return x, y


def initializeParams(x_max_, h,  x0, y0):
    n_max = int(x_max_ / h)  # numero de passos
    x = [None] * (n_max + 1)
    x[0] = x0
    y = [None] * (n_max + 1)
    y[0] = y0
    return n_max, x, y


if __name__ == '__main__':
    X_max = 1.0       # max valor em x
    h = 0.5         # passo em x
    x0 = 0.0
    y0 = 8.0

    n_max, x, y = initializeParams(X_max, h, x0, y0)
    print("X max = {}\nPasso = {}\nPassos = {}".format(X_max, h, n_max))
    sol_exata = 2 + 8 * np.exp(-0.5)
    print("Usando solução exata: Para x=1 => y= {}".format(sol_exata))
    for i in range(0, n_max+1):
        y[i] = 2 * x[i] + 8 * np.exp(-x[i]/2)
        if(i < n_max):
            x[i+1] = x[i] + h

    print("Usando equação diferencial exata:\nx = {}\ny = {}".format(x, y))
    plt.text(x[n_max], y[n_max], str(y[n_max]))
    plt.plot(x, y, label="Exata = " + str(y[n_max]))

    x0 = 0.0
    n_max, x, y = initializeParams(X_max, h, x0, y0)
    x, y = euler(n_max, h, x, y)
    print("Usando Euler: \nx = {}\ny = {}".format(x, y))
    plt.plot(x, y, label="Euller = " + str(y[n_max]))

    n_max, x, y = initializeParams(X_max, h, x0, y0)
    x, y = rk4(n_max, h, x, y)
    print("Usando Runge-Kuta de 4a ordem: \nx = {}\ny = {}".format(x, y))
    # plt.plot(x, y, label="RK4   = " + str(y[n_max]))
    plt.legend()
    plt.show()
