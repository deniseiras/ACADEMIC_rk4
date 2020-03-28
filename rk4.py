# ============================================================================================
# Programa para interpolação do COVID-19 utilizando métodos de Problema de Valor Inicial (PVI)
#
# autor: Denis M. de Almeida Eiras
# Data: 27/03/2020
#
# Objetivo: Executar os método de Runge-Kuta de 4a ordem (RK4) para prever casos de COVID-19.
# Realizar comparações com o método de Euller e com um polinômio gerado
#
# Metodologia:
#
#  - Foram utillizados 3 dias de dados de COVID-19 para geração de um polinômio de 2a ordem,
#    os dias 28, 29 e 31.
#  - São executados N passos a partir do dia 29 para se prever o dia 30, usando três métodos:
#    Polinômio (solução exata), RK4 e, adicionalmente, Euler para efeito de comparação.
#  - A quantidade de passos é incrementada para se avaliar o erro dos métodos RK4 com relação
#    à solução exata.
#
# Dados do problema:
#
# Data                  23/03     24/03     25/03     26/03
# Dias de casos = x        28        29        30        31
# Casos         = f(x)   1924      2247      2554      2985
# Gerado pelo poliômio                       2626,78
# Fonte: ...

#
# Resultados:
#  - .......
#  - ......
#  - São gerados gráficos para os três métodos

# ============================================================================================

import matplotlib.pyplot as plt


# ============================================================================================
# Função para inicializar o número de passos e os vetores x e y
# ============================================================================================
def initializeParams(x_max_, h,  x0, y0):
    n_max = int((x_max_ - x0) / h)  # numero de passos
    x = [None] * (n_max + 1)
    x[0] = x0
    y = [None] * (n_max + 1)
    y[0] = y0
    return n_max, x, y


# ============================================================================================
# Função derivada do PVI
# ============================================================================================
def f(x_n, y_n):

    # COVID-19
    # Solução exata utilizando polinômio de 4a ordem
    # Utilização da derivada do polinômio 2,9417 x^2 - 0,6384x - 11,5930
    y_this = y_n + (5.8834 * x_n) - 0.6384

    return y_this


# ============================================================================================
# Método de Euler
# ============================================================================================
def euler(n_max, h, x, y):

    for n in range(0, n_max):
        y[n + 1] = y[n] + h * f(x[n], y[n])
        x[n + 1] = x[n] + h

    return x, y

# ============================================================================================
# Método Runge-Kuta de 4a ordem (RK4)
# ============================================================================================
def rk4(n_max, h, x, y):

    for n in range(0, n_max):
        k1 = h * f(x[n], y[n])
        k2 = h * f(x[n] + h / 2, y[n] + k1 / 2)
        k3 = h * f(x[n] + h / 2, y[n] + k2 / 2)
        k4 = h * f(x[n] + h, y[n] + k3)

        y[n + 1] = y[n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[n + 1] = x[n] + h

    return x, y


# ============================================================================================
# Função principal
# ============================================================================================
if __name__ == '__main__':

    #
    # O polinômio  2,9417 x^2 - 0,6384x - 11,5930 foi gerado interpolando os dias 23, 24 e 26
    # de março de 2020.
    # casos calculados pelo polinômio em 30 dias de COVID-19 (25/03) = 2616,78


    X_max = 30   # valor em x a atingir = dias de COVID-19
    x0 = 29      # valor inicial de x   = dias de COVID-19
    y0 = 2247    # valor inicial de y   = casos em 29 dias de COVID-19
                 # Valor a prever em 30 dias y = 2616,78

    h = 0.1  # passo em x
    n_max, x, y = initializeParams(X_max, h, x0, y0)
    print("X max = {}\nPasso = {}\nPassos = {}".format(X_max, h, n_max))


    # ========== Exata ==========================================================================
    # Solução exata utilizando polinômio  2,9417 x^2 - 0,6384x - 11,5930
    sol_exata_x_max_dias = 2.9417 * pow(X_max, 2) - 0.6384 * X_max - 11.5930
    print("Usando solução exata: Para x={} => y= {}".format(X_max, sol_exata_x_max_dias))
    for i in range(0, n_max+1):
        y[i] = 2.9417 * pow(x[i], 2) - 0.6384 * x[i] - 11.5930
        if(i < n_max):
            x[i+1] = x[i] + h
    print("Usando equação diferencial exata:\nx = {}\ny = {}".format(x, y))
    plt.text(x[n_max], y[n_max], str(y[n_max]))
    plt.plot(x, y, label="Exata = " + str(y[n_max]))

    # ========== Euler ==========================================================================
    n_max, x, y = initializeParams(X_max, h, x0, y0)
    x, y = euler(n_max, h, x, y)
    print("Usando Euler: \nx = {}\ny = {}".format(x, y))
    plt.plot(x, y, label="Euller = " + str(y[n_max]))

    # ========== RK4 ==========================================================================
    n_max, x, y = initializeParams(X_max, h, x0, y0)
    x, y = rk4(n_max, h, x, y)
    print("Usando Runge-Kuta de 4a ordem: \nx = {}\ny = {}".format(x, y))
    plt.plot(x, y, label="RK4   = " + str(y[n_max]))
    plt.legend()
    plt.show()
