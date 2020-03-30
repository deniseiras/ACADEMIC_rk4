# ============================================================================================
# Programa para interpolação do COVID-19 utilizando métodos de Problema de Valor Inicial (PVI)
#
# autor: Denis M. de Almeida Eiras
# Data: 27/03/2020
#
#
# 1. Objetivo
# ~~~~~~~~~~~
#
# Executar o método de Runge-Kuta de 4a ordem (RK4) para demonstrar uma solução de
# Problema de Valor Inicial (PVI), prevendo dessa forma, a quantidade de casos de COVID-19 em
# um determinado dia.
#
#
# 2. Metodologia
# ~~~~~~~~~~~~~~
#
#  - Foram utillizados 3 dias de dados de COVID-19 para geração de um polinômio de 2a ordem,
#    os dias 28, 29 e 31.
#  - Teste 1: São executados N passos a partir do dia 29 para se prever o dia 30, usando três métodos:
#    Polinômio (solução exata), RK4 e, adicionalmente, Euler para efeito de comparação.
#  - Teste 2: São executados N passos a partir do dia 30 para se prever o dia 90 de COVID-19.
#
#    Para se avaliar o erro dos métodos RK4 com relação à solução exata, são feitas algumas
#    execuções. A cada execução, a quantidade de passos é aumentada para se avaliar a qualidade
#    da solução (erro).
#
# 2.1 Polinômio
#
# O polinômio  2,9417 x^2 - 0,6384x - 11,5930 foi gerado interpolando os dias 23, 24 e 26
# de março de 2020. Casos calculados pelo polinômio em 30 dias de COVID-19 (25/03) = 2616,78
#r
# 2.2 Função derivada
#
# A seguinte função foi derivada do polinômio para ser utilizada pelos métodos RK4 e Euler.
# y' = 5.8834 * x - 0.6384
# A função derivada está implementada na função f(x_n) no código.
#
# 2.3 Runge-Kuta de 4a ordem
#
# O método RK foi implementado na função def rk4(n_passos_, h, x, y), onde n_passos
# representa o número total de passos, h o tamanho do passos, x a matriz de valores em x e
# y a matriz de valores em y
#
# 2.4 Euler
#
# Analogamente ao método RK4, o método Euler foi implementado na função
# euler(n_passos_, h, x, y),
#
# 2.5 Dados do problema:
#
# Data                  23/03     24/03     25/03     26/03
# Dias de casos = x        28        29        30        31
# Casos Reais   = f(x)   1924      2247      2554      2985
# Fonte: ...
#
#
# 3. Resultados
# ~~~~~~~~~~~~~
#
# São gerados gráficos apresentando os resultados durante a execução do programa
#
# 4. Conclusão
#
# O teste 1 revelou que não se obteve um qualidade ótima na interpolação dos pontos. O aumento
# de passos revelou um aumento no erro. Uma maior quantidade de pontos na criação do polinômio
# pode ser uma solução.
#
# O teste 2 demonstra que, um aumento na quantidade de passos diminui o erro e traz uma ótima
# qualidade de representação de ambos os métodos.
#
# Apesar do método RK4 ser mais refinado que o método de Euler, os testes não demonstraram
# superioridade do método RK4.
#
# Outro teste feito, não exibido neste vídeo, utilizando um polinônio de quarta ordem, gerado
# utilizando-se diversos pontos, revelaram superioridade do método RK4.
#
# ============================================================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================================
# Função para inicializar o número de passos e os vetores x e y
# ============================================================================================
def initializeParams(x_max_, h,  x0, y0):
    num_passos = calc_num_passos(h, x0, x_max_)
    x = [None] * (num_passos + 2)
    x[0] = x0
    y = [None] * (num_passos + 2)
    y[0] = y0
    return num_passos, x, y


def calc_num_passos(h, x0, x_max_):
    num_passos = int((x_max_ - x0) / h)  # numero de passos
    return num_passos


# ============================================================================================
# Função derivada do PVI
# ============================================================================================
def f(x_n, y_n):

    # Utilização da derivada do polinômio 2,9417 x^2 - 0,6384x - 11,5930
    y_this = 5.8834 * x_n - 0.6384
    return y_this


# ============================================================================================
# Método de Euler
# ============================================================================================
def euler(num_passos_, h, x, y):

    for n in range(0, num_passos_ + 1):
        y[n + 1] = y[n] + h * f(x[n], y[n])
        x[n + 1] = x[n] + h

    return x, y


# ============================================================================================
# Método Runge-Kuta de 4a ordem (RK4)
# ============================================================================================
def rk4(num_passos_, h, x, y):

    for n in range(0, num_passos_ + 1):
        k1 = h * f(x[n], y[n])
        k2 = h * f(x[n] + h / 2, y[n] + k1 / 2)
        k3 = h * f(x[n] + h / 2, y[n] + k2 / 2)
        k4 = h * f(x[n] + h, y[n] + k3)

        y[n + 1] = y[n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[n + 1] = x[n] + h

    return x, y


def calc_polinomio(X_max, h, x0, y0):
    num_passos, x, y = initializeParams(X_max, h, x0, y0)
    print("X max = {}\nPasso = {}\nPassos = {}".format(X_max, h, num_passos))
    sol_exata_x_max_dias = 2.9417 * pow(X_max, 2) - 0.6384 * X_max - 11.5930
    print("Usando solução exata: Para x={} => y= {:.2f}".format(X_max, sol_exata_x_max_dias))
    for i in range(0, num_passos + 1):
        y[i + 1] = 2.9417 * pow(x[i], 2) - 0.6384 * x[i] - 11.5930
        if (i <= num_passos):
            x[i + 1] = x[i] + h
    # print("Usando equação diferencial exata:\nx = {}\ny = {}".format(x, y))
    plt.plot(x, y, label="Exata = {:.2f}".format(y[num_passos + 1]))
    return y[num_passos + 1]


def calc_euler(X_max, h, x0, y0, solucao_pol):
    num_passos, x, y = initializeParams(X_max, h, x0, y0)
    x, y = euler(num_passos, h, x, y)
    solucao_euler = y[num_passos + 1]
    print("Solução usando Euler: Para x={} => y= {:.2f}".format(X_max, y[num_passos + 1]))
    # print("Matriz Euler: \nx = {}\ny = {}".format(x, y))
    erro_euler = np.fabs((solucao_euler - solucao_pol) / solucao_pol)
    print("Erro_relativo euler = {:.2f}%".format(erro_euler * 100))
    plt.plot(x, y, label="Euller = {:.2f} - Erro = {:.2f}%".format(solucao_euler, erro_euler * 100))
    return solucao_euler


def calc_rk4(X_max, h, x0, y0, solucao_pol):
    num_passos, x, y = initializeParams(X_max, h, x0, y0)
    x, y = rk4(num_passos, h, x, y)
    solucao_rk4 = y[num_passos + 1]
    print("Solução usando RK4  : Para x={} => y= {:.2f}".format(X_max, y[num_passos + 1]))
    # print("Matriz RK4: \nx = {}\ny = {}".format(x, y))
    erro_rk4 = np.fabs((solucao_rk4 - solucao_pol) / solucao_pol)
    print("Erro_relativo rk4   = {:.2f}%".format(erro_rk4 * 100))
    plt.plot(x, y, label="RK4    = {:.2f} - Erro = {:.2f}%".format(solucao_rk4, erro_rk4 * 100))

    return solucao_rk4


def executa_testes(titulo):

    print("\n\n============================================================")
    solucao_pol = calc_polinomio(X_max, h, x0, y0)
    solucao_euler = calc_euler(X_max, h, x0, y0, solucao_pol)
    solucao_rk4 = calc_rk4(X_max, h, x0, y0, solucao_pol)
    plt.title("{} - Passos: {}".format(titulo, calc_num_passos(h, x0, X_max)))
    plt.legend()
    plt.show()


# ============================================================================================
# Função principal
# ============================================================================================
if __name__ == '__main__':


    # Primeiro teste
    # Interpolação de ponto interno ao intervalo do polinômio

    X_max = 30.0   # valor em x a atingir = dias de COVID-19
    x0 = 29.0      # valor inicial de x   = dias de COVID-19
    y0 = 2247.0  # valor inicial de y   = casos em 29 dias de COVID-19

    # h = passo em x
    for h in [1, 0.5, 0.1]:
        executa_testes("Num. casos interpolação")


    # Segundo Teste
    # Previsão de casos de COVID-19 no futuro, com passo crescente

    X_max = 90.0  # valor em x a atingir = dias de COVID-19
    x0 = 30.0     # valor inicial de x   = dias de COVID-19
    y0 = 2616.78   # valor inicial de y   = casos em 31 dias de COVID-19

    # h = passo em x
    for h in [10, 1, 0.1, 0.001]:
        executa_testes("Num. casos com 90 dias de COVID-19")