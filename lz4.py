import random
from matplotlib import pyplot as plt
import numpy as np

def matrix(i,j):
    battle = [[6,-3],
              [-4,8]]
    return battle[i][j]

def rand(p):
    num = random.random()
    if num <= p:
        return 0
    else:
        return 1

def mat_exp(pA,pB):
    return (pA*pB * matrix(0,0) + pA*(1- pB) * matrix(0,1) + (1-pA)*pB*matrix(1,0) + (1 - pA)*(1 - pB) * matrix(1,1))

def sqr_mat_exp(pA,pB):
    return (pA * pB * matrix(0, 0)**2 + pA * (1 - pB) * matrix(0, 1)**2 + (1 - pA) * pB * matrix(1, 0)**2 + (1 - pA) * (1 - pB) * matrix(1, 1)**2)

def avg_dev(rounds, avg_round_result):
    dev = 0
    for i , j in rounds:
        dev += (matrix(i,j) - avg_round_result)**2
    dev /= 100
    dev = dev**(1/2)
    return dev


def dispersion(pA,pB):
    return sqr_mat_exp(pA,pB) - mat_exp(pA,pB)**2

def deviation(pA,pB):
    return dispersion(pA,pB)**(1/2)

def learning_r(pB):
    balls = [100 , 100]
    for i in range(1000):
        ch = rand(balls[0] / (sum(balls)))
        res = matrix(ch,rand(pB))
        if res > 0:
            balls[ch] += res
    print("Количество шаров: ", balls)
    return balls[0]/(sum(balls))

def learning_p(pB):
    balls = [10000 , 10000]
    for i in range(1000):
        ch = rand(balls[0] / (sum(balls)))
        res = matrix(ch,rand(pB))
        if res < 0:
            balls[ch] += res
    print("Количество шаров: ",balls)
    return balls[0]/(sum(balls))

def learning():
    ballsA = [10000, 10000]
    ballsB = [10000, 10000]
    for i in range(1000):
        chA = rand(ballsA[0] / (sum(ballsA)))
        chB = rand(ballsB[0] / (sum(ballsB)))
        res = matrix(chA, chB)
        ballsA[chA] += res
        ballsB[chB] -= res
    print("Количество шаров игрока А: ", ballsA)
    print("Количество шаров игрока Б: ", ballsB)
    return ballsA[0] / (sum(ballsA)) , ballsB[0] / (sum(ballsB))

def game():
    playerA = 0
    playerB = 0
    rounds = []
    pA , pB = learning()
    avg_round_result = 0
    N = 10000
    for i in range(N):
        rounds += [(rand(pA), rand(pB))]
        result = matrix(*rounds[i])
        playerA += result
        playerB -= result

    avg_round_result = playerA / N
    print("Вероятность игрока А: ", pA)
    print("Вероятность игрока Б: ", pB)
    print("игры: ", rounds)
    print("Очки игрока А:", playerA)
    print("Очки игрока Б: ", playerB)
    print("Среднее значение в игре: ", avg_round_result)
    print("Мат. ожидание: ", mat_exp(pA,pB))
    print("СКО: ", avg_dev(rounds, avg_round_result))
    print("Дисперсия: ", dispersion(pA,pB))
    print("Теор. СКО: ", deviation(pA,pB))

game()

# def func(pA,pB):
#     res =  (21*pA*pB) - (11*pA) - (12*pB) + 8
#     if res < 0:
#         return -1
#     return 1
#
# x = y = np.linspace(0, 1, 100)
# z = np.array([func(i, j) for j in y for i in x])
# Z = z.reshape(100, 100)
# plt.imshow(Z, interpolation='bilinear')
# plt.colorbar()
# plt.show()
