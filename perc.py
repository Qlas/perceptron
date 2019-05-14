import numpy as np
from random import uniform, shuffle
import matplotlib.pyplot as plt


def rate():
    while True:
        try:
            rate = float(input("wspolczynnik uczenia sie:\n"))
            if 0 < rate < 1:
                break
            else:
                print("Podaj wartosc od 0 do 1")
        except ValueError:
            print("Podaj liczbe")
    return rate


def perceptron(dataset, n_epoch, rate, deci):

    def predict(row, weights, deci):
        activation = weights[0]
        for i in range(len(row) - 1):
            activation += weights[i + 1] * row[i]
        return 1 if activation >= 0 else deci

    def plto(weights,epoch, deci):
        x = np.arange(-1+deci, 2, 0.1)
        y = []
        for i in range(len(x)):
            y.append(-(weights[1] * x[i] + weights[0]) / weights[2])
        plt.plot(x, y)
        plt.plot(deci, deci, marker="o", markersize=3, color="red")
        plt.plot(deci, 1, marker="o", markersize=3, color="red")
        plt.plot(1, deci, marker="o", markersize=3, color="red")
        plt.plot(1, 1, marker="o", markersize=3, color="red")
        plt.grid(True)
        plt.title(epoch)
        plt.show()

    # unipolarny/bipolarny
    if int(deci) == 1:
        deci = 0
    else:
        deci = -1
    deci = int(deci)

    weights = [uniform(-1,1), uniform(-1,1), uniform(-1,1)]

    for epoch in range(n_epoch):
        print("Epoka: ", epoch+1)
        z = np.arange(4)
        print("b = ", weights[0], "w1 = ", weights[1], "w2 = ", weights[2])

        # wyswietlanie
        for i in range(len(dataset)):
            row = dataset[z[i]]
            prediction = predict(row, weights, deci)
            print("x1= %d, x2= %d, Spodziewane= %d, Przewidywane= %d" % (row[0], row[1], row[-1], prediction))
        plto(weights, epoch + 1, deci)

        shuffle(z)
        check = 0
        q = weights[0]
        for i in range(len(dataset)):
            row = dataset[z[i]]
            prediction = predict(row, weights, deci)

            # sprawdzamy wynik przewidywania
            if prediction == row[-1]:
                check += 1

            error = row[-1] - prediction
            for j in range(len(row)-1):
                weights[j+1] += rate * error * row[j]
            weights[0] += rate * error

        # stop
        if int(check) == 4 and q == weights[0]:
            break
        if epoch+1 == n_epoch:
            print("Perceptron nie nauczyl sie")
            return

    # testowanie
    while True:
        try:
            first = int(input("Podaj wartosc pierwsze liczby:\n"))
            if first != 1 and first != deci:
                print("Wartosc nie jest", deci,"ani 1")
                continue
            second = int(input("Podaj wartosc drugiej liczby:\n"))
        except ValueError:
            continue
        if second != 1 and second != deci:
            print("Wartosc nie jest", deci, "ani 1")
            continue

        prediction = predict([first,second,0],weights, deci)
        print("Wynik:", prediction)
        try:
            cont = int(input("Continue? (0;1):\n"))
        except ValueError:
            break
        if cont == 0:
            break


while True:
    try:
        deci = input("Podaj jaka funkcje chcesz brac pod uwage:\n1 - unipolarna\n2 - bipolarna\n")
        if int(deci) == 1:
            dataset = [[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]]
            i = 0
            while True:
                print("Podaj y[0;1] dla", dataset[i])
                try:
                    y = int(input())
                    if y == 0 or y == 1:
                        dataset[i].append(y)
                        i += 1
                    if i == 4:
                        break
                except ValueError:
                    print("Podaj liczbe")
            break

        elif int(deci) == 2:
            dataset = [[-1, -1],
                       [-1, 1],
                       [1, -1],
                       [1, 1]]
            i = 0
            while True:
                print("Podaj y[-1;1] dla", dataset[i])
                try:
                    y = int(input())
                    if y == -1 or y == 1:
                        dataset[i].append(y)
                        i += 1
                    if i == 4:
                        break
                except ValueError:
                    print("Podaj liczbe")
            break

        else:
            print("Podaj wartosc 1;2")

    except ValueError:
        print("Podaj liczbe")

n_epoch = 20
perceptron(dataset, n_epoch, rate(), deci)


