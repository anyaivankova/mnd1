from random import *

a_0 = 4
a_1 = 5
a_2 = 8
a_3 = 2

x_1 = sample(range(20), 8)
x_2 = sample(range(20), 8)
x_3 = sample(range(20), 8)

Y = []
for i in range(8):
    Y.append(a_0 + a_1*x_1[i] + a_2*x_2[i] + a_3*x_3[i])

x0_1 = (max(x_1) + min(x_1))/2
x0_2 = (max(x_2) + min(x_2))/2
x0_3 = (max(x_3) + min(x_3))/2

dx_1 = x0_1 - min(x_1)
dx_2 = x0_2 - min(x_2)
dx_3 = x0_3 - min(x_3)

xn_1 = []
for i in range(8):
    xn_1.append(round((x_1[i] - x0_1)/dx_1, 2))

xn_2 = []
for i in range(8):
    xn_2.append(round((x_2[i] - x0_2)/dx_2, 2))

xn_3 = []
for i in range(8):
    xn_3.append(round((x_3[i] - x0_3)/dx_3, 2))

print("a_0 = ", a_0, ", a_1 = ", a_1, ", a_2 = ", a_2, ", a_3 = ", a_3, "\n")
print("№  |  X1 |  X2  |  X3  |  Y  |  Xn_1  |  Xn_2  |  Xn_3")
for i in range(8):
    print(i, " | ", x_1[i], " | ", x_2[i], " | ", x_3[i], " | ", Y[i], " | ", xn_1[i], " | ", xn_2[i], " | ", xn_3[i])

print("\n", "X0 | ", x0_1, " | ", x0_2, " | ", x0_3, "\n", "dx | ", dx_1, " | ", dx_2, " | ", dx_3, "\n")

mid_Y = sum(Y)/len(Y)
print(mid_Y)
less_Y = []
for i in Y:
    if i < mid_Y:
        less_Y.append(i)
print(max(less_Y))