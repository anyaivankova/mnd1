from random import *
from numpy import linalg
from math import *

m = 5
x1_min = 10
x1_max = 60
x2_min = -70
x2_max = -10
y_max = (30-111)*10
y_min = (20-111)*10
x0 = 1
arr_x = [[-1, -1], [1, -1], [-1, 1]]
func = main_dev = sqrt((2*(2*m-2))/(m*(m-4)))


y_arr_1 = [randint(y_min, y_max) for i in range(5)]
y_arr_2 = [randint(y_min, y_max) for k in range(5)]
y_arr_3 = [randint(y_min, y_max) for n in range(5)]

y_avg_1 = sum(y_arr_1)/len(y_arr_1)
y_avg_2 = sum(y_arr_2)/len(y_arr_2)
y_avg_3 = sum(y_arr_3)/len(y_arr_3)

disp_1 = 0
for i in range(5):
    disp_1 += (y_avg_1 - y_arr_1[i])**2
disp_1 /= 5

disp_2 = 0
for i in range(5):
    disp_2 += (y_avg_2 - y_arr_2[i])**2
disp_2 /= 5

disp_3 = 0
for i in range(5):
    disp_3 += (y_avg_3 - y_arr_3[i])**2
disp_3 /= 5

disp_sum = disp_1 + disp_2 + disp_3
disp1_percent = disp_1/disp_sum
disp2_percent = disp_2/disp_sum
disp3_percent = disp_3/disp_sum

Fuv_1 = disp_1/disp_2
Fuv_2 = disp_3/disp_1
Fuv_3 = disp_3/disp_2

Ouv_1 = ((m - 2) / m) * Fuv_1
Ouv_2 = ((m - 2) / m) * Fuv_2
Ouv_3 = ((m - 2) / m) * Fuv_3

Ruv_1 = abs(Ouv_1 - 1) / func
Ruv_2 = abs(Ouv_2 - 1) / func
Ruv_3 = abs(Ouv_3 - 1) / func

mx1 = (arr_x[0][0] + arr_x[1][0] + arr_x[2][0]) / 3
mx2 = (arr_x[0][1] + arr_x[1][1] + arr_x[2][1]) / 3
my = (y_avg_1 + y_avg_2 + y_avg_3) / 3

a1 = ((arr_x[0][0])**2 + (arr_x[1][0])**2 + (arr_x[2][0])**2) / 3
a2 = (arr_x[0][0] * arr_x[0][1] + arr_x[1][0] * arr_x[1][1] + arr_x[2][0] * arr_x[2][1]) / 3
a3 = ((arr_x[0][1])**2 + (arr_x[1][1])**2 + (arr_x[2][1])**2) / 3
a11 = (arr_x[0][0] * y_avg_1 + arr_x[1][0] * y_avg_2 + arr_x[2][0] * y_avg_3) / 3
a22 = (arr_x[0][1] * y_avg_1 + arr_x[1][1] * y_avg_2 + arr_x[2][1] * y_avg_3) / 3

b0 = (linalg.det([[my, mx1, mx2],
                  [a11, a1, a2],
                  [a22, a2, a3]])) / (linalg.det([[1, mx1, mx2],
                                                  [mx1, a1, a2],
                                                  [mx2, a2, a3]]))

b1 = (linalg.det([[1, my, mx2],
                  [mx1, a11, a2],
                  [mx2, a22, a3]]))/(linalg.det([[1, mx1, mx2],
                                                [mx1, a1, a2],
                                                [mx2, a2, a3]]))

b2 = (linalg.det([[1, mx1, my],
                  [mx1, a1, a11],
                  [mx2, a2, a22]]))/(linalg.det([[1, mx1, mx2],
                                                [mx1, a1, a2],
                                                [mx2, a2, a3]]))


d_X1 = abs(x1_max-x1_min)/2
d_X2 = abs(x2_max-x2_min)/2
x_10 = (x1_max+x1_min)/2
x_20 = (x2_max+x2_min)/2
a0 = b0 - b1*x_10/d_X1-b2*x_20/d_X2
a1 = b1/d_X1
a2 = b2/d_X2

# output

print("y_min = " + str(y_min) + " y_max = " + str(y_max))

print("Experiment №1, Y: ", y_arr_1, ". Еxperiment №2, Y: ", y_arr_2, ". Еxperiment №3, Y: ", y_arr_3)
print("\nAverage value №1 Y: ", y_avg_1, ". Average value №2 Y: ", y_avg_2, ". Average value №3 Y: ", y_avg_3)

print("\nDispersion 1: ", round(disp_1, 2), " in percent:", round(disp1_percent,2))
print("Dispersion 2: ", round(disp_2,2), " in percent:", round(disp2_percent,2))
print("Dispersion 3: ", round(disp_3,2), " in percent:", round(disp3_percent,2))

print("\nFuv_1: ", round(Fuv_1,2), ". Fuv_2: ", round(Fuv_2,2), ". Fuv_3: ", round(Fuv_3,2))
print("\nOuv_1: ", round(Ouv_1,2), ". Ouv_2: ", round(Ouv_2,2), ". Ouv_3: ", round(Ouv_3,2))
print("\nRuv_1: ", round(Ruv_1,2), ". Ruv_2: ", round(Ruv_2,2), ". Ruv_3: ", round(Ruv_3,2))

print("\nmx1: ", round(mx1,2), ", mx2: ", round(mx2,2), ", my: ", round(my,2))
print("\na1 - ", round(a1,2), ", a2 - ", round(a2,2), ", a3 - ", round(a3,2), ", a11 - ", round(a11,2), ", a22 - ", round(a22,2))
print("\nb0 - ", round(b0,2), ", b1 - ", round(b1,2), ", b2 - ", round(b2,2))

print("\nNormalized regression equation: y=", round(b0,2), "+", round(b1,2), "*x1 +", round(b2,2), "*x2")
print("Naturalized regression equation \n y = ", round(a0,2), "+", round(a1,2), "*x1 +", round(a2,2), "*x2")
