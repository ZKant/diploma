import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

sorted2 = [((121.14910047603507, 118.81210170451273), np.float64(85.3812102586887), np.float64(0.34793693919609664)), ((141.09375, 119.53125), np.float64(1.482317653203928), np.float64(0.0)), ((120.9375, 139.21875), np.float64(1.8114039611910115), np.float64(0.12130180081726674)), ((121.40625, 99.375), np.float64(1.1675212429686843), np.float64(0.168806722961079)), ((101.25, 119.0625), np.float64(0.9375), np.float64(0.0)), ((141.09375, 139.21875), np.float64(1.482317653203928), np.float64(0.0)), ((101.71875, 98.90625), np.float64(1.482317653203928), np.float64(0.0)), ((141.2784090909091, 100.02840909090908), np.float64(1.418542349276008), np.float64(0.14959052237654613)), ((100.3125, 139.21875), np.float64(1.1675212429686843), np.float64(0.168806722961079)), ((160.78125, 119.53125), np.float64(1.482317653203928), np.float64(0.0)), ((120.9375, 158.90625), np.float64(1.1675212429686843), np.float64(0.168806722961079)), ((121.59090909090908, 79.40340909090908), np.float64(1.418542349276008), np.float64(0.14959052237654613)), ((81.09375, 118.59375), np.float64(1.482317653203928), np.float64(0.0)), ((160.78125, 139.21875), np.float64(1.482317653203928), np.float64(0.0)), ((100.96590909090908, 158.1534090909091), np.float64(1.418542349276008), np.float64(0.14959052237654613)), ((82.03125, 98.90625), np.float64(1.482317653203928), np.float64(0.0)), ((142.03125, 80.15625), np.float64(1.482317653203928), np.float64(0.0)), ((102.1875, 79.21875), np.float64(1.1675212429686843), np.float64(0.168806722961079)), ((80.9090909090909, 138.4659090909091), np.float64(1.4185423492760063), np.float64(0.14959052237655168)), ((141.09375, 159.84375), np.float64(1.482317653203928), np.float64(0.0)), ((161.71875, 99.84375), np.float64(1.482317653203928), np.float64(0.0)), ((160.59659090909088, 159.65909090909088), np.float64(1.4185423492760012), np.float64(0.1495905223765682)), ((82.03125, 78.75), np.float64(1.1675212429686843), np.float64(0.168806722961079)), ((161.71875, 80.15625), np.float64(1.482317653203928), np.float64(0.0)), ((80.9090909090909, 158.72159090909088), np.float64(1.4185423492760028), np.float64(0.1495905223765627)), ((121.875, 59.24107142857143), np.float64(1.6757089581086961), np.float64(0.1907544044479934)), ((61.40625, 118.59375), np.float64(1.482317653203928), np.float64(0.0)), ((120.75892857142856, 179.0625), np.float64(1.6757089581086981), np.float64(0.190754404447989)), ((181.40625, 119.53125), np.float64(1.482317653203928), np.float64(0.0)), ((100.78125, 178.59375), np.float64(1.482317653203928), np.float64(0.0)), ((140.15625, 178.59375), np.float64(1.482317653203928), np.float64(0.0)), ((180.58823529411765, 139.48529411764702), np.float64(1.7174486486889418), np.float64(0.1860271832765338)), ((141.74107142857142, 59.0625), np.float64(1.6757089581086941), np.float64(0.19075440444799785)), ((181.11607142857142, 100.3125), np.float64(1.6757089581086941), np.float64(0.19075440444799785)), ((62.34375, 98.90625), np.float64(1.482317653203928), np.float64(0.0)), ((61.40625, 138.28125), np.float64(1.482317653203928), np.float64(0.0)), ((101.71875, 58.59375), np.float64(1.482317653203928), np.float64(0.0)), ((160.78125, 179.53125), np.float64(1.482317653203928), np.float64(0.0)), ((181.40625, 80.15625), np.float64(1.482317653203928), np.float64(0.0)), ((82.03125, 58.59375), np.float64(1.482317653203928), np.float64(0.0)), ((161.71875, 59.53125), np.float64(1.482317653203928), np.float64(0.0)), ((81.09375, 178.125), np.float64(1.8114039611910115), np.float64(0.12130180081726674)), ((180.58823529411765, 160.11029411764702), np.float64(1.7174486486889418), np.float64(0.1860271832765338)), ((62.15909090909091, 78.0965909090909), np.float64(1.4185423492760045), np.float64(0.14959052237655712)), ((60.9375, 157.5), np.float64(1.875), np.float64(0.0))]
mri_circles = sorted2

small_step = 20  # mm
medium_step1, medium_step2 = 25, 10  # mm
large_step = 50  # mm
x, y = 119.53125, 119.53125
# x, y = 119.35891804443419, 118.7860403825105 #но так ведь по сути тоже неправильно и надо задать другую точку, как раньше задавали
# x , y = 119.531
the_one = [(x, y), 85.8]

# Большие окружности с эталона
active_circles = [
    [(x + small_step, y + small_step), 1.5],
    [(x + small_step, y), 1.5],
    [(x, y + small_step), 1.5],
    [(x - small_step, y - small_step), 1.5],
    [(x, y - small_step), 1.5],
    [(x - small_step, y), 1.5],
    [(x + small_step, y - small_step), 1.5],
    [(x - small_step, y + small_step), 1.5],
    [(x + 2*small_step, y + 2*small_step), 1.5],
    [(x + 2*small_step, y), 1.5],
    [(x, y + 2*small_step), 1.5],
    [(x - 2*small_step, y - 2*small_step), 1.5],
    [(x, y - 2*small_step), 1.5],
    [(x - 2*small_step, y), 1.5],
    [(x + 2*small_step, y - 2*small_step), 1.5],
    [(x - 2*small_step, y + 2*small_step), 1.5],
    [(x + small_step, y + 2*small_step), 1.5],
    [(x + 2*small_step, y + small_step), 1.5],
    [(x - small_step, y - 2*small_step), 1.5],
    [(x - 2*small_step, y - small_step), 1.5],
    [(x + small_step, y - 2*small_step), 1.5],
    [(x + 2*small_step, y - small_step), 1.5],
    [(x - small_step, y + 2*small_step), 1.5],
    [(x - 2*small_step, y + small_step), 1.5],
    [(x - 2*small_step, y - 3*small_step), 1.5],
    [(x - small_step, y - 3*small_step), 1.5],
    [(x, y - 3*small_step), 1.5],
    [(x + small_step, y - 3*small_step), 1.5],
    [(x + 2*small_step, y - 3*small_step), 1.5],
    [(x - 2*small_step, y + 3*small_step), 1.5],
    [(x - small_step, y + 3*small_step), 1.5],
    [(x, y + 3*small_step), 1.5],
    [(x + small_step, y + 3*small_step), 1.5],
    [(x + 2*small_step, y + 3*small_step), 1.5],
    [(x + 3*small_step, y - 2*small_step), 1.5],
    [(x + 3*small_step, y - small_step), 1.5],
    [(x + 3*small_step, y), 1.5],
    [(x + 3*small_step, y + small_step), 1.5],
    [(x + 3*small_step, y + 2*small_step), 1.5],
    [(x - 3*small_step, y - 2*small_step), 1.5],
    [(x - 3*small_step, y - small_step), 1.5],
    [(x - 3*small_step, y), 1.5],
    [(x - 3*small_step, y + small_step), 1.5],
    [(x - 3*small_step, y + 2*small_step), 1.5]
]

# Список из списков с координатами
all_circles = [[the_one] + active_circles, mri_circles]

# Списки для хранения длин
lengths_lists = []

for idx, data_set in enumerate(all_circles, start=1):
    # Извлечение центра и радиуса большой окружности
    big_circle_center = data_set[0][0]

    # Центры маленьких окружностей
    small_centers = [circle[0] for circle in data_set[1:]]

    # Сортировка центров по расстоянию от центра большой окружности
    small_centers.sort(key=lambda center: np.linalg.norm(np.array(center) - np.array(big_circle_center)))

    def calculate_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    lents = []
    distances_from_center = []

    # Соединение точек (из вторых четырех) всеми возможными парами
    for start_point, end_point in combinations(small_centers, 2):
        line_length = calculate_distance(start_point, end_point)
        lents.append(line_length)

        # Подсчет суммарного расстояния точек отрезка до центра
        distance_from_center = (
                np.linalg.norm(np.array(start_point) - np.array((x, y))) +
                np.linalg.norm(np.array(end_point) - np.array((x, y)))
        )
        distances_from_center.append(distance_from_center)

    # Сортировка и сохранение длин
    lents.sort()
    lengths_lists.append(lents)
    
# Вычисление ошибки между двумя списками длин
if len(lengths_lists) == 2 and len(lengths_lists[0]) == len(lengths_lists[1]):
    errors = [abs(a - b) for a, b in zip(lengths_lists[0], lengths_lists[1])]
    print("Ошибки между двумя датасетами", errors)

    # Теперь сортируем ошибки по удаленности от центра
    errors_with_distances = list(zip(errors, distances_from_center))
    errors_with_distances.sort(key=lambda item: item[1])
    sorted_errors = [item[0] for item in errors_with_distances]


    import csv
    with open('C:/Users/k.zhukov/Desktop/Диплом/sorted_errors_4fant.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(sorted_errors)

    degree = 2  # Степень полинома
    coefficients = np.polyfit(range(1, len(sorted_errors) + 1), sorted_errors, degree)
    polynomial = np.poly1d(coefficients)

    # Создание аппроксимированной кривой
    x = np.arange(1, len(sorted_errors) + 1)
    approximated_curve = polynomial(x)

    # Отрисовка графика
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(sorted_errors) + 1), sorted_errors, 
            marker='o',
            linestyle='none',
            label='Ошибки')
    plt.plot(x, approximated_curve, color='red', label='Апроксимация')
    plt.title('Зависимость ошибки от удаленности точек(суммы удаленностей) от центра')
    plt.xlabel('Номер по мере удаления от центра большой окружности кожуха')
    plt.ylabel('Ошибка (расстояние)')
    plt.grid(True)
    plt.legend()
    plt.show()