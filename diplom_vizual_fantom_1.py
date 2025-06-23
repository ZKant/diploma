import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import seaborn as sns
from scipy import stats
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.interpolate import make_interp_spline
from itertools import combinations
from tqdm import tqdm
from datetime import datetime


#Функция для определения того находится ли точка внутри окружности с заданным радиусом и цетром
def is_point_in_circle(x, y, circle_x, circle_y, radius):
    distance = math.sqrt((x - circle_x) ** 2 + (y - circle_y) ** 2)
    return distance <= radius


#Функция для расчета среднего центра и радиуса, надо заменить все это на свое, чтобы потестить
def calculate_average_center_and_radius(contour):
    if len(contour) == 0:
        return None, None

    points = np.squeeze(contour)

    if points.ndim == 1:
        points = np.expand_dims(points, axis=0)

    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    center = (center_x, center_y)

    radii = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)

    average_radius = np.max(radii)
    std_rad = np.std(radii)

    return center, average_radius, std_rad


#Функции для поворота и сдвига(поменять на итерованный сдвиг аксиального среза начиная с первого(от кожуха фантома до конца)
def translate(x, y, center):
    return x - center[0], y - center[1]


def rotate(x, y, angle):
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    product = np.dot(matrix, np.array([[x], [y]]))
    return product.ravel()[0], product.ravel()[1]


#Функция для сортировки точек по мере удаления от центра снимка
#,так как предполагаем что искажений больше по краям от изоцентра фантома
def sort_by_distance_with_radius(points, reference_point):
    """
    Сортирует массив точек с радиусами в порядке увеличения расстояния от заданной точки.
    
    :param points: Список элементов [(x, y), radius], где (x, y) — координаты точки, radius — радиус.
    :param reference_point: Кортеж (x, y), заданная точка для расчета расстояния.
    :return: Отсортированный список элементов [(x, y), radius].
    """
    def distance(point1, point2):
        # Вычисление Евклидова расстояния
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    # Сортировка по расстоянию до reference_point
    sorted_points = sorted(points, key=lambda item: distance(item[0], reference_point))
    return sorted_points


#Сортировка на основе минимальной оценки соответствия
def sort_by_min_distance_and_radius(array1, array2):
    """
    Сортирует второй массив по минимальному расстоянию между координатами и радиусом,
    чтобы соответствовать элементам первого массива.

    :param array1: Список элементов [(x, y), radius] — первый массив.
    :param array2: Список элементов [(x, y), radius] — второй массив.
    :return: Отсортированный второй массив.
    """
    def distance(point1, point2):
        # Вычисление Евклидова расстояния между двумя точками
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def match_score(item1, item2):
        # Вычисление "оценки соответствия" по расстоянию и разнице радиусов
        coord_distance = distance(item1[0], item2[0])
        radius_difference = abs(item1[1] - item2[1])
        return coord_distance + radius_difference  # Можно добавить веса, если расстояние или радиус важнее
    
    # Создаём копию второго массива, чтобы исключать элементы при сопоставлении
    remaining_array2 = array2[:]
    sorted_array2 = []

    for item1 in array1:
        # Ищем элемент из второго массива с минимальной "оценкой соответствия"
        best_match = min(remaining_array2, key=lambda item2: match_score(item1, item2))
        sorted_array2.append(best_match)
        remaining_array2.remove(best_match)  # Удаляем элемент, чтобы он не повторялся

    return sorted_array2

# ===========вычисление векторной ошибки с угловой частью================ #
# ==================================================================== #
# ==================================================================== #

def calculate_errors(array1, array2, center=(127.5, 127.5)):
    """
    Расширенный метод 1: объединяет радиальную и тангенциальную ошибки.
    
    :param array1: Эталонные данные [(x, y), radius].
    :param array2: Измеренные данные [(x, y), radius, std_rad].
    :param center: Центр фантома (cx, cy).
    :return: Объединенные ошибки, радиальные, тангенциальные, радиусы.
    """

    coordinate_errors = []
    radial_errors = []
    tangential_errors = []
    radius_errors = []
    std_rad_list = []
    radius_list = []
    # rast_errors = []


    if len(array1) != len(array2):
        raise ValueError("Массивы должны быть одинаковой длины")

    for (coord1, radius1), (coord2, radius2, std_rad) in zip(array1, array2):
        r1 = math.hypot(coord1[0] - center[0], coord1[1] - center[1])
        r2 = math.hypot(coord2[0] - center[0], coord2[1] - center[1])
        theta1 = math.atan2(coord1[1] - center[1], coord1[0] - center[0])
        theta2 = math.atan2(coord2[1] - center[1], coord2[0] - center[0])


        # Тангенциальная ошибка (в мм)
        delta_theta = (theta2 - theta1)
        delta_r = r2 * math.cos(delta_theta) - r1
        delta_L = r2 * math.sin(delta_theta)

        total_error = math.sqrt(delta_r**2 + delta_L**2)

        
        coordinate_errors.append(total_error)


        # Вычисление ошибки радиуса (модуль разности радиусов)
        # radius_error = abs(radius2 - radius1)
        # вычисление ошибки с учетом направления 
        radius_error = (radius2 - radius1)
        radius_errors.append(radius_error)
        std_rad_list.append(std_rad)
        radius_list.append(radius2)
        radial_errors.append(delta_r)
        tangential_errors.append(delta_L)


    return coordinate_errors, radius_errors, std_rad_list, radius_list, radial_errors, tangential_errors


########################################################################################################
#Читаем dicom файлы
dicom_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_3/Фантом_3_снимки'
output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_3/Фантом_3_результаты'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Весь_фантом/Минимальный_радиус_вект_ошибка'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Весь_фантом/Максимальный_радиус_вект_ошибка'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Маленькие_контуры/Максимальный_радиус_вект_ошибка'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Весь_фантом/Максимальный_радиус_старая_ошибка'

# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Большие контуры/Средний_радиус_вект_ошибка'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Большие контуры/Минимальный_радиус_вект_ошибка'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Большие контуры/Максимальный_радиус_вект_ошибка'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Большие контуры/Средний_радиус_старая_ошибка'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Большие контуры/Максимальный_радиус_старая_ошибка'
########################################################################################################

#Читаем все файлы подряд от нижнего аксиального среза до верхнего аксиального среза
dicom_files = sorted([f for f in os.listdir(dicom_dir) if f.endswith('.dcm')])

full_errors = [] #средняя ошибка радиусов по срезу 
full_errors_xy = []
matrix = [] #матрица ошибок для хитмапа
matrix_xy = []
matrix_std = []
matrix_rad = []
centers0 = []
big_circles_cent = []
matrix_rast = []

spacing = 0.9375
# x, y = (127.5*spacing), (127.5*spacing)
# x, y = 119.40408595234895, 118.74743218077447 #первый фантом
# x, y = 119.38193401132555, 118.81577680187988 #первый фантом  

# x, y = 121.1616162087226, 118.82485024219974 #второй фантом

x, y = 119.3275781325184, 119.99951990113935 #третий фантом
# x, y = (127.726*spacing), (127.531*spacing)
# big_circles_cent.append((largest_circle_center_x, largest_circle_center_y))
# print('-'*50)
# print(big_circles_cent)
# print('-'*50)
# x, y = largest_circle_center_x, largest_circle_center_y #включить сюда центры окружности которая нашлась выше
# print(x, y)

shift_x = 0
shift_y = 0
x = x + shift_x
y = y + shift_y
small_step = (20) #mm
medium_step1, medium_step2 = (25), (10) #mm
large_step = (50) #mm
# x, y = 127.5, 127.5
the_one = [(x , y), 85.8] #Большая окружность(кожух фантома) диаметром 171.6мм

rads = 3

# Размечаем маленькие окружности с чертежа диаметром 3мм
small_circles = [
    [(x + small_step, y + small_step), rads],
    [(x + small_step, y), rads],
    [(x, y + small_step), rads],
    [(x - small_step, y - small_step), rads],
    [(x, y - small_step), rads],
    [(x - small_step, y), rads],
    [(x + small_step, y - small_step), rads],
    [(x - small_step, y + small_step), rads],
    [(x + 2*small_step, y + 2*small_step), rads],
    [(x + 2*small_step, y), rads],
    [(x, y + 2*small_step), rads],
    [(x - 2*small_step, y - 2*small_step), rads],
    [(x, y - 2*small_step), rads],
    [(x - 2*small_step, y), rads],
    [(x + 2*small_step, y - 2*small_step), rads],
    [(x - 2*small_step, y + 2*small_step), rads],
    [(x + small_step, y + 2*small_step), rads],
    [(x + 2*small_step, y + small_step), rads],
    [(x - small_step, y - 2*small_step), rads],
    [(x - 2*small_step, y - small_step), rads],
    [(x + small_step, y - 2*small_step), rads],
    [(x + 2*small_step, y - small_step), rads],
    [(x - small_step, y + 2*small_step), rads],
    [(x - 2*small_step, y + small_step), rads],
    [(x - 2*small_step, y - 3*small_step), rads],
    [(x - small_step, y - 3*small_step), rads],
    [(x, y - 3*small_step), rads],
    [(x + small_step, y - 3*small_step), rads],
    [(x + 2*small_step, y - 3*small_step), rads],
    [(x - 2*small_step, y + 3*small_step), rads],
    [(x - small_step, y + 3*small_step), rads],
    [(x, y + 3*small_step), rads],
    [(x + small_step, y + 3*small_step), rads],
    [(x + 2*small_step, y + 3*small_step), rads],
    [(x + 3*small_step, y - 2*small_step), rads],
    [(x + 3*small_step, y - small_step), rads],
    [(x + 3*small_step, y), rads],
    [(x + 3*small_step, y + small_step), rads],
    [(x + 3*small_step, y + 2*small_step), rads],
    [(x - 3*small_step, y - 2*small_step), rads],
    [(x - 3*small_step, y - small_step), rads],
    [(x - 3*small_step, y), rads],
    [(x - 3*small_step, y + small_step), rads],
    [(x - 3*small_step, y + 2*small_step), rads]
]

# Размечаем средние окружности(ближе к краям, диаметром 5.9мм) с чертежа
medium_circles = [
    [(x + large_step, y + large_step), 2.95],
    [(x - large_step, y + large_step), 2.95],
    [(x + large_step, y - large_step), 2.95],
    [(x - large_step, y - large_step), 2.95]
]
# Размечаем большие окружности с чертежа(диаметром 10мм)
large_circles = [
    [(x + medium_step1, y - medium_step2), 5],
    [(x + medium_step1 + (15), y - medium_step2), 5],
    [(x + medium_step1 + (45), y - medium_step2), 5],
    [(x + medium_step2, y + medium_step1), 5],
    [(x + medium_step2, y + medium_step1 + (15)), 5],
    [(x + medium_step2, y + medium_step1 + (45)), 5],
    [(x - medium_step1, y + medium_step2), 5],
    [(x - medium_step1 - (15), y + medium_step2), 5],
    [(x - medium_step1 - (45), y + medium_step2), 5],
    [(x - medium_step2, y - medium_step1), 5],
    [(x - medium_step2, y - medium_step1 - (15)), 5],
    [(x - medium_step2, y - medium_step1 - (45)), 5],
]
# Все окружности с чертежа
all_circles = [the_one] + small_circles
# large_circles + small_circles + medium_circles
# large_circles + small_circles + medium_circles
# + small_circles + medium_circles + large_circles +


# Читаем каждый файл и сохраняем картинки оконтуривания для каждого среза в папку
for n, dicom_file in tqdm(enumerate(dicom_files, start=1)):
    dicom_file_path = os.path.join(dicom_dir, dicom_file)
    dataset = pydicom.dcmread(dicom_file_path)
    # Извлечение пиксельных данных
    pixel_array = dataset.pixel_array
    # нормализация
    image = 255 * (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
    image = image.astype(np.uint8)
    # ПРОСМОТР тут на uint 16
    # Нормализация пиксельных значений к диапазону 0-255
    # image = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
    # image = np.uint8(image)

    # Применение сглаживания Гаусса, для снижения шума, но лучше без него? 
    # blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # blurred = cv2.fastNlMeansDenoising(image)
    blurred = image
    # Применение порогового преобразования (Оцу)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Поиск контуров
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy.astype(np.uint16) #gпереводим обратно
    # Второй метод для поиска кругов на снимке(метод Хафа), настроить параметры
    # contours = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
                                #  param1=50, param2=30, minRadius=1, maxRadius=50)
    # Создание копии изображения для рисования контуров
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Рисование контуров
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 1)


# Задаем массивы центр+радиус, радиус, центр
    centers_and_radii = []
    radiuses = []
    centers = []

    i = 0 #счетчик контуров
    lya = np.array([127.5, 127.5])


# для минимальной описывающей окр
# потестить для нее и вообще мб без нее и без ЦМ

    # for cnt in contours:
    #     # Находим минимальную окружность, которая охватывает контур
    #     (x, y), radius = cv2.minEnclosingCircle(cnt)

    #     # Приводим координаты к масштабу
    #     x = x * spacing
    #     y = y * spacing
    #     radius = radius * spacing
    #     std_rad = 0

    #     if is_point_in_circle(x, y, 127.5*spacing, 127.5*spacing, (87)) == True and radius < 200 and radius > 1:
    #         # x, y = translate(x, y, lya)
    #         # x, y = rotate(x, y, -n*(0.007)*math.pi/180)
    #         # x, y = translate(x, y, -lya)
    #         radiuses.append(radius)
    #         center = (x, y)
    #         # print(radius)
    #         # print(center)
    #         buf = int(round(x)), int(round(y))  # Округляем центр
    #         # buf = x, y
    #         # print(buf)
    #         centers_and_radii.append((center, radius, std_rad))
    #         radius1 = int(round(radius))  # Приводим радиус к int
    #         centers.append(center)
    #         i += 1
    #         # cv2.circle(output_image, buf, radius1, (0, 0, 255), 1)    
    # # ///////////////////////////////////////////////////
    #     # Фильтрация по условиям
    #     if is_point_in_circle(x, y, 127.5 * spacing, 127.5 * spacing, 85.8) and 1 < radius < 200:
    #         radiuses.append(radius)
    #         center = (x, y)
    #         # print(center)
    #         buf = int(round(x/0.9375)), int(round(y/0.9375))  # Округляем центр
    #         centers_and_radii.append(((x, y), radius))
    #         radius1 = int(round(radius/0.9375))  # Приводим радиус к int
    #         centers.append(center)
    #         print(centers)
    #         # Рисуем окружность на выходном изображении
    #         # cv2.circle(output_image, buf, radius1, (0, 0, 255), 1)



# # а можно таким способом : с помощью функции `calculate_average_center_and_radius()`
#     for cnt in contours:
#         center, radius, std_rad = calculate_average_center_and_radius(cnt)
#         if center is not None and radius is not None:
#             # Приводим координаты и радиус к масштабу
#             x = center[0] * spacing
#             y = center[1] * spacing
#             radius = radius * spacing
#         if is_point_in_circle(x, y, 127.5*spacing, 127.5*spacing, (87)) == True and radius < 200 and radius > 1:
#             # x, y = translate(x, y, lya)
#             # x, y = rotate(x, y, -n*(0.007)*math.pi/180)
#             # x, y = translate(x, y, -lya)
#             radiuses.append(radius)
#             center = (x, y)
#             # print(radius)
#             # print(center)
#             buf = int(round(x)), int(round(y))  # Округляем центр
#             # buf = x, y
#             # print(buf)
#             centers_and_radii.append((center, radius, std_rad))
#             radius1 = int(round(radius))  # Приводим радиус к int
#             centers.append(center)
#             i += 1
#             cv2.circle(output_image, buf, radius1, (0, 0, 255), 1)     
    
    
    
    # for cnt in contours:
    #     centr, avg_radius = calculate_average_center_and_radius(cnt)
    #     print(centr)

    #     center = tuple(x * spacing for x in centr)

    #     avg_radius = avg_radius * spacing
        
    #     # print(center)

    #     if center is None or avg_radius is None:
    #         continue
    
    #     # buf = int(round(center)), int(round(y)
    #     radiuses.append(avg_radius)
    #     centers_and_radii.append((center, avg_radius))
    #     avg_radius = int(avg_radius)
    #     centers.append(center)
    #     i += 1
    
        # buf = (int(center[0]), int(center[1]))
        # cv2.circle(output_image, buf, avg_radius, (0, 0, 255), 1)

    # width, height = image.shape[1], image.shape[0]
    # dwg = svgwrite.Drawing(f"/home/kirill_zh/folder_py/Results/contours_with_circles{n}.svg", size=(width, height))


#Через взвешивание контура и поиск центра масс контура 
# (+возможна апроксимация контура-потестить как будет меняться ошибка от этого
# ...................................
    for cnt in contours:
        # Аппроксимация контура для повышения точности
        # epsilon = 0.01 * cv2.arcLength(cnt, True)
        # approx_contour = cv2.approxPolyDP(cnt, epsilon, True)
    
        # Найдем центр масс контура
        M = cv2.moments(cnt)
        if M["m00"] != 0:  # Чтобы избежать деления на ноль
            x = M["m10"] / M["m00"]
            y = M["m01"] / M["m00"]
        else:
            x, y = 0, 0
        # (x, y), radius = cv2.minEnclosingCircle(cnt)

        # rect = cv2.minAreaRect(cnt)
        # но можно и так сделать, так тоже будет прикольно и правильно в целом 
        # вопрос почему? и можно добавить это в мой диплом по идее тоже как один из методов
        # center, size, angle = rect

        # # Координаты центра
        # x, y = center
    
        # Вычисляем радиус как расстояние до самой удаленной точки на контуре
        # radius = max(np.linalg.norm(np.array((x, y)) - np.array(pt[0])) for pt in approx_contour)
        distances = []
        for point in cnt:
            distance = np.sqrt((point[0][0] - x) ** 2 + (point[0][1] - y) ** 2)
            distances.append(distance)
    
        radius = np.mean(distances)*spacing #тут макс
        rad_std = np.std(distances)*spacing
        x = x*spacing
        y = y*spacing

        # x, y = rotate(x, y, n*(0.0005)*math.pi/180)
        lya = np.array([127.5*spacing, 127.5*spacing])
        # Фильтрация по условиям
        if is_point_in_circle(x, y, 127.5*spacing, 127.5*spacing, (87)) == True and radius < 200 and radius > 1:
            x, y = translate(x, y, lya)
            x, y = rotate(x, y, (3.4)*math.pi/180)
            # 3.4 третий, 1.4 второй, 
            x, y = translate(x, y, -lya)
            radiuses.append(radius)
            center = (x, y)
            # print(radius)
            print(center)
            buf = int(round(x/spacing)), int(round(y/spacing))  # Округляем центр
            # buf = x, y
            # print(buf)
            centers_and_radii.append((center, radius, rad_std))
            radius1 = int(round(radius/spacing))  # Приводим радиус к int
            centers.append(center)
            i += 1
            cv2.circle(output_image, buf, radius1, (0, 0, 255), 1)    
# .....................................................................


# ===================================================================================================================================================
# Берем параметры большой окружности - краев фантома для совмещени их с центром эталон
    if centers_and_radii:  # Проверяем, что список не пуст
    # Ищем элемент с максимальным радиусом
        max_circle = max(centers_and_radii, key=lambda item: item[1])
        
        # Извлекаем координаты центра и радиус
        largest_circle_center_x, largest_circle_center_y = max_circle[0]  # Координаты центра
        largest_circle_radius = max_circle[1]  # Радиус
    else:
    # Если список пуст, задаём значения по умолчанию
        largest_circle_center_x, largest_circle_center_y, largest_circle_radius = None, None, None

    print(largest_circle_center_x, largest_circle_center_y, largest_circle_radius)
    print('*'*50)
    big_circles_cent.append((largest_circle_center_x, largest_circle_center_y))

# Теперь выстраиваем эталонны контур согласно размерам с чертежа
    # small_circles + medium_circles + large_circles
# Функция для поиска расстояния между точками
    output_with_reference = image.copy()

    # Рисование ЭТАЛОННЫХ контуров поверх найденных
    for circle in all_circles:
        (x_ref_mm, y_ref_mm), radius_ref_mm = circle
        
        # Конвертация мм -> пиксели
        x_ref_px = x_ref_mm / spacing
        y_ref_px = y_ref_mm / spacing
        radius_ref_px = radius_ref_mm / spacing
        
        # Определение стиля для разных типов контуров
        if radius_ref_mm == 85.8:  # Основной контур фантома
            color = (255, 255, 0)  # Голубой (BGR)
            thickness = 2
        elif radius_ref_mm == 1:  # Маленькие стержни
            color = (0, 0, 255)    # Красный (BGR)
            thickness = 1
        else:                      # Остальные
            color = (255, 0, 0)    # Синий (BGR)
            thickness = 1
        
        # Рисование окружности
        cv2.circle(
            output_with_reference,
            (int(x_ref_px), int(y_ref_px)),
            int(radius_ref_px),
            color,
            thickness
        )


    plt.figure(figsize=(18, 6))  # Увеличиваем ширину для трёх изображений

    # Исходное изображение
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Исходное (Срез №{n})")
    plt.axis('off')

    # Выделенные контуры
    plt.subplot(1, 3, 2)
    plt.imshow(output_image)
    plt.title(f"Найденные контуры (Срез №{n})")
    plt.axis('off')

    # Контуры с эталонами
    plt.subplot(1, 3, 3)
    plt.imshow(output_with_reference)
    plt.title(f"Сравнение с эталоном (Срез №{n})")
    plt.axis('off')

    plt.savefig(os.path.join(output_dir, f"Контуры_сравнение_{n}.png"))
    plt.close()


    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    # centers[0] = (127.5*spacing, 127.5*spacing)
    # Сортировка и расчет ошибок для радиуса и координаты    
    acc_sorted_points = sort_by_distance_with_radius(all_circles, the_one[0]) #((127.5*spacing), (127.5*spacing))

    nonacc_sorted_points = sort_by_min_distance_and_radius(acc_sorted_points, centers_and_radii)
    
    # print(acc_sorted_points, nonacc_sorted_points)

    # print(f"sorted1 = {acc_sorted_points}, sorted2 = {nonacc_sorted_points}")


# ====================ПОГРЕШНОСТЬ РАСССТОЯНИЙ=================

    def calculate_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    all_circles_2 = [acc_sorted_points, nonacc_sorted_points]
    # target_dist = 20.0
    # target_distance = 30.0  # Эталонное расстояние в мм
    # target_distance_1 = 15.0
    # target_distance_2 = 38.0788655293
    # target_distance_3 = 58.3095189485
    # target_distance_4 = 100
    # tolerance = 0.1 

    big_circle_centers = []
    small_centers_list = [] 

    for data_set in all_circles_2:
        big_circle_centers.append(data_set[0][0])  # Центр большой окружности
        small_centers_list.append([circle[0] for circle in data_set[1:]])  # Центры малых окружностей

    # Сопоставляем центры между наборами
    small_centers_acc = small_centers_list[0]  # Эталонные центры
    small_centers_nonacc = small_centers_list[1]  # Сопоставляемые центры
    matched_nonacc_centers = []  # Результат сопоставления
    used_indices = set()

    for acc_center in small_centers_acc:
        min_dist = float('inf')
        min_index = None
        for j, nonacc_center in enumerate(small_centers_nonacc):
            if j in used_indices:
                continue
            dist = calculate_distance(acc_center, nonacc_center)
            if dist < min_dist:
                min_dist = dist
                min_index = j
        if min_index is not None:
            matched_nonacc_centers.append(small_centers_nonacc[min_index])
            used_indices.add(min_index)

    # Собираем пары с расстоянием ~20 мм в эталоне
    pairs_ref = [] 
    errors = []  # Ошибки расстояний
    dist_sums = []  # Суммарные расстояния для сортировки
    big_center_acc = big_circle_centers[0]  # Центр большой окружности эталона

# Перебираем все уникальные пары в эталоне
    for i in range(len(small_centers_acc)):
        for j in range(i + 1, len(small_centers_acc)):
            point1_acc = small_centers_acc[i]
            point2_acc = small_centers_acc[j]


            # Проверяем расстояние в эталоне
            distance_acc = calculate_distance(point1_acc, point2_acc)
            # if abs(distance_acc - target_dist) > tolerance: #and abs(distance_acc - target_distance) > tolerance and abs(distance_acc - target_distance_1) > tolerance and abs(distance_acc - target_distance_2) > tolerance and abs(distance_acc - target_distance_3) > tolerance and abs(distance_acc - target_distance_4) > tolerance:
            #     continue  # Пропускаем если не 20±0.1 мм

            pairs_ref.append((point1_acc, point2_acc))
            
            # Берем соответствующие точки в nonacc
            point1_nonacc = matched_nonacc_centers[i]
            point2_nonacc = matched_nonacc_centers[j]

            
            # Вычисляем ошибку (отклонение от 20 мм)
            distance_nonacc = calculate_distance(point1_nonacc, point2_nonacc)

            error = (distance_nonacc - distance_acc)
            # error = abs(distance_nonacc - target_distance)
            # if distance_nonacc < 17.5:
            #     error = (distance_nonacc - target_distance_1)
            # elif distance_nonacc < 25.0 and distance_nonacc > 17.5:
            #     error = (distance_nonacc - target_dist)
            # elif distance_nonacc > 25.0 and distance_nonacc < 35.0:
            #     error = (distance_nonacc - target_distance)
            # elif distance_nonacc > 35.0 and distance_nonacc < 45.0:
            #      error = (distance_nonacc - target_distance_2)
            # elif distance_nonacc > 45.00 and distance_nonacc < 80.0:
            #     error = (distance_nonacc - target_distance_3)
            # else:
            #     error = (distance_nonacc - target_distance_4)


            # print('-------------------')
            # print(distance_nonacc, target_distance)
            # print('-------------------')

            
            # Рассчитываем dist_sum (сумма расстояний до центра большой окружности)
            dist_sum = (
                calculate_distance(point1_acc, big_center_acc) +
                calculate_distance(point2_acc, big_center_acc) +
                calculate_distance(point1_acc, point2_acc)
            )

            norm_sum = calculate_distance(point1_acc, point2_acc)

            error = 100*(error / calculate_distance(point1_acc, point2_acc))
            
            errors.append(error)
            dist_sums.append(dist_sum)

    print("длина")
    print(len(pairs_ref))

# Сортируем ошибки по dist_sum
    error_dist_pairs = sorted(zip(errors, dist_sums), key=lambda x: x[1])
    sorted_errors_rast = [error for error, _ in error_dist_pairs]


    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    ax.set_aspect('equal')
    height = 255 * spacing

    BIG_CIRCLE_STYLE = {'color': '#A0A0A0', 'alpha': 0.5, 'linewidth': 1}  # Блеклый серый
    SMALL_CIRCLE_STYLE = {'color': '#505050', 'alpha': 0.8, 'linewidth': 1}  # Темный серый
    CENTER_DOT_STYLE = {'color': '#303030', 's': 15, 'alpha': 0.9, 'zorder': 3}

    for idx, circle in enumerate(all_circles):
        center, radius = circle
        center_display = (center[0], height - center[1])
        
        # Стиль в зависимости от типа круга
        style = BIG_CIRCLE_STYLE if idx == 0 else SMALL_CIRCLE_STYLE
        circle_patch = Circle(center_display, radius, fill=False, **style)
        ax.add_patch(circle_patch)
        
        # Для маленьких кругов добавляем центральную точку
        if idx > 0:
            ax.scatter(*center_display, **CENTER_DOT_STYLE)

    # Отладочная информация
    print(f"Срез {n}: Найдено {len(errors)} ошибок расстояний")
    if not errors:
        print("Нет ошибок расстояний для отображения!")
    else:
        print(f"Min ошибка: {min(errors):.2f} мм, Max ошибка: {max(errors):.2f} мм")

    # Определяем min и max ошибки для нормализации
    if errors:
        min_error = min(errors)
        max_error = max(errors)
    else:
        min_error = 0
        max_error = 1  # Значение по умолчанию

    norm = plt.Normalize(vmin=min_error, vmax=max_error)
    cmap = plt.cm.mpl.cm.YlOrRd

    # 1. Рисуем фон - все эталонные контуры
    for circle in all_circles:
        center, radius = circle
        center_display = (center[0], height - center[1])
        circle_patch = Circle(center_display, radius, fill=False, color='#CCCCCC', alpha=0.9, linewidth=1)
        ax.add_patch(circle_patch)

# 2. Рисуем толстые цветные линии для пар
    if errors:
        for idx, ((p1_acc, p2_acc), error) in enumerate(zip(pairs_ref, errors)):
            try:
                # Получаем соответствующие найденные центры
                i = small_centers_acc.index(p1_acc)
                j = small_centers_acc.index(p2_acc)
                p1_nonacc = matched_nonacc_centers[i]
                p2_nonacc = matched_nonacc_centers[j]
                
                # Конвертация координат для отображения
                p1_display = (p1_nonacc[0], height - p1_nonacc[1])
                p2_display = (p2_nonacc[0], height - p2_nonacc[1])
                
                # Цвет линии в зависимости от величины ошибки
                line_color = cmap(norm(error))
                
                # Рисуем толстую линию
                line = plt.Line2D(
                    [p1_display[0], p2_display[0]], 
                    [p1_display[1], p2_display[1]],
                    color=line_color,
                    linewidth=12,  # Фиксированная толщина
                    alpha=0.9,
                    solid_capstyle='round',
                    zorder=10  # Высокий приоритет отрисовки
                )
                ax.add_line(line)
                
                # Вычисляем середину линии для подписи
                mid_x = (p1_display[0] + p2_display[0]) / 2
                mid_y = (p1_display[1] + p2_display[1]) / 2
                
                # Подпись с контрастным оформлением
                ax.text(mid_x, mid_y, f"{error:.2f}", 
                        fontsize=8, fontweight='bold',
                        ha='center', va='center',
                        color='white',
                        bbox=dict(
                            boxstyle='round,pad=0.2',
                            facecolor='black',
                            alpha=0.8,
                            edgecolor='none'
                        ),
                        zorder=20  # Самый высокий приоритет
                )
                
                print(f"Линия #{idx+1}: ({p1_display[0]:.1f},{p1_display[1]:.1f}) - ({p2_display[0]:.1f},{p2_display[1]:.1f}), Ошибка: {error:.2f} мм")
                
            except Exception as e:
                print(f"Ошибка при отрисовке линии #{idx+1}: {str(e)}")

    # 3. Настройка осей
    ax.set_xlim(0, 255 * spacing)
    ax.set_ylim(0, 255 * spacing)
    plt.title(f"Тепловая карта ошибок расстояний (срез №{n})", fontsize=14, fontweight='bold')
    plt.axis('off')

    # 4. Добавляем цветовую шкалу
    if errors:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(
            sm, 
            ax=ax, 
            orientation='horizontal', 
            fraction=0.035,      # Уменьшенная высота
            pad=0.0001,            # Уменьшенный отступ от графика
            # location='top',       # Расположена сверху
            shrink=0.6,          # Укороченная длина
            aspect=25             # Узкий профиль
        )
        cbar.set_label('Ошибка расстояния, мм', 
                    fontsize=10, 
                    fontweight='bold',
                    labelpad=8)  # Отступ текста
        cbar.ax.tick_params(labelsize=8, pad=2)  # Компактные метки
        # cbar.ax.xaxis.set_ticks_position('top')  # Метки сверху

    # Настройка границ для лучшего размещения
    plt.subplots_adjust(top=0.92, bottom=0.08)

    # Сохранение
    plt.savefig(os.path.join(output_dir, f"Тепловая_карта_ошибок_расстояний_срез_{n}.png"), 
                dpi=200, bbox_inches='tight')
    plt.close()

# конец погрешностностей по расстояниямм.
        
    xy_errors, radius_errors, std_rad, radis, radials_err, tang_err = [], [], [], [], [], []
    xy_errors, radius_errors, std_rad, radis, radials_err, tang_err = calculate_errors(acc_sorted_points, nonacc_sorted_points)

    print(radius_errors)

    plt.figure(figsize=(14, 14), dpi=150)
    ax = plt.gca()
    ax.set_aspect('equal')
    
    # Получаем эталонные позиции и радиусы
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import matplotlib as mpl
    import matplotlib.patheffects as patheffects

    reference_positions = [item[0] for item in acc_sorted_points]
    reference_radii = [item[1] for item in acc_sorted_points]

    height = 255 * spacing  # Рассчитываем высоту изображения
    reference_positions = [(x, height - y) for (x, y) in reference_positions]

    # Нормализация ошибок для цветовой карты
    norm = plt.Normalize(vmin=min(radius_errors), vmax=max(radius_errors))
    cmap = mpl.cm.YlOrRd  # Более контрастная палитра
    text_cmap = mpl.cm.binary  # Используем "желто-оранжево-красную" палитру

    max_radius = max(reference_radii)
    ii = 1
    for (x, y), r_ref, err in zip(reference_positions, reference_radii, radius_errors):
        if r_ref == max_radius:
            current_alpha = 0.15   # Фон! Очень нежный уровень прозрачности
        else:
            current_alpha = 1.0    # Основные круги полностью видимы
        if r_ref == 5:
            spc = 1.05
        elif r_ref == 85.8:
            spc = 1
        elif r_ref == 2.95:
            spc = 1.3
        else:
            spc = 1.9
        

        circle = Circle(
        (x, y), 
        r_ref*spc, 
        fill=True, 
        color=cmap(norm(err)),
        alpha=1,
        linewidth=0          # Убираем обводку
        )
        ax.add_patch(circle)
        # plt.text(x, y, f'{err:.2f}', ha='center', va='center', fontsize=12, color='#555555', fontweight='bold')
        # # plt.text(x, y, f'{ii}', ha='center', va='center', fontsize=15, color='green', fontweight='bold')
        # ii += 1
        bg_color = cmap(norm(err))
        # Вычисляем яркость фона
        bg_brightness = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        text_color = 'black' if bg_brightness > 0.6 else 'white'
        
        plt.text(
            x, y, 
            f'{err:.2f}', 
            ha='center', 
            va='center', 
            fontsize=14,  # Увеличенный размер шрифта
            fontweight='bold',
            color=text_color,
            path_effects=[
                patheffects.withStroke(
                    linewidth=2, 
                    foreground='black' if text_color == 'white' else 'white'
                )
            ]
        )
        ii += 1



    # Настройка границ и цветовой шкалы
    ax.set_xlim(0, 255 * spacing)
    ax.set_ylim(0, 255 * spacing)
    plt.title(f'Тепловая карта ошибок радиуса (Срез №{n})')

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax = inset_axes(
        ax,
        width="50%",  # Ширина
        height="10%",  # Высота
        loc='lower center',
        bbox_to_anchor=(0, 0.01, 1, 0.2),  # Сдвиг ближе к графику
        bbox_transform=ax.transAxes
    )
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation='horizontal'
    )

    cb.ax.tick_params(which='both', width=2, labelsize=14)
    for label in cb.ax.get_xticklabels():
        label.set_fontweight('bold')
    cb.set_label('Величина погрешности', fontsize=16, fontweight='bold', labelpad=8)
    cb.ax.tick_params(labelsize=14, width=2) 

    # Оформление
    ax.set_facecolor('white')
    # plt.gca().invert_yaxis()

    # Убираем рамки (оси)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout(pad=3.0)

    # Сохранение
    plt.savefig(os.path.join(output_dir, f"Тепловая_карта_срез_№{n}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # cb = plt.colorbar(
    #     plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
    #     ax=ax,
    #     fraction=0.046, pad=0.04
    # )
    # cb.set_label('Величина отклонений')

    # ax.set_facecolor('white')
    # plt.gca().invert_yaxis()   # Для соответствия ориентации DICOM
    # plt.tight_layout()

    # # Убираем рамки (оси)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    
    # # Сохранение
    # plt.savefig(os.path.join(output_dir, f"Тепловая_карта_срез_№{n}.png"), dpi=150)
   
    # plt.close()

# Задание оси для графиков, показывающий номер стержня по мере удаления от центра изоцентра фантома
    y_axis = []
    for i in range (0, len(xy_errors)):
        y_axis.append(i)

# Считаем полную ошибку для среза по радиусу и координате(можно еще для каждого стержня отдельно(это будет на хитмапе у меня)
    full_error = sum(radius_errors) / len(radius_errors)

    full_error_xy = sum(xy_errors) / len(xy_errors)

    # print(radius_errors)
    full_errors.append(full_error)
    full_errors_xy.append(full_error_xy)
    print(dicom_file_path)


    # Для графиков ошибок на каждом срезе
    plt.figure(figsize=(12, 6))

    # График ошибок радиуса
    plt.subplot(1, 2, 1)
    plt.plot(y_axis, radius_errors, 'b-o', markersize=5, linewidth=1, label='Ошибки')

    # Аппроксимация
    if len(y_axis) > 2:  # Минимум 3 точки для полинома 2-й степени
        x = np.array(range(1, len(radius_errors)+1))
        y = np.array(radius_errors)
        coefficients = np.polyfit(x, y, 3)
        polynomial = np.poly1d(coefficients)
        x_vals = np.linspace(min(x), max(x), 100)
        plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2, label='Тренд')

    plt.title(f"Ошибка радиуса (Срез №{n})")
    plt.xlabel('Номер стержня')
    plt.ylabel('Ошибка, мм')
    plt.grid(True)
    plt.legend()

    # График ошибок координат
    plt.subplot(1, 2, 2)
    plt.plot(y_axis, xy_errors, 'b-o', markersize=5, linewidth=1, label='Ошибки')

    # Аппроксимация
    if len(y_axis) > 2:
        x = np.array(range(1, len(xy_errors)+1))
        y = np.array(xy_errors)
        coefficients = np.polyfit(x, y, 3)
        polynomial = np.poly1d(coefficients)
        x_vals = np.linspace(min(x), max(x), 100)
        plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2, label='Тренд')

    plt.title(f"Ошибка координат (Срез №{n})")
    plt.xlabel('Номер стержня')
    plt.ylabel('Ошибка, мм')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Ошибки_на_срезе_график№{n}.png"))
    plt.close()
    # В секции графиков для каждого среза:

    radis = radis

    matrix.append(radius_errors)
    matrix_xy.append(xy_errors)
    matrix_std.append(std_rad)
    matrix_rad.append(radis)
    matrix_rast.append(sorted_errors_rast)
    

#   centers0.append(centers[0])

# print(centers0)

error_matrix = np.array(matrix)
np.savetxt(f"{output_dir}/Матрица_ошибок_по_радиусу_средний_новое.csv", error_matrix, delimiter=";", fmt="%.5f")
print("Матрица ошибок по радиусу")
print(error_matrix)

error_matrix_xy = np.array(matrix_xy)
np.savetxt(f"{output_dir}/Матрица ошибок по координате_средний_новое.csv", error_matrix_xy, delimiter=";", fmt="%.5f")
print("Матрица ошибок по координате")
print(error_matrix_xy)

stdrad_matrix = np.array(matrix_std)
np.savetxt(f"{output_dir}/Матрица СКО_средний_новое.csv", stdrad_matrix, delimiter=";", fmt="%.5f")
print("Матрица СКО")
print(stdrad_matrix)

rad_matrix = np.array(matrix_rad)
np.savetxt(f"{output_dir}/Матрица радиусов_средний_новое.csv", rad_matrix, delimiter=";", fmt="%.5f")
print("Матрица радиусов")
print(rad_matrix)

matrix_rast = np.array(matrix_rast)
np.savetxt(f"{output_dir}/Матрица ошибок по расстояниям_средний_новое.csv", matrix_rast, delimiter=";", fmt="%.5f")
print("Матрица ошибок по расстояниям")
print(matrix_rast)


reference_centers = [circle[0] for circle in all_circles[1:]]  # Центры стержней
phantom_center = all_circles[0][0]  # Центр фантома (the_one)

distances = [np.linalg.norm(np.array(phantom_center) - np.array(ref_center)) 
             for ref_center in reference_centers]

rounded_distances = np.round(distances, 1)
unique_dists, group_indices = np.unique(rounded_distances, return_inverse=True)
n_groups = len(unique_dists)

error_matrix_rods = error_matrix[:, 1:]  # Ошибки радиуса для стержней
error_matrix_xy_rods = error_matrix_xy[:, 1:]

group_radius_errors = np.zeros((error_matrix_rods.shape[0], n_groups))
group_radius_errors_count = np.zeros((error_matrix_rods.shape[0], n_groups))
group_xy_errors = np.zeros((error_matrix_xy_rods.shape[0], n_groups))
group_xy_errors_count = np.zeros((error_matrix_xy_rods.shape[0], n_groups))

# Заполнение групповых матриц
for i in range(error_matrix_rods.shape[0]):  # По срезам
    for j in range(error_matrix_rods.shape[1]):  # По стержням
        g = group_indices[j]
        group_radius_errors[i, g] += error_matrix_rods[i, j]
        group_xy_errors[i, g] += error_matrix_xy_rods[i, j]
        group_radius_errors_count[i, g] += 1
        group_xy_errors_count[i, g] += 1

# Усреднение ошибок в группах
group_radius_errors /= np.where(group_radius_errors_count > 0, group_radius_errors_count, 1)
group_xy_errors /= np.where(group_xy_errors_count > 0, group_xy_errors_count, 1)


average_values = np.mean(error_matrix, axis=0)
average_values_xy = np.mean(error_matrix_xy, axis=0)

# Номера столбцов
column_indices = np.arange(1, error_matrix.shape[1] + 1)
column_indices_xy = np.arange(1, error_matrix_xy.shape[1] + 1)

# Получаем радиусы для каждого стержня (без фантома)
rod_radii = [circle[1] for circle in all_circles[1:]]

# Для каждой группы собираем радиусы всех стержней в этой группе
group_radii = []
for g in range(n_groups):
    idxs = np.where(group_indices == g)[0]  # Индексы стержней группы
    radii = [rod_radii[i] for i in idxs]
    group_radii.append(radii)

# Формируем подписи для оси X: Расстояние (r=...)
xticklabels = []

xts = []
for dist, radii in zip(unique_dists, group_radii):
    radii_str = ', '.join([str(np.round(r, 1)) for r in radii])
    xticklabels.append(f"{dist} мм (r={radii_str})")
    xts.append(radii_str)

print(xts)




# 1. Тепловая карта ошибок радиуса по группам
plt.figure(figsize=(14, 12))
sns.heatmap(
    group_radius_errors,
    cmap="coolwarm",
    cbar=True,
    linewidths=0.5,
    linecolor='white',
    xticklabels=xticklabels ,  # Расстояния групп
    yticklabels=10,
    annot=False,
)
plt.title("Ошибки радиуса по группам стержней", fontsize=14, fontweight='bold')
plt.xlabel("Расстояние от центра группы, мм", fontsize=12)
plt.ylabel("№ среза", fontsize=12)
plt.xticks(fontsize=9, rotation=45)
plt.yticks(fontsize=9)
plt.savefig(os.path.join(output_dir, "Тепловая_карта_ошибок_радиуса_группы.png"), bbox_inches='tight')
plt.close()

# 2. Тепловая карта ошибок координат по группам
plt.figure(figsize=(14, 12))
sns.heatmap(
    group_xy_errors,
    cmap="coolwarm",
    cbar=True,
    linewidths=0.5,
    linecolor='white',
    xticklabels=np.round(unique_dists, 1),
    yticklabels=10,
    annot=False,
)
plt.title("Ошибки координат по группам стержней", fontsize=14, fontweight='bold')
plt.xlabel("Расстояние от центра группы, мм", fontsize=12)
plt.ylabel("№ среза", fontsize=12)
plt.xticks(fontsize=9, rotation=45)
plt.yticks(fontsize=9)
plt.savefig(os.path.join(output_dir, "Тепловая_карта_ошибок_координат_группы.png"), bbox_inches='tight')
plt.close()

# 3. График средней ошибки радиуса по группам
mean_radius_group = np.mean(group_radius_errors, axis=0)
std_radius_group = np.std(group_radius_errors, axis=0)

plt.figure(figsize=(10, 6))
plt.errorbar(
    unique_dists, 
    mean_radius_group,
    yerr=std_radius_group,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=2
)
plt.title("Средняя ошибка радиуса по группам", fontsize=14)
plt.xlabel("Расстояние от центра группы, мм", fontsize=12)
plt.ylabel("Ошибка радиуса, мм", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Средняя_ошибка_радиуса_по_группам.png"))
plt.close()

# 4. График средней ошибки координат по группам
mean_xy_group = np.mean(group_xy_errors, axis=0)
std_xy_group = np.std(group_xy_errors, axis=0)

plt.figure(figsize=(10, 6))
plt.errorbar(
    unique_dists, 
    mean_xy_group,
    yerr=std_xy_group,
    fmt='o-',
    capsize=5,
    ecolor='gray',
    elinewidth=2
)
plt.title("Средняя ошибка координат по группам", fontsize=14)
plt.xlabel("Расстояние от центра группы, мм", fontsize=12)
plt.ylabel("Ошибка координат, мм", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Средняя_ошибка_координат_по_группам.png"))
plt.close()

# # Построение графика
# plt.plot(column_indices, average_values, marker='o')
# plt.title('Средняя ошибка радиуса по стрежню')
# plt.xlabel('Номер стержня')
# plt.ylabel('Ошибка, мм')
# plt.grid(True)
# plt.savefig(os.path.join(output_dir, "Средняя_ошибка_радиуса_по_стержню.png"))
# plt.close()

# # Построение графика для координаты
# plt.plot(column_indices_xy, average_values_xy, marker='o')
# plt.title('Средняя ошибка координат по стрежню')
# plt.xlabel('Номер стержня')
# plt.ylabel('Ошибка, мм')
# plt.grid(True)
# plt.savefig(os.path.join(output_dir, "Средняя_ошибка_координат_по_стрежню.png"))
# plt.close()


plt.figure(figsize=(12, 6))

# Средняя ошибка радиуса
plt.subplot(1, 2, 1)
plt.plot(column_indices, average_values, 'b-o', markersize=5, linewidth=1)

if len(column_indices) > 2:
    x = np.array(column_indices)
    y = np.array(average_values)
    coefficients = np.polyfit(x, y, 3)
    polynomial = np.poly1d(coefficients)
    x_vals = np.linspace(min(x), max(x), 100)
    plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2)

plt.title('Средняя ошибка стержня по радиусу от n стержня')
plt.xlabel('Номер стержня')
plt.ylabel('Ошибка, мм')
plt.grid(True)

# Средняя ошибка координат
plt.subplot(1, 2, 2)
plt.plot(column_indices_xy, average_values_xy, 'b-o', markersize=5, linewidth=1)

if len(column_indices_xy) > 2:
    x = np.array(column_indices_xy)
    y = np.array(average_values_xy)
    coefficients = np.polyfit(x, y, 3)
    polynomial = np.poly1d(coefficients)
    x_vals = np.linspace(min(x), max(x), 100)
    plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2)


plt.title('Средняя ошибка стержня по координате от n стержня')
plt.xlabel('Номер стержня')
plt.ylabel('Ошибка, мм')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Средние_ошибки_по_стержню.png"))
plt.close()





from scipy.ndimage import zoom
smooth_error_matrix = zoom(error_matrix, zoom=4)  # Увеличиваем разрешение
smooth_error_matrix_xy = zoom(error_matrix_xy, zoom=4)


GROUP_SIZE = 7  # Объединяем по 5 столбцов
n_groups = (matrix_rast.shape[1] + GROUP_SIZE - 1) // GROUP_SIZE  # Рассчитываем количество групп

# Создаем агрегированную матрицу (108, n_groups)
agg_matrix = np.zeros((matrix_rast.shape[0], n_groups))

x_labels = []
for group_idx in range(n_groups):
    start_col = group_idx * GROUP_SIZE
    end_col = min((group_idx + 1) * GROUP_SIZE, matrix_rast.shape[1])
    
    # Агрегируем данные (среднее по группе)
    agg_matrix[:, group_idx] = np.mean(matrix_rast[:, start_col:end_col], axis=1)
    
    # Формируем понятные подписи
    if end_col - start_col == 1:
        x_labels.append(f"{start_col+1}")
    else:
        x_labels.append(f"{start_col+1}-{end_col}")


plt.figure(figsize=(26, 14))
sns.heatmap(
    matrix_rast,
    # cmap="plasma",
    cmap="coolwarm",  # Цветовая схема (можете попробовать другие, например "coolwarm", "plasma")
    cbar=True,       # Отображение цветовой шкалы
    linewidths=0,  # Убираем линии между ячейками
    linecolor='white',  # Цвет линий между ячейками
    xticklabels=10,  # Отображение меток через 10 ячеек для оси X
    yticklabels=10,  # Отображение меток через 10 ячеек для оси Y
    annot=False,     # Включение отображения значений внутри ячеек (можно True для отображения)
    # square=True      # Заставляет ячейки быть квадратными
)

xticks_pos = np.arange(0, n_groups, 5)
ax.set_xticks(xticks_pos)
ax.set_xticklabels([x_labels[i] for i in xticks_pos], rotation=45, ha='right', fontsize=8)

# Настройка заголовков и шрифтов
# plt.title("Карта ошибок по расстояниям между центрами контуров", fontsize=10, fontweight='bold')
# plt.xlabel("№ расстояния по мере удаления от центра", fontsize=12)
plt.title("Карта ошибок по расстояниям между центрами контуров (агрегированные данные)", 
          fontsize=14, fontweight='bold')
plt.xlabel(f"№ группы расстояний (каждая группа = {GROUP_SIZE} последовательных расстояний)", fontsize=12)
plt.ylabel("№ среза", fontsize=12)
plt.xticks(fontsize=9)  # Размер шрифта меток оси X
plt.yticks(fontsize=9)  # Размер шрифта меток оси Y
# Сохранение и отображение
plt.savefig(os.path.join(output_dir, "Карта_ошибок_по_расстояним_меж_центрами.png"), bbox_inches='tight')
plt.show()
plt.close()
# Сохранение матрицы в файл
# np.savetxt(f"{output_dir}/Матрица_ошибок_по_расстояниям_старая.csv", matrix_rast, delimiter=";", fmt="%.5f")



plt.figure(figsize=(14, 12))
sns.heatmap(
    error_matrix,
    cmap="coolwarm",  # Цветовая схема (можете попробовать другие, например "coolwarm", "plasma")
    cbar=True,       # Отображение цветовой шкалы
    linewidths=0.5,  # Убираем линии между ячейками
    linecolor='white',  # Цвет линий между ячейками
    xticklabels=10,  # Отображение меток через 10 ячеек для оси X
    yticklabels=10,  # Отображение меток через 10 ячеек для оси Y
    annot=False,     # Включение отображения значений внутри ячеек (можно True для отображения)
    # square=True      # Заставляет ячейки быть квадратными
)

# Настройка заголовков и шрифтов
plt.title("Карта ошибок по радиусу ", fontsize=10, fontweight='bold')
plt.xlabel("№ стержня", fontsize=12)
plt.ylabel("№ среза", fontsize=12)
plt.xticks(fontsize=9)  # Размер шрифта меток оси X
plt.yticks(fontsize=9)  # Размер шрифта меток оси Y
# Сохранение и отображение
plt.savefig(os.path.join(output_dir, "Карта_ошибок_по_радиусу.png"), bbox_inches='tight')
plt.show()
plt.close()
# Сохранение матрицы в файл
# np.savetxt(f"{output_dir}/Матрица_ошибок_по_радиусу_макс_старая.csv", error_matrix, delimiter=";", fmt="%.5f")


# Создание фигуры для координат
plt.figure(figsize=(14, 12))
sns.heatmap(
    error_matrix_xy,
    cmap="coolwarm",  # Цветовая схема (можете попробовать другие, например "coolwarm", "plasma")
    cbar=True,       # Отображение цветовой шкалы
    linewidths=0.5,  # Убираем линии между ячейками
    linecolor='white',  # Цвет линий между ячейками
    xticklabels=10,  # Отображение меток через 10 ячеек для оси X
    yticklabels=10,  # Отображение меток через 10 ячеек для оси Y
    annot=False,     # Включение отображения значений внутри ячеек (можно True для отображения)
    # square=True      # Заставляет ячейки быть квадратными
)

# Настройка заголовков
plt.title("Карта ошибок по координате ", fontsize=10, fontweight='bold')
plt.xlabel("№ стержня", fontsize=12)
plt.ylabel("№ среза", fontsize=12)
plt.xticks(fontsize=9)  # Размер шрифта меток оси X
plt.yticks(fontsize=9)  # Размер шрифта меток оси Y
# Сохранение и отображение
plt.savefig(os.path.join(output_dir, "Карта_ошибок_по_координате.png"), bbox_inches='tight')
plt.show()
plt.close()
# np.savetxt(f"{output_dir}/Матрица_ошибок_по_координате_старая.csv", error_matrix_xy, delimiter=";", fmt="%.5f")



# 
# Создание фигуры для СКО
plt.figure(figsize=(14, 12))
sns.heatmap(
    stdrad_matrix,
    cmap="coolwarm",  # Цветовая схема (можете попробовать другие, например "coolwarm", "plasma")
    cbar=True,       # Отображение цветовой шкалы
    linewidths=0.5,  # Убираем линии между ячейками
    linecolor='white',  # Цвет линий между ячейками
    xticklabels=10,  # Отображение меток через 10 ячеек для оси X
    yticklabels=10,  # Отображение меток через 10 ячеек для оси Y
    annot=False,     # Включение отображения значений внутри ячеек (можно True для отображения)
    # square=True      # Заставляет ячейки быть квадратными
)

# Настройка заголовков
plt.title("Карта СКО радиусов ", fontsize=10, fontweight='bold')
plt.xlabel("№ стержня", fontsize=12)
plt.ylabel("№ среза", fontsize=12)
plt.xticks(fontsize=9)  # Размер шрифта меток оси X
plt.yticks(fontsize=9)  # Размер шрифта меток оси Y
# Сохранение и отображение
plt.savefig(os.path.join(output_dir, "Карта СКО радиусов.png"), bbox_inches='tight')
plt.show()
plt.close()
# np.savetxt(f"{output_dir}/Матрица_СКО_радиусов.csv", stdrad_matrix, delimiter=";", fmt="%.5f")
# np.savetxt(f"{output_dir}/Матрица_радиусов.csv", rad_matrix, delimiter=";", fmt="%.5f")
#
# поиск


# plt.plot(range(1, len(dicom_files) + 1), full_errors, marker='o')  # X - от 1 до n, Y - значения error_n
# plt.title('Ошибка по радиусу от n среза')
# plt.xlabel('Номер среза, n')
# plt.ylabel('Средняя ошибка радиуса на срезе, мм')
# plt.grid()
# plt.savefig(os.path.join(output_dir, "Ошибка_по_радиусу_от_n_среза.png"))
# plt.close()
   
# plt.plot(range(1, len(dicom_files) + 1), full_errors_xy, marker='o')  # X - от 1 до n, Y - значения error_n
# plt.title('Ошибка по координате от n среза')
# plt.xlabel('Номер среза, n')
# plt.ylabel('Средняя ошибка координаты на срезе, мм')
# plt.grid()
# plt.savefig(os.path.join(output_dir, "Ошибка_координаты_от_n_среза.png"))
# plt.close()


plt.figure(figsize=(12, 6))

# Ошибка радиуса по срезам
plt.subplot(1, 2, 1)
x_points = range(1, len(dicom_files)+1)
plt.plot(x_points, full_errors, 'b-o', markersize=5, linewidth=1)

if len(x_points) > 2:
    x = np.array(x_points)
    y = np.array(full_errors)
    coefficients = np.polyfit(x, y, 3)
    polynomial = np.poly1d(coefficients)
    x_vals = np.linspace(min(x), max(x), 100)
    plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2)

plt.title('Средняя ошибка среза по радиусу от n среза ')
plt.xlabel('Номер среза, n')
plt.ylabel('Ошибка, мм')
plt.grid(True)

# Ошибка координат по срезам
plt.subplot(1, 2, 2)
plt.plot(x_points, full_errors_xy, 'b-o', markersize=5, linewidth=1)

if len(x_points) > 2:
    x = np.array(x_points)
    y = np.array(full_errors_xy)
    coefficients = np.polyfit(x, y, 3)
    polynomial = np.poly1d(coefficients)
    x_vals = np.linspace(min(x), max(x), 100)
    plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2)

plt.title('Средняя ошибка среза по координате от n среза ')
plt.xlabel('Номер среза, n')
plt.ylabel('Ошибка, мм')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Ошибки_по_срезам.png"))
plt.close()





print('-'*50)


# print(big_circles_cent)
# # Анализ корреляций
# # 1. Подготовка данных
# # Расстояния от центра для каждого стержня
# reference_distances = [np.linalg.norm(np.array(circle[0]) - np.array(the_one[0])) 
#                       for circle in all_circles[1:]]  # Исключаем the_one

# # Эталонные радиусы
# reference_radii = [circle[1] for circle in all_circles[1:]]

# # Средние ошибки по всем срезам для каждого стержня
# mean_radius_errors = np.mean(error_matrix, axis=0)

# # 2. Расчет корреляций
# corr_with_distance, _ = pearsonr(reference_distances, mean_radius_errors)
# corr_with_radius, _ = pearsonr(reference_radii, mean_radius_errors)

# print(f"\nКорреляция ошибки с расстоянием от центра: {corr_with_distance:.3f}")
# print(f"Корреляция ошибки с эталонным радиусом: {corr_with_radius:.3f}")

# # 3. Визуализация
# plt.figure(figsize=(12, 5))

# # График зависимости ошибки от расстояния
# plt.subplot(1, 2, 1)
# sns.regplot(x=reference_distances, y=mean_radius_errors, 
#             scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
# plt.title('Зависимость ошибки от расстояния до центра')
# plt.xlabel('Расстояние от центра, мм')
# plt.ylabel('Средняя ошибка радиуса, мм')
# plt.grid(True)

# # График зависимости ошибки от эталонного радиуса
# plt.subplot(1, 2, 2)
# sns.regplot(x=reference_radii, y=mean_radius_errors, 
#             scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
# plt.title('Зависимость ошибки от эталонного радиуса')
# plt.xlabel('Эталонный радиус, мм')
# plt.ylabel('Средняя ошибка радиуса, мм')
# plt.grid(True)

# plt.tight_layout()
# plt.savefig("C:/Users/k.zhukov/Downloads/phantom (2)/phantom_MRI/correlation_analysis.png")
# 
print('-'*50)
