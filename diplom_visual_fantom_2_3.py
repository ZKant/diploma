#импорт библиотек
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import svgwrite  # Для сохранения в SVG
import math
import random
import os
import seaborn as sns
from scipy import stats
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


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

    average_radius = np.mean(radii)

    return center, average_radius


#Функции для поворота и сдвига(поменять на итерованный сдвиг аксиального среза начиная с первого(от кожуха фантома до конца)
# def translate(x, y, center):
#     return x - center[0], y - center[1]

# def rotate(x, y, angle):
#     matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
#     product = np.dot(matrix, np.array([[x], [y]]))
#     return product.ravel()[0], product.ravel()[1]




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
# ==================================================================== #
# ==================================================================== #
# ===========вычисление векторной ошибки================ #
# ==================================================================== #
# ==================================================================== #
# Вычисление ошибок по радиусу и по координатам
# def calculate_errors(array1, array2):
#     """
#     Вычисляет ошибки для координат и радиусов между двумя массивами.
    
#     :param array1: Список элементов [(x, y), radius] — первый массив (эталонный).
#     :param array2: Список элементов [(x, y), radius] — второй массив (проверяемый).
#     :return: Два списка: ошибки для координат и ошибки для радиусов.
#     """
#     coordinate_errors = []
#     radius_errors = []
#     std_rad_list = []
#     radius_list = []

#     # def distance(point1, point2): #это расстояние не векторное 
#     #     # Вычисление Евклидова расстояния между двумя точками
#     #     return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

#     def distance(point1, point2, center=(127.5, 127.5)):
#         r1 = math.hypot(point1[0] - center[0], point1[1] - center[1])
#         r2 = math.hypot(point2[0] - center[0], point2[1] - center[1])
#         return r1 - r2


#     # def distance(point1, point2): #это расстояние векторное
#     # # Вычисляем расстояние каждой точки от центра (0, 0)
#     #     r1 = math.hypot(point1[0], point1[1])
#     #     r2 = math.hypot(point2[0], point2[1])
#     #     # Векторное расстояние: если point1 дальше — плюс, если ближе — минус
#     #     return r1 - r2
#     # Проверяем, что массивы одинаковой длины
#     if len(array1) != len(array2):
#         raise ValueError("Массивы должны быть одинаковой длины")

#     for item1, item2 in zip(array1, array2):
#         coord1, radius1 = item1
#         coord2, radius2, std_rad = item2

#         # Вычисление ошибки координат (расстояние между точками)
#         coord_error = distance(coord1, coord2)
#         coordinate_errors.append(coord_error)


#         # Вычисление ошибки радиуса (модуль разности радиусов)
#         # radius_error = abs(radius2 - radius1)
#         # вычисление ошибки с учетом направления 
#         radius_error = (radius2 - radius1)
#         radius_errors.append(radius_error)
#         std_rad_list.append(std_rad)
#         radius_list.append(radius2)


#     return coordinate_errors, radius_errors, std_rad_list, radius_list

# ==================================================================== #
# ==================================================================== #
# ===========вычисление просто ошибки================ #
# ==================================================================== #
# ==================================================================== #
# def calculate_errors(array1, array2):
#     """
#     Вычисляет ошибки для координат и радиусов между двумя массивами.
    
#     :param array1: Список элементов [(x, y), radius] — первый массив (эталонный).
#     :param array2: Список элементов [(x, y), radius] — второй массив (проверяемый).
#     :return: Два списка: ошибки для координат и ошибки для радиусов.
#     """
#     coordinate_errors = []
#     radius_errors = []
#     std_rad_list = []
#     radius_list = []

#     def distance(point1, point2):
#         # Вычисление Евклидова расстояния между двумя точками
#         return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

#     # Проверяем, что массивы одинаковой длины
#     if len(array1) != len(array2):
#         raise ValueError("Массивы должны быть одинаковой длины")

#     for item1, item2 in zip(array1, array2):
#         coord1, radius1 = item1
#         coord2, radius2, std_rad = item2

#         # Вычисление ошибки координат (расстояние между точками)
#         coord_error = distance(coord1, coord2)
#         coordinate_errors.append(coord_error)


#         # Вычисление ошибки радиуса (модуль разности радиусов)
#         radius_error = abs(radius1 - radius2)
#         radius_errors.append(radius_error)
#         std_rad_list.append(std_rad)
#         radius_list.append(radius2)


#     return coordinate_errors, radius_errors, std_rad_list, radius_list
# ===========конец================ #
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
# ==================================================================== #

#Читаем dicom файлы
dicom_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_2/Фантом_2_снимки'
# dicom_dir = '/home/kirill_zh/folder_py/test/'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_2/Фантом_2_результаты/Средний_радиус_вект_ошибка'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_2/Фантом_2_результаты/Минимальный_радиус_вект_ошибка'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_2/Фантом_2_результаты/Максимальный_радиус_вект_ошибка'
output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_2/Фантом_2_результаты/Средний_радиус_вект_ошибка'
# output_dir = 'C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Максимальный_радиус_старая_ошибка'

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

# Читаем каждый файл и сохраняем картинки оконтуривания для каждого среза в папку
for n, dicom_file in enumerate(dicom_files, start=1):
    dicom_file_path = os.path.join(dicom_dir, dicom_file)
    # output_dir = '/home/kirill_zh/folder_py/MRT_slices/'
    dataset = pydicom.dcmread(dicom_file_path)
    # Извлечение пиксельных данных
    pixel_array = dataset.pixel_array
    
    spacing = 0.9375 #кэф из метадаты в мм/пиксел

    image = 255 * (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
    image = image.astype(np.uint8)

    # # Нормализация пиксельных значений к диапазону 0-255
    # # image = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
    # # image = np.uint8(image)

    # Применение сглаживания для снижения шума
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

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

    # Сохранение графиков
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title(f"Исходное изображение(Срез №{n})")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(output_image)
    # plt.title(f"Выделенные контуры(Срез №{n})")
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, f"Выделенные_контуры_{n}.png"))
    # plt.close()


# Задаем массивы центр+радиус, радиус, центр
    centers_and_radii = []
    radiuses = []
    centers = []

    i = 0 #счетчик контуров
    lya = np.array([127.5, 127.5])


# ==================================================================== #
# ==================================================================== #
# ===========подбор способы поиска параметров контура================ #
# ==================================================================== #
# ==================================================================== #


# Для каждого выделенного контура находим параметры - радиус и координаты
# Есть три способа, тестируем все три
#Первый способ: с помощью функции `calculate_average_center_and_radius()`
    # for cnt in contours:
    #     center, radius = calculate_average_center_and_radius(cnt)
    #     if center is not None and radius is not None:
    #         # Приводим координаты и радиус к масштабу
    #         x = center[0] * spacing
    #         y = center[1] * spacing
    #         radius = radius * spacing
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
    #         centers_and_radii.append((center, radius))
    #         radius1 = int(round(radius))  # Приводим радиус к int
    #         centers.append(center)
    #         i += 1
    #         cv2.circle(output_image, buf, radius1, (0, 0, 255), 1)     
    
    
    
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
    
    #     buf = (int(center[0]), int(center[1]))
    #     # cv2.circle(output_image, buf, avg_radius, (0, 0, 255), 1)

    # width, height = image.shape[1], image.shape[0]
    # dwg = svgwrite.Drawing(f"/home/kirill_zh/folder_py/Results/contours_with_circles{n}.svg", size=(width, height))

# Второй способ: через cv функцию для поиска минимальной окружности, 
# которая охватывает контур и +функция фильтрации по условиям

    # for cnt in contours:
    #     # Находим минимальную окружность, которая охватывает контур
    #     (x, y), radius = cv2.minEnclosingCircle(cnt)

    #     # Приводим координаты к масштабу
    #     x = x * spacing
    #     y = y * spacing
    #     radius = radius * spacing

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
    #         centers_and_radii.append((center, radius))
    #         radius1 = int(round(radius))  # Приводим радиус к int
    #         centers.append(center)
    #         i += 1
    #         cv2.circle(output_image, buf, radius1, (0, 0, 255), 1)    
# ///////////////////////////////////////////////////
    #     # Фильтрация по условиям
    #     if is_point_in_circle(x, y, 127.5 * spacing, 127.5 * spacing, 85.8) and 1 < radius < 200:
    #         radiuses.append(radius)
    #         center = (x, y)
    #         # print(center)
    #         buf = int(round(x)), int(round(y))  # Округляем центр
    #         centers_and_radii.append(((x, y), radius))
    #         radius1 = int(round(radius))  # Приводим радиус к int
    #         centers.append(center)
    #         print(centers)
    #         # Рисуем окружность на выходном изображении
    #         cv2.circle(output_image, buf, radius1, (0, 0, 255), 1)

# Третий способ: Через взвешивание контура и поиск центра масс контура
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
    
        # Вычисляем радиус как расстояние до самой удаленной точки на контуре
        # radius = max(np.linalg.norm(np.array((x, y)) - np.array(pt[0])) for pt in approx_contour)
        distances = []
        for point in cnt:
            distance = np.sqrt((point[0][0] - x) ** 2 + (point[0][1] - y) ** 2)
            distances.append(distance)
    
        radius = np.mean(distances)*spacing
        rad_std = np.std(distances)*spacing
        x = x*spacing
        y = y*spacing

        # x, y = rotate(x, y, n*(0.0005)*math.pi/180)
        
        # Фильтрация по условиям
        if is_point_in_circle(x, y, 127.5*spacing, 127.5*spacing, (87)) == True and radius < 200 and radius > 0.3:
            radiuses.append(radius)
            center = (x, y)
            # print(radius)
            # print(center)
            buf = int(round(x/spacing)), int(round(y/spacing))  # Округляем центр
            # buf = x, y
            # print(buf)
            centers_and_radii.append((center, radius, rad_std))
            radius1 = int(round(radius/spacing))  # Приводим радиус к int
            centers.append(center)
            i += 1
            cv2.circle(output_image, buf, radius1, (0, 0, 255), 1)    
# .....................................................................
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

# Теперь выстраиваем эталонны контур согласно размерам с чертежа
    x, y = (127.5*spacing), (127.5*spacing)
    # x, y = (127.726*spacing), (127.531*spacing)
    big_circles_cent.append((largest_circle_center_x, largest_circle_center_y))
    # print('-'*50)
    # print(big_circles_cent)
    # print('-'*50)
    # x, y = largest_circle_center_x, largest_circle_center_y #включить сюда центры окружности которая нашлась выше
    # print(x, y)

#     small_step = (20) #mm
#     medium_step1, medium_step2 = (25), (10) #mm
#     large_step = (50) #mm

    

#     shift_x = 0 #2.7
#     shift_y = 0


#     x = x + shift_x
#     y = y + shift_y

#     the_one = [(x , y), 85.8] #Большая окружность(кожух фантома) диаметром 171.6мм
# # Угол поворота в градусах (например, 30)
#     rotation_angle_deg = 0
#     rotation_angle = math.radians(rotation_angle_deg)

    small_step = 20  # mm
    medium_step1, medium_step2 = 25, 10  # mm
    large_step = 50  # mm

    shift_x = 2.81 # 2.7
    shift_y = 0

    x = x + shift_x
    y = y + shift_y

    the_one = [(x, y), 85.8]  # Центр остается (x, y)

    # Угол поворота и центр вращения
    rotation_angle_deg = 0
    rotation_angle = math.radians(rotation_angle_deg)
    cx = x + 0  # Пример: центр вращения смещён на 30 мм по X от (x, y)
    cy = y + 0  # Пример: центр вращения смещён на 30 мм по Y от (x, y)

    # Относительные смещения для маленьких кругов
    small_circles_offsets = [
        [(small_step, small_step), 1.5],
        [(small_step, 0), 1.5],
        [(0, small_step), 1.5],
        [(-small_step, -small_step), 1.5],
        [(0, -small_step), 1.5],
        [(-small_step, 0), 1.5],
        [(small_step, -small_step), 1.5],
        [(-small_step, small_step), 1.5],
        [(2*small_step, 2*small_step), 1.5],
        [(2*small_step, 0), 1.5],
        [(0, 2*small_step), 1.5],
        [(-2*small_step, -2*small_step), 1.5],
        [(0, -2*small_step), 1.5],
        [(-2*small_step, 0), 1.5],
        [(2*small_step, -2*small_step), 1.5],
        [(-2*small_step, 2*small_step), 1.5],
        [(small_step, 2*small_step), 1.5],
        [(2*small_step, small_step), 1.5],
        [(-small_step, -2*small_step), 1.5],
        [(-2*small_step, -small_step), 1.5],
        [(small_step, -2*small_step), 1.5],
        [(2*small_step, -small_step), 1.5],
        [(-small_step, 2*small_step), 1.5],
        [(-2*small_step, small_step), 1.5],
        [(-2*small_step, -3*small_step), 1.5],
        [(-small_step, -3*small_step), 1.5],
        [(0, -3*small_step), 1.5],
        [(small_step, -3*small_step), 1.5],
        [(2*small_step, -3*small_step), 1.5],
        [(-2*small_step, 3*small_step), 1.5],
        [(-small_step, 3*small_step), 1.5],
        [(0, 3*small_step), 1.5],
        [(small_step, 3*small_step), 1.5],
        [(2*small_step, 3*small_step), 1.5],
        [(3*small_step, -2*small_step), 1.5],
        [(3*small_step, -small_step), 1.5],
        [(3*small_step, 0), 1.5],
        [(3*small_step, small_step), 1.5],
        [(3*small_step, 2*small_step), 1.5],
        [(-3*small_step, -2*small_step), 1.5],
        [(-3*small_step, -small_step), 1.5],
        [(-3*small_step, 0), 1.5],
        [(-3*small_step, small_step), 1.5],
        [(-3*small_step, 2*small_step), 1.5]
    ]

# Применяем поворот к смещениям и формируем small_circles
    # small_circles = []
    # for offset, radius in small_circles_offsets:
    #     dx, dy = offset
    #     # Вычисляем повернутые смещения
    #     rotated_dx = dx * math.cos(rotation_angle) - dy * math.sin(rotation_angle)
    #     rotated_dy = dx * math.sin(rotation_angle) + dy * math.cos(rotation_angle)
    #     # Абсолютные координаты после поворота
    #     circle_x = x + rotated_dx
    #     circle_y = y + rotated_dy
    #     small_circles.append([(circle_x, circle_y), radius])

    # # Все окружности с чертежа
    # all_circles = [the_one] + small_circles
    small_circles = []
    for offset, radius in small_circles_offsets:
        dx, dy = offset
        # Глобальные координаты точки до поворота
        original_x = x + dx
        original_y = y + dy
        
        # Смещение относительно центра вращения (cx, cy)
        rel_x = original_x - cx
        rel_y = original_y - cy
        
        # Применяем поворот
        rotated_rel_x = rel_x * math.cos(rotation_angle) - rel_y * math.sin(rotation_angle)
        rotated_rel_y = rel_x * math.sin(rotation_angle) + rel_y * math.cos(rotation_angle)
        
        # Новые координаты после поворота вокруг (cx, cy)
        circle_x = cx + rotated_rel_x
        circle_y = cy + rotated_rel_y
        
        small_circles.append([(circle_x, circle_y), radius])

    all_circles = [the_one] + small_circles



#     x = x + shift_x
#     y = y + shift_y

#     small_step = (20) #mm
#     medium_step1, medium_step2 = (25), (10) #mm
#     large_step = (50) #mm
    
#     the_one = [(x , y), 85.8] #Большая окружность(кожух фантома) диаметром 171.6мм

# # Размечаем маленькие окружности с чертежа диаметром 3мм
#     small_circles = [
#         [(x + small_step, y + small_step), 1.5],
#         [(x + small_step, y), 1.5],
#         [(x, y + small_step), 1.5],
#         [(x - small_step, y - small_step), 1.5],
#         [(x, y - small_step), 1.5],
#         [(x - small_step, y), 1.5],
#         [(x + small_step, y - small_step), 1.5],
#         [(x - small_step, y + small_step), 1.5],
#         [(x + 2*small_step, y + 2*small_step), 1.5],
#         [(x + 2*small_step, y), 1.5],
#         [(x, y + 2*small_step), 1.5],
#         [(x - 2*small_step, y - 2*small_step), 1.5],
#         [(x, y - 2*small_step), 1.5],
#         [(x - 2*small_step, y), 1.5],
#         [(x + 2*small_step, y - 2*small_step), 1.5],
#         [(x - 2*small_step, y + 2*small_step), 1.5],
#         [(x + small_step, y + 2*small_step), 1.5],
#         [(x + 2*small_step, y + small_step), 1.5],
#         [(x - small_step, y - 2*small_step), 1.5],
#         [(x - 2*small_step, y - small_step), 1.5],
#         [(x + small_step, y - 2*small_step), 1.5],
#         [(x + 2*small_step, y - small_step), 1.5],
#         [(x - small_step, y + 2*small_step), 1.5],
#         [(x - 2*small_step, y + small_step), 1.5],
#         [(x - 2*small_step, y - 3*small_step), 1.5],
#         [(x - small_step, y - 3*small_step), 1.5],
#         [(x, y - 3*small_step), 1.5],
#         [(x + small_step, y - 3*small_step), 1.5],
#         [(x + 2*small_step, y - 3*small_step), 1.5],
#         [(x - 2*small_step, y + 3*small_step), 1.5],
#         [(x - small_step, y + 3*small_step), 1.5],
#         [(x, y + 3*small_step), 1.5],
#         [(x + small_step, y + 3*small_step), 1.5],
#         [(x + 2*small_step, y + 3*small_step), 1.5],
#         [(x + 3*small_step, y - 2*small_step), 1.5],
#         [(x + 3*small_step, y - small_step), 1.5],
#         [(x + 3*small_step, y), 1.5],
#         [(x + 3*small_step, y + small_step), 1.5],
#         [(x + 3*small_step, y + 2*small_step), 1.5],
#         [(x - 3*small_step, y - 2*small_step), 1.5],
#         [(x - 3*small_step, y - small_step), 1.5],
#         [(x - 3*small_step, y), 1.5],
#         [(x - 3*small_step, y + small_step), 1.5],
#         [(x - 3*small_step, y + 2*small_step), 1.5]
#     ]
# # Все окружности с чертежа
#     all_circles = [the_one] + small_circles 


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

    plt.figure(figsize=(6, 6))  # Увеличиваем ширину для трёх изображений
    # Исходное изображение
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f"фант3_снимки{n}.png"))
    plt.close()

    # Сохранение графиков
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
# Функция для поиска расстояния между точками
    def distance(point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    # centers[0] = (127.5*spacing, 127.5*spacing)
# Сортировка и расчет ошибок для радиуса и координаты    
    acc_sorted_points = sort_by_distance_with_radius(all_circles, the_one[0]) #((127.5*spacing), (127.5*spacing))

    nonacc_sorted_points = sort_by_min_distance_and_radius(acc_sorted_points, centers_and_radii)
    
    # print(acc_sorted_points, nonacc_sorted_points)

    # print(f"sorted1 = {acc_sorted_points}, sorted2 = {nonacc_sorted_points}")
        
  
    xy_errors, radius_errors, std_rad, radis, radials_err, tang_err = [], [], [], [], [], []
    xy_errors, radius_errors, std_rad, radis, radials_err, tang_err = calculate_errors(acc_sorted_points, nonacc_sorted_points)

    # print(radius_errors)

    plt.figure(figsize=(14, 14), dpi=150)
    ax = plt.gca()
    ax.set_aspect('equal')
    
    # Получаем эталонные позиции и радиусы
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    import matplotlib as mpl

    reference_positions = [item[0] for item in acc_sorted_points]
    reference_radii = [item[1] for item in acc_sorted_points]

    # Нормализация ошибок для цветовой карты
    norm = plt.Normalize(vmin=min(radius_errors), vmax=max(radius_errors))
    cmap = mpl.cm.YlOrRd  # Используем "желто-оранжево-красную" палитру

    max_radius = max(reference_radii)

    for (x, y), r_ref, err in zip(reference_positions, reference_radii, radius_errors):
        if r_ref == max_radius:
            current_alpha = 0.15   # Фон! Очень нежный уровень прозрачности
        else:
            current_alpha = 1.0    # Основные круги полностью видимы
        if r_ref == 5:
            spc = 1.2
        elif r_ref == 85.8:
            spc = 1.2
        elif r_ref == 2.95:
            spc = 1.5
        else:
            spc = 4
        

        circle = Circle(
            (x, y), 
            r_ref*spc, 
            fill=True, 
            color=cmap(norm(err)),
            alpha=current_alpha,
            linewidth=0.0           # Убираем обводку
        )
        ax.add_patch(circle)
        plt.text(x, y, f'{err:.2f}', ha='center', va='center', fontsize=12, color='#555555', fontweight='bold')

    # Настройка границ и цветовой шкалы
    ax.set_xlim(0, 255 * spacing)
    ax.set_ylim(0, 255 * spacing)
    plt.title(f'Тепловая карта ошибок радиуса (Срез №{n})')

    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
        ax=ax,
        fraction=0.046, pad=0.04
    )
    cb.set_label('Величина отклонений')

    ax.set_facecolor('white')
    plt.gca().invert_yaxis()   # Для соответствия ориентации DICOM
    plt.tight_layout()

    # Убираем рамки (оси)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    
    # Сохранение
    plt.savefig(os.path.join(output_dir, f"Тепловая_карта_срез_№{n}.png"), dpi=150)
   
    plt.close()

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
    # print(dicom_file_path)

# Cохранение графиков ошибки для каждого срез
    # Для графиков ошибок на каждом срезе
    plt.figure(figsize=(12, 6))

    # График ошибок радиуса
    # plt.subplot(1, 2, 1)
    plt.plot(y_axis, radius_errors, 'b-o', markersize=5, linewidth=1, label='Ошибки')

    # Аппроксимация
    if len(y_axis) > 2:  # Минимум 3 точки для полинома 2-й степени
        x = np.array(range(1, len(radius_errors)+1))
        y = np.array(radius_errors)
        coefficients = np.polyfit(x, y, 3)
        polynomial = np.poly1d(coefficients)
        x_vals = np.linspace(min(x), max(x), 100)
        plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2, label='Тренд')

    plt.title(f"Ошибка радиуса (Срез №{n}, Все контуры)")
    plt.ylabel('мм')
    plt.legend()
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

    plt.title(f"Ошибка координат (Срез №{n}, Все контуры)")
    plt.legend()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Ошибки_на_срезе_график№{n}_2ф.png"))
    plt.close()


    # radis = radis[]

    matrix.append(radius_errors)
    matrix_xy.append(xy_errors)
    matrix_std.append(std_rad)
    matrix_rad.append(radis)
    

#   centers0.append(centers[0])

# print(centers0)

error_matrix = np.array(matrix)
print("Матрица ошибок по радиусу")
# print(error_matrix)

error_matrix_xy = np.array(matrix_xy)
print("Матрица ошибок по координате")
# print(error_matrix_xy)

stdrad_matrix = np.array(matrix_std)
print("Матрица СКО")
# print(stdrad_matrix)

rad_matrix = np.array(matrix_rad)
print("Матрица радиусов")
# print(rad_matrix)

average_values = np.mean(error_matrix, axis=0)
average_values_xy = np.mean(error_matrix_xy, axis=0)

# Номера столбцов
column_indices = np.arange(1, error_matrix.shape[1] + 1)
column_indices_xy = np.arange(1, error_matrix_xy.shape[1] + 1)

# Построение графика
plt.figure(figsize=(6, 6))

# Средняя ошибка радиуса
# plt.subplot(1, 2, 1)
plt.plot(column_indices, average_values, 'b-o', markersize=5, linewidth=1)

if len(column_indices) > 2:
    x = np.array(column_indices)
    y = np.array(average_values)
    coefficients = np.polyfit(x, y, 3)
    polynomial = np.poly1d(coefficients)
    x_vals = np.linspace(min(x), max(x), 100)
    plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2)

plt.title('Средняя ошибка радиуса по стрежню (Все контуры)')
plt.xlabel('Номер стержня')
plt.ylabel('Ошибка, мм')
plt.grid(True)

# Средняя ошибка координат
# plt.subplot(1, 2, 2)
# plt.plot(column_indices_xy, average_values_xy, 'b-o', markersize=5, linewidth=1)

# if len(column_indices_xy) > 2:
#     x = np.array(column_indices_xy)
#     y = np.array(average_values_xy)
#     coefficients = np.polyfit(x, y, 3)
#     polynomial = np.poly1d(coefficients)
#     x_vals = np.linspace(min(x), max(x), 100)
#     plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2)


# plt.title('Средняя ошибка координат по стрежню (Все контуры)')
# plt.xlabel('Номер стержня')
# plt.ylabel('Ошибка, мм')
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Средние_ошибки_по_стержню_3максф.png"))
plt.close()


from scipy.ndimage import zoom
smooth_error_matrix = zoom(error_matrix, zoom=4)  # Увеличиваем разрешение
smooth_error_matrix_xy = zoom(error_matrix_xy, zoom=4)



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
plt.title("Карта ошибок по радиусу", fontsize=10, fontweight='bold')
plt.xlabel("№ стержня", fontsize=12)
plt.ylabel("№ среза", fontsize=12)
plt.xticks(fontsize=9)  # Размер шрифта меток оси X
plt.yticks(fontsize=9)  # Размер шрифта меток оси Y
# Сохранение и отображение
plt.savefig(os.path.join(output_dir, "Карта_ошибок_по_радиусу_2ф.png"), bbox_inches='tight')
plt.show()
plt.close()
# Сохранение матрицы в файл
np.savetxt(f"{output_dir}/Матрица_ошибок_по_радиусу.csv", error_matrix, delimiter=";", fmt="%.5f")


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
plt.title("Карта ошибок по координате", fontsize=10, fontweight='bold')
plt.xlabel("№ стержня", fontsize=12)
plt.ylabel("№ среза", fontsize=12)
plt.xticks(fontsize=9)  # Размер шрифта меток оси X
plt.yticks(fontsize=9)  # Размер шрифта меток оси Y
# Сохранение и отображение
plt.savefig(os.path.join(output_dir, "Карта_ошибок_по_координате_2ф.png"), bbox_inches='tight')
plt.show()
plt.close()
np.savetxt(f"{output_dir}/Матрица_ошибок_по_координате.csv", error_matrix_xy, delimiter=";", fmt="%.5f")



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
plt.title("Карта СКО радиусов", fontsize=10, fontweight='bold')
plt.xlabel("№ стержня", fontsize=12)
plt.ylabel("№ среза", fontsize=12)
plt.xticks(fontsize=9)  # Размер шрифта меток оси X
plt.yticks(fontsize=9)  # Размер шрифта меток оси Y
# Сохранение и отображение
plt.savefig(os.path.join(output_dir, "Карта СКО радиусов_2ф.png"), bbox_inches='tight')
plt.show()
plt.close()
np.savetxt(f"{output_dir}/Матрица_СКО_радиусов.csv", stdrad_matrix, delimiter=";", fmt="%.5f")
np.savetxt(f"{output_dir}/Матрица_радиусов.csv", rad_matrix, delimiter=";", fmt="%.5f")
#
# поиск


plt.figure(figsize=(6, 6))

# Ошибка радиуса по срезам
# plt.subplot(1, 2, 1)
x_points = range(1, len(dicom_files)+1)
plt.plot(x_points, full_errors, 'b-o', markersize=5, linewidth=1)

if len(x_points) > 2:
    x = np.array(x_points)
    y = np.array(full_errors)
    coefficients = np.polyfit(x, y, 3)
    polynomial = np.poly1d(coefficients)
    x_vals = np.linspace(min(x), max(x), 100)
    plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2)

plt.title('Ошибка по радиусу от n среза (Все контуры)')
plt.xlabel('Номер среза, n')
plt.ylabel('Средняя ошибка радиуса на срезе, мм')
plt.grid(True)

# # Ошибка координат по срезам
# plt.subplot(1, 2, 2)
# plt.plot(x_points, full_errors_xy, 'b-o', markersize=5, linewidth=1)

# if len(x_points) > 2:
#     x = np.array(x_points)
#     y = np.array(full_errors_xy)
#     coefficients = np.polyfit(x, y, 3)
#     polynomial = np.poly1d(coefficients)
#     x_vals = np.linspace(min(x), max(x), 100)
#     plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=2)

# plt.title('Ошибка по координате от n среза (Все контуры)')
# plt.xlabel('Номер среза, n')
# plt.ylabel('Средняя ошибка координаты на срезе, мм')
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Ошибки_по_срезам_3максф.png"))
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
