import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Замените 'имя_файла.csv' на путь к вашему файлу
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/diploma/Матрица_ошибок_по_радиусу.csv', delimiter=';')
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Весь_фантом/Максимальный_радиус_вект_ошибка/Матрица_ошибок_по_радиусу.csv', delimiter=';')
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/diploma/Матрица_ошибок_по_координате_3.csv', delimiter=';')
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/diploma/Матрица_ошибок_по_расстояниям_new.csv', delimiter=';')
error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/Диплом/Фантом_3/Фантом_3_результаты/Матрица ошибок по расстояниям_средний_новое.csv', delimiter=';')
print(error_matrix.shape)

group_indices_list = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    [13, 14, 15, 16, 17, 18],
    [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78],
    [79, 80, 81, 82, 83, 84],
    [85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96],
    [97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
    [109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132],
    [133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156],
    [157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168],
    [169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192],
    [193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216],
    [217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228],
    [229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252],
    [253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276],
    [277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300],
    [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324],
    [325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348],
    [349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372],
    [373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396],
    [397, 398, 399, 400, 401, 402],
    [403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426],
    [427, 428, 429, 430, 431, 432, 433, 434],
    [435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446],
    [447, 448, 449, 450, 451, 452, 453, 454],
    [455, 456, 457, 458],
    [459, 460, 461, 462],
    [463, 464, 465, 466, 467, 468, 469, 470],
    [471, 472, 473, 474, 475, 476, 477, 478],
    [479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502],
    [503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514],
    [515, 516, 517, 518],
    [519, 520, 521, 522, 523, 524, 525, 526],
    [527, 528, 529, 530],
    [531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554],
    [555, 556, 557, 558, 559, 560, 561, 562],
    [563, 564, 565, 566, 567, 568, 569, 570],
    [571, 572, 573, 574, 575, 576, 577, 578],
    [579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590],
    [591, 592, 593, 594, 595, 596, 597, 598],
    [599, 600, 601, 602, 603, 604, 605, 606],
    [607, 608, 609, 610, 611, 612, 613, 614],
    [615, 616, 617, 618],
    [619, 620, 621, 622, 623, 624, 625, 626],
    [627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650],
    [651, 652, 653, 654, 655, 656, 657, 658],
    [659, 660, 661, 662],
    [663, 664, 665, 666, 667, 668, 669, 670],
    [671, 672, 673, 674],
    [675, 676, 677, 678, 679, 680, 681, 682],
    [683, 684, 685, 686, 687, 688, 689, 690],
    [691, 692, 693, 694, 695, 696, 697, 698],
    [699, 700, 701, 702, 703, 704, 705, 706],
    [707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722],
    [723, 724, 725, 726, 727, 728, 729, 730],
    [731, 732, 733, 734, 735, 736, 737, 738],
    [739, 740, 741, 742, 743, 744, 745, 746],
    [747, 748, 749, 750, 751, 752, 753, 754],
    [755, 756, 757, 758, 759, 760, 761, 762],
    [763, 764, 765, 766, 767, 768, 769, 770],
    [771, 772, 773, 774, 775, 776, 777, 778],
    [779, 780, 781, 782, 783, 784, 785, 786],
    [787, 788, 789, 790, 791, 792, 793, 794],
    [795, 796, 797, 798, 799, 800, 801, 802],
    [803, 804, 805, 806],
    [807, 808, 809, 810, 811, 812, 813, 814],
    [815, 816],
    [817, 818, 819, 820, 821, 822, 823, 824],
    [825, 826, 827, 828, 829, 830, 831, 832],
    [833, 834, 835, 836, 837, 838, 839, 840],
    [841, 842, 843, 844, 845, 846, 847, 848],
    [849, 850, 851, 852],
    [853, 854],
    [855, 856, 857, 858, 859, 860, 861, 862],
    [863, 864, 865, 866, 867, 868, 869, 870],
    [871, 872, 873, 874, 875, 876, 877, 878],
    [879, 880, 881, 882],
    [883, 884, 885, 886, 887, 888, 889, 890],
    [891, 892, 893, 894],
    [895, 896, 897, 898, 899, 900, 901, 902],
    [903, 904, 905, 906, 907, 908, 909, 910],
    [911, 912, 913, 914, 915, 916, 917, 918],
    [919, 920, 921, 922, 923, 924, 925, 926],
    [927, 928, 929, 930],
    [931, 932, 933, 934, 935, 936, 937, 938],
    [939, 940, 941, 942],
    [943, 944, 945, 946]
]

group_labels = [
    "68.3 мм", "80.0 мм", "93.0 мм", "96.6 мм", "104.7 мм", "113.1 мм", "117.7 мм", "120.0 мм", 
    "121.3 мм", "128.0 мм", "129.4 мм", "131.5 мм", "133.0 мм", "136.6 мм", "143.2 мм", "145.1 мм", 
    "148.1 мм", "148.7 мм", "152.7 мм", "155.4 мм", "156.8 мм", "160.0 мм", "161.3 мм", "165.7 мм", 
    "166.5 мм", "167.2 мм", "169.4 мм", "169.7 мм", "170.7 мм", "171.5 мм", "172.1 мм", "172.5 мм", 
    "174.3 мм", "177.0 мм", "178.9 мм", "180.1 мм", "181.0 мм", "181.6 мм", "182.9 мм", "183.1 мм", 
    "183.8 мм", "186.0 мм", "188.1 мм", "190.4 мм", "193.1 мм", "194.2 мм", "195.4 мм", "196.8 мм", 
    "200.0 мм", "200.4 мм", "201.3 мм", "204.9 мм", "205.2 мм", "206.3 мм", "206.7 мм", "208.0 мм", 
    "211.1 мм", "212.1 мм", "215.7 мм", "215.9 мм", "217.8 мм", "218.8 мм", "219.8 мм", "221.8 мм", 
    "223.2 мм", "224.2 мм", "224.3 мм", "226.3 мм", "228.7 мм", "230.0 мм", "233.5 мм", "236.4 мм", 
    "239.6 мм", "240.0 мм", "243.1 мм", "244.9 мм", "246.2 мм", "246.5 мм", "248.7 мм", "253.0 мм", 
    "256.7 мм", "257.0 мм", "258.6 мм", "263.4 мм", "264.2 мм", "269.5 мм", "285.6 мм", "288.4 мм"
]

n_slices = error_matrix.shape[0]

zero_column = np.zeros((error_matrix.shape[0], 1))

# Объединяем по горизонтали
error_matrix = np.concatenate((zero_column, error_matrix), axis=1)

# Готовим финальную матрицу: 108 строк, 13 столбцов (1 + 12 групп)
result_matrix = np.zeros((n_slices, 89))
result_matrix[:, 0] = error_matrix[:, 0]  # первая колонка — большая окружность

# Усреднение по группам
for i, col_idxs in enumerate(group_indices_list):
    # result_matrix[:, i+1] = np.mean(error_matrix[:,col_idxs], axis=1)
    result_matrix[:, i+1] = np.mean(error_matrix[:,col_idxs], axis=1)

# Собираем подписи для оси X: первая — "Больш.окр.", дальше — группы
xticklabels = ["Больш.окр."] + group_labels

plt.figure(figsize=(19, 18))
sns.heatmap(
    result_matrix[:, 1:],
    cmap="viridis",
    cbar=True,
    linewidths=0.5,
    linecolor='white',
    xticklabels=xticklabels[1:],
    yticklabels=10,
    annot=False,
)

plt.title("Тепловая карта ошибок по группам (одинаковые периметры), мм", fontsize=16, fontweight='bold')
plt.xlabel("Группа (периметры)", fontsize=14, fontweight='bold')
plt.ylabel("№ среза", fontsize=14, fontweight='bold')
plt.xticks(fontsize=10, rotation=40, ha='right', fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("C:/Users/k.zhukov/Desktop/diploma/heatmap_grouped_errors_2.png", bbox_inches='tight')
plt.show()
plt.close()




# Вычисляем средние значения по срезам для каждой группы
mean_per_group = np.mean(result_matrix[:, 1:], axis=0)

overall_mean = np.mean(mean_per_group)
overall_median = np.median(mean_per_group)

# Создаем фигуру
plt.figure(figsize=(20, 10))

# Построение столбчатой диаграммы
bars = plt.bar(group_labels, mean_per_group, color='skyblue', label='Погрешность', alpha=0.7)

# Добавляем линию, соединяющую вершины столбцов
x_positions = np.arange(len(group_labels))  # Числовые позиции групп
line = plt.plot(x_positions, mean_per_group, 
                'o-', color='red', linewidth=2, markersize=8,
                label='')

# Добавляем горизонтальные линии для среднего и медианы
plt.axhline(y=overall_mean, color='green', linestyle='--', linewidth=3, 
            label=f'Среднее по всем группам: {overall_mean:.2f} мм')
plt.axhline(y=overall_median, color='purple', linestyle='--', linewidth=3, 
            label=f'Медиана по всем группам: {overall_median:.2f} мм')

# Настройка внешнего вида
plt.title('Средняя погрешность расстояний по группам периметров между центрами контуров и центром кожуха', 
          fontsize=18, fontweight='bold')
plt.xlabel('Группы пар контуров с совпадающим периметром', fontsize=16, fontweight='bold')
plt.ylabel('Средняя погрешность расстояний, мм', fontsize=16, fontweight='bold')

# Жирные подписи на осях
plt.xticks(rotation=90, fontsize=10, ha='center', fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Добавление сетки
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Добавление значений на точки линии
for i, value in enumerate(mean_per_group):
    # Подписи красным жирным шрифтом над точками
    plt.text(x_positions[i], value + 0.02 * max(mean_per_group), 
             f'{value:.2f}', 
             ha='center', va='bottom', 
             fontsize=10, fontweight='bold', color='red',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

# Добавляем легенду
plt.legend(fontsize=14, loc='best', frameon=True, shadow=True)

# Устанавливаем пределы оси Y для лучшего отображения
plt.ylim(min(mean_per_group) * 1.15, max(mean_per_group) * 1.15)

plt.text(len(group_labels)*0.98, overall_mean*1.01, 
         f'', 
         fontsize=12, color='green', fontweight='bold', ha='right')
plt.text(len(group_labels)*0.98, overall_median*0.99, 
         f'', 
         fontsize=12, color='purple', fontweight='bold', ha='right')

plt.tight_layout()
plt.savefig("C:/Users/k.zhukov/Desktop/diploma/group_means_with_trend.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# mean_per_group = np.mean(result_matrix[:, 1:], axis=0)

# # Создаем фигуру
# plt.figure(figsize=(20, 8))

# # Построение столбчатой диаграммы
# bars = plt.bar(group_labels, mean_per_group, color='skyblue')

# # Настройка внешнего вида
# plt.title('Средняя ошибка по группам (усреднение по всем срезам)', fontsize=16, fontweight='bold')
# plt.xlabel('Группы точек', fontsize=14, fontweight='bold')
# plt.ylabel('Средняя ошибка, мм', fontsize=14, fontweight='bold')
# plt.xticks(rotation=90, fontsize=8, ha='center')
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Добавление значений на столбцы
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height*1.01,
#              f'{height:.2f}',
#              ha='center', va='bottom', 
#              fontsize=6, rotation=90)

# plt.tight_layout()
# plt.savefig("C:/Users/k.zhukov/Desktop/diploma/group_means.png", dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()


slice_means = np.mean(error_matrix, axis=1)

# Создаем номера срезов (от 1 до 108)
slice_numbers = np.arange(1, n_slices + 1)

mean_error = np.mean(slice_means)  # Среднее арифметическое
median_error = np.median(slice_means)  # Медиана
min_error = min(slice_means)  # Минимальное значение
max_error = max(slice_means)  # Максимальное значение

plt.figure(figsize=(16, 10))

# Основной график средних значений
plt.plot(slice_numbers, slice_means, 
         marker='o', color='b', linewidth=2, markersize=6,
         label='Средняя ошибка')

# Аппроксимация полиномом 6-й степени
if len(slice_numbers) > 6:  # Минимум 7 точек для полинома 6-й степени
    x = np.array(range(1, len(slice_means)+1))
    y = np.array(slice_means)
    coefficients = np.polyfit(x, y, 6)
    polynomial = np.poly1d(coefficients)
    x_vals = np.linspace(min(x), max(x), 500)  # Более плавная кривая
    plt.plot(x_vals, polynomial(x_vals), 'r-', linewidth=3, label='Аппроксимация (полином 6-й степени)')

# Настройка сетки
plt.grid(True, linestyle='--', alpha=0.7)

# Настройка осей
plt.xlabel("№ среза", fontsize=20, fontweight='bold')
plt.ylabel("Среднее значение погрешности расстояний, мм", fontsize=17, fontweight='bold')
plt.title("Среднее значение погрешности расстояний по срезам", fontsize=16, fontweight='bold')

# Настраиваем метки оси X с шагом 10
plt.xticks(np.arange(1, n_slices+1, 10), fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Добавляем вертикальную линию на срезе 54
plt.axvline(x=54, color='g', linestyle='--', linewidth=2, label='Срединий срез (54)')
plt.axhline(y=mean_error, color='green', linestyle='-', linewidth=2.5, 
           label=f'Среднее значение: {mean_error:.3f} мм')

# Линия медианы (фиолетовая штрих-пунктирная)
plt.axhline(y=median_error, color='purple', linestyle='-.', linewidth=2.5, 
           label=f'Медиана: {median_error:.3f} мм')

# Добавляем подпись "сред.срез" на срезе 54
y_max = np.max(slice_means)
plt.text(54, y_max * 0.95, 'сред.срез', 
         ha='center', va='top', 
         fontsize=14, fontweight='bold', color='g',
         bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))

# Подписываем каждую точку (только для каждого 10-го среза для избежания нагромождения)
for i in range(0, n_slices, 10):
    # Для среза 54 смещаем подпись выше, чтобы не пересекалась с линией
    if i == 53:  # индекс 53 соответствует срезу 54
        va_pos = 'bottom'
        y_offset = 0.015
    else:
        va_pos = 'bottom'
        y_offset = 0.005
        
    plt.text(slice_numbers[i], slice_means[i] + y_offset, 
             f'{slice_means[i]:.3f}', 
             ha='center', va=va_pos, 
             fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

# Добавляем легенду
plt.legend(loc='best', fontsize=12)

plt.tight_layout()
plt.savefig("C:/Users/k.zhukov/Desktop/diploma/mean_error+rast_per_slice_with_trend_2.png", bbox_inches='tight', dpi=300)
plt.show()

