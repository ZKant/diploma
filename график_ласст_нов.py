import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Замените 'имя_файла.csv' на путь к вашему файлу
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/diploma/Матрица_ошибок_по_радиусу.csv', delimiter=';')
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Весь_фантом/Максимальный_радиус_вект_ошибка/Матрица_ошибок_по_радиусу.csv', delimiter=';')
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/diploma/Матрица_ошибок_по_координате_3.csv', delimiter=';')
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/diploma/Матрица_ошибок_по_расстояниям_new.csv', delimiter=';')
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/Диплом/Фантом_3/Фантом_3_результаты/Матрица ошибок по расстояниям_средний_новое.csv', delimiter=';')

# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/diploma/СРАВНЕНИЕ/вставка1/Матрица ошибок по расстояниям_72.csv', delimiter=';')
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/diploma/СРАВНЕНИЕ/вставка2/Матрица ошибок по расстояниям_средний_72.csv', delimiter=';')
error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/diploma/СРАВНЕНИЕ/вставка3/Матрица ошибок по расстояниям_средний_72.csv', delimiter=';')

print(error_matrix.shape)

import numpy as np

# Предположим, что у вас есть 2D-массив (матрица)

# Сложим значения в первых 8 столбцах каждой строки
# sum_of_columns = np.sum(error_matrix[:, :8], axis=1)

# Если нужно сложить суммы всех строк
total_sum = np.sum(error_matrix[:, :8])

# print("Сумма значений в первых 8 столбцах каждой строки:",)
print("Общая сумма всех этих значений:", total_sum)

group_indices_list = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [9,10,11,12],
    [13,14,15,16,17,18,19,20],
    [21,22,23,24,25,26,27,28],
    [29,30,31,32],
    [33,34,35,36,37,38,39,40],
    [41,42,43,44,45,46,47,48],
    [49,50,51,52,53,54,55,56],
    [57,58,59,60,61,62,63,64],
    [65,66,67,68,69,70,71,72]
]

group_labels = [
    "68.3 мм", "80.0 мм", "93.0 мм", "104.7 мм", "120.0 мм", 
    "121.3 мм", "128.0 мм", "143.2 мм", "148.7 мм", "155.4 мм"
]

n_slices = error_matrix.shape[0]

zero_column = np.zeros((error_matrix.shape[0], 1))

# Объединяем по горизонтали
error_matrix = np.concatenate((zero_column, error_matrix), axis=1)

# Готовим финальную матрицу: 108 строк, 13 столбцов (1 + 12 групп)
result_matrix = np.zeros((n_slices, 11))
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


# print(result_matrix[:, 1:])

# Вычисляем средние значения по срезам для каждой группы
mean_per_group = np.mean(result_matrix[:, 1:], axis=0)

print(mean_per_group)

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
plt.title('Средняя погрешность расстояний (20 мм) по группам периметров, обр. центрами контуров и центром кожуха', 
          fontsize=19, fontweight='bold')
plt.xlabel('Группы пар контуров с совпадающим периметром', fontsize=17, fontweight='bold')
plt.ylabel('Средняя погрешность расстояний, мм', fontsize=17, fontweight='bold')

# Жирные подписи на осях
plt.xticks(rotation=45, fontsize=12, ha='center', fontweight='bold')
plt.yticks(fontsize=13, fontweight='bold')

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
plt.title("Среднее значение погрешности групп расстояний (по эталону 20 мм) по срезам", fontsize=16, fontweight='bold')

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

