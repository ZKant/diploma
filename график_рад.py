import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных (ваш путь)
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Весь_фантом/Средний_радиус_вект_ошибка_НОВОЕ/Матрица_ошибок_по_радиусу_средний_новое.csv', delimiter=';')

error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Матрица_ошибок_по_радиусу_средний_новое.csv', delimiter=';')
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/Диплом/Фантом_3/Фантом_3_результаты/Матрица ошибок по координате_средний_новое.csv', delimiter=';')
# error_matrix = np.loadtxt('C:/Users/k.zhukov/Desktop/Диплом/Фантом_1/Фантом_1_результаты/Весь_фантом/Средний_радиус_вект_ошибка_НОВОЕ/Матрица ошибок по координате_средний_новое.csv', delimiter=';')


# group_indices_list = [
#     [1,2,3,4],             # rast = 20.0, r: 1.5
#     [5,6,7,8],             # rast = 26.9, r: 5
#     [9,10,11,12],          # rast = 28.3, r: 1.5
#     [13,14,15,16],         # rast = 40.0, r: 1.5
#     [17,18,19,20],         # rast = 41.2, r: 5
#     [21,22,23,24,25,26,27,28], # rast = 44.7, r: 1.5
#     [29,30,31,32],         # rast = 56.6, r: 1.5
#     [33,34,35,36],         # rast = 60.0, r: 1.5
#     [37,38,39,40,41,42,43,44], # rast = 63.2, r: 1.5
#     [45,46,47,48],         # rast = 70.7, r: 3.0
#     [49,50,51,52],         # rast = 70.7, r: 5
#     [53,54,55,56,57,58,59,60]  # rast = 72.1, r: 1.5
# ]


group_indices_list = [
    [1,2,3,4],             # rast = 20.0, r: 1.5
    [5,6,7,8],          # rast = 28.3, r: 1.5
    [9,10,11,12],         # rast = 40.0, r: 1.5
    [13,14,15,16,17,18,19,20], # rast = 44.7, r: 1.5
    [21,22,23,24],         # rast = 56.6, r: 1.5
    [25,26,27,28],         # rast = 60.0, r: 1.5
    [29,30,31,32,33,34,35,36], # rast = 63.2, r: 1.5
    [37,38,39,40,41,42,43,44]  # rast = 72.1, r: 1.5
]

group_labels = [
    "20.0 мм (r=3)",
    # "26.9 мм (r=5)",
    "28.3 мм (r=3)",
    "40.0 мм (r=3)",
    # "41.2 мм (r=5)",
    "44.7 мм (r=3)",
    "56.6 мм (r=3)",
    "60.0 мм (r=3)",
    "63.2 мм (r=3)",
    # "70.7 мм (r=3.0)",
    # "70.7 мм (r=5)",
    "72.1 мм (r=3)",
]

n_slices = error_matrix.shape[0]

# Формирование результирующей матрицы
result_matrix = np.zeros((n_slices, 9))
result_matrix[:, 0] = error_matrix[:, 0]  # большая окружность

# Усреднение по группам
for i, col_idxs in enumerate(group_indices_list):
    result_matrix[:, i+1] = np.mean(error_matrix[:, col_idxs], axis=1)

# Тепловая карта (без изменений)
xticklabels = ["Больш.окр."] + group_labels

plt.figure(figsize=(19, 18))
sns.heatmap(
    result_matrix[:, 1:],
    cmap="coolwarm",
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
plt.savefig("C:/Users/k.zhukov/Desktop/diploma/heatmap_grouped_errors.png", bbox_inches='tight')
plt.show()
plt.close()

# ===================================================
# ИЗМЕНЕННЫЙ БЛОК ДЛЯ ВТОРОГО ГРАФИКА
# ===================================================

# Вычисляем средние для всех групп ВКЛЮЧАЯ большую окружность
big_circle_mean = np.mean(result_matrix[:, 0])  # среднее для большой окружности
group_means = np.mean(result_matrix[:, 1:], axis=0)  # средние для 12 групп
all_means = np.concatenate(([big_circle_mean], group_means))  # объединенный массив

# Создаем новые метки групп
new_group_labels = ["Кожух"] + group_labels

# Определяем цвета и подписи для каждого столбца
colors = []
bar_labels = []

# Цвета для радиусов
color_map = {
    1.5: 'dodgerblue',   # синеватый для 1.5 мм
    3.0: 'salmon',       # красноватый для 3.0 мм
    5.0: 'crimson'       # насыщенный красный для 5.0 мм
}

# Обработка большой окружности (специальный стиль)
colors.append('gray')  # серый цвет
bar_labels.append("Бол.")

# Обработка остальных групп
for label in group_labels:
    # Извлекаем радиус из строки метки
    start_idx = label.find('(r=') + 3
    end_idx = label.find(')', start_idx)
    radius_str = label[start_idx:end_idx]
    radius = float(radius_str)
    
    # Сохраняем подпись радиуса
    bar_labels.append(f"{radius:.1f}" if radius.is_integer() else f"{radius}")
    
    # Выбираем цвет по радиусу
    colors.append(color_map.get(radius, 'gray'))

# Пересчитываем общие статистики с учетом большой окружности
overall_mean = np.mean(group_means)
overall_median = np.median(group_means)

# Создаем фигуру
plt.figure(figsize=(25, 20))

# Построение столбцов с цветовой кодировкой
bars = plt.bar(new_group_labels, all_means, color=colors, alpha=0.7)

# Линия, соединяющая средние значения
x_positions = np.arange(len(new_group_labels))
plt.plot(x_positions, all_means, 'o-', color='red', linewidth=2, markersize=8)

# Добавляем подписи ВНУТРИ столбцов
for i, bar in enumerate(bars):
    height = bar.get_height()
    # Для больших столбцов - белый текст, для малых - черный
    text_color = 'white' if height > max(all_means)*0.3 else 'black'
    plt.text(
        bar.get_x() + bar.get_width()/2., 
        height/1.2, 
        bar_labels[i],
        ha='center', 
        va='center', 
        fontsize=12,
        fontweight='bold',
        color=text_color
    )

# Горизонтальные линии для среднего и медианы
plt.axhline(y=overall_mean, color='green', linestyle='--', linewidth=3, 
            label=f'Среднее по всем группам: {overall_mean:.2f} мм')
plt.axhline(y=overall_median, color='purple', linestyle='--', linewidth=3, 
            label=f'Медиана по всем группам: {overall_median:.2f} мм')

# Настройка оформления Средняя по всем срезам от номера группы
plt.title('Погрешность максимального радиуса для каждой группы', 
          fontsize=37, fontweight='bold')
plt.xlabel('ГРУППА по расстоянию от центра фантома и радиусу', fontsize=35, fontweight='bold')
plt.ylabel('Значение погрешности радиуса, мм', fontsize=35, fontweight='bold')
plt.xticks(rotation=45, fontsize=23, ha='center', fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Добавление значений НАД столбцами
for i, value in enumerate(all_means):
    plt.text(
        x_positions[i], 
        value + 0.02 * max(all_means), 
        f'{value:.2f}', 
        ha='center', 
        va='bottom', 
        fontsize=10, 
        fontweight='bold', 
        color='red',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
    )

# Легенда и пределы
plt.legend(fontsize=33, loc='best', frameon=True, shadow=True)
plt.ylim(min(all_means) * 0.90, max(all_means) * 1.15)

# Сохранение и отображение
plt.tight_layout()
plt.savefig("C:/Users/k.zhukov/Desktop/diploma/group_means_with_trend_rad.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ===================================================
# ОСТАЛЬНОЙ КОД БЕЗ ИЗМЕНЕНИЙ
# ===================================================

# Вычисление средних по срезам (без большой окружности)
slice_means = np.mean(result_matrix[:, 1:], axis=1)

# Создаем номера срезов (от 1 до 108)
slice_numbers = np.arange(1, n_slices + 1)

plt.figure(figsize=(25, 20))

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
plt.xlabel("№ среза", fontsize=35, fontweight='bold')
plt.ylabel("Среднее значение погрешности максимального радиуса, мм", fontsize=35, fontweight='bold')
plt.title("Средняя погрешность максимального радиуса по срезу от номера среза", fontsize=37, fontweight='bold')

# Настраиваем метки оси X с шагом 10
plt.xticks(np.arange(1, n_slices+1, 10), fontsize=23, fontweight='bold')
plt.yticks(fontsize=30, fontweight='bold')

# Добавляем вертикальную линию на срезе 54
plt.axvline(x=54, color='g', linestyle='--', linewidth=2, label='Срединий срез (54)')

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
# plt.legend(loc='best', fontsize=12)
plt.legend(fontsize=33, loc='best', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig("C:/Users/k.zhukov/Desktop/diploma/mean_error+rast_per_slice_with_trend_2.png", bbox_inches='tight', dpi=300)
plt.show()