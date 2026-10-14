import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine

print("Библиотеки подключены успешно")

wine = load_wine()

df = pd.DataFrame(
    data=wine.data,
    columns=wine.feature_names
)

df["target"] = wine.target

print("\nПервые 5 строк датасета:")
print(df.head())

print("\nРазмер датасета:")
print(df.shape)

print("\nНазвания столбцов:")
print(df.columns)

print("\nОсновная статистика:")
print(df.describe())

print("\nКоличество пропущенных значений:")
print(df.isnull().sum())

print("\nКоличество объектов каждого класса:")
print(df["target"].value_counts().sort_index())

# 1. Столбчатая диаграмма количества объектов по классам

df["target"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Количество объектов каждого класса")
plt.xlabel("Класс")
plt.ylabel("Количество объектов")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 2. Гистограмма содержания алкоголя

plt.hist(
    df["alcohol"],
    bins=20
)

plt.title("Распределение содержания алкоголя")
plt.xlabel("Содержание алкоголя")
plt.ylabel("Количество объектов")
plt.tight_layout()
plt.show()

# 3. Диаграмма рассеяния

sns.scatterplot(
    data=df,
    x="alcohol",
    y="color_intensity",
    hue="target",
    palette="deep"
)

plt.title("Зависимость содержания алкоголя и интенсивности цвета")
plt.xlabel("Содержание алкоголя")
plt.ylabel("Интенсивность цвета")
plt.tight_layout()
plt.show()

# 4. Диаграмма размаха

sns.boxplot(
    data=df,
    x="target",
    y="alcohol"
)

plt.title("Содержание алкоголя по классам")
plt.xlabel("Класс")
plt.ylabel("Содержание алкоголя")
plt.tight_layout()
plt.show()

# 5. Матрица корреляции признаков

corr_matrix = df.drop(columns=["target"]).corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)

plt.title("Матрица корреляции признаков")
plt.tight_layout()
plt.show()