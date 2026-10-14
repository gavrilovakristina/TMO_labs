import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv("houseprices.csv")

print("Размер датасета:")
print(data.shape)

print("\nПервые 5 строк выбранных признаков:")
print(
    data[
        ["GarageYrBlt", "MasVnrType", "SaleType", "SalePrice"]
    ].head()
)

print("\nКоличество пропусков:")
print(
    data[
        ["GarageYrBlt", "MasVnrType", "SaleType", "SalePrice"]
    ].isnull().sum()
)

num_imputer = SimpleImputer(strategy="median")

data[["GarageYrBlt"]] = num_imputer.fit_transform(
    data[["GarageYrBlt"]]
)

print("\nПропусков в GarageYrBlt после обработки:")
print(data["GarageYrBlt"].isnull().sum())

print("\nМедиана GarageYrBlt:")
print(num_imputer.statistics_[0])

cat_imputer = SimpleImputer(strategy="most_frequent")

data[["MasVnrType"]] = cat_imputer.fit_transform(
    data[["MasVnrType"]]
)

print("\nПропусков в MasVnrType после обработки:")
print(data["MasVnrType"].isnull().sum())

encoder = LabelEncoder()

data["MasVnrType_encoded"] = encoder.fit_transform(
    data["MasVnrType"]
)

print("\nКатегории MasVnrType:")
print(encoder.classes_)

print("\nПример кодирования:")
print(
    data[
        ["MasVnrType", "MasVnrType_encoded"]
    ].head()
)

scaler = StandardScaler()

data["SalePrice_scaled"] = scaler.fit_transform(
    data[["SalePrice"]]
)

print("\nЦена до и после масштабирования:")
print(
    data[
        ["SalePrice", "SalePrice_scaled"]
    ].head()
)

print("\nСреднее значение после масштабирования:")
print(round(data["SalePrice_scaled"].mean(), 3))

print("\nСтандартное отклонение после масштабирования:")
print(round(data["SalePrice_scaled"].std(ddof=0), 3))

print("\nИтоговые данные:")
print(
    data[
        [
            "GarageYrBlt",
            "MasVnrType",
            "MasVnrType_encoded",
            "SalePrice",
            "SalePrice_scaled"
        ]
    ].head()
)

print("\nКоличество пропусков после обработки:")
print(
    data[
        ["GarageYrBlt", "MasVnrType"]
    ].isnull().sum()
)