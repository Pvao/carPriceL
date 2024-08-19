import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import currencyRate as cr
import customsDuty
import delivery

pd.set_option("future.no_silent_downcasting", True)
##Чтение файла
cd = r'C:\data\cars_raw.csv'
df = pd.read_csv(cd)

##Анализ первичных данных
print(df.head(3))
print(len(df))
df.info()
df.describe()

##Чистка первичных данных
df.drop_duplicates(inplace=True)
df = df.dropna()
df = df[df['Price'].notna()]

##Считаем количество уникальных значений по каждому столбцу и выводим результаты
for column in df.columns:
    print(f"Количество уникальных значений в столбце {column}: {len(df[column].unique())}")


##Подготовка и кодирование данных
##Обработка колонки Тип топлива
df['FuelType'] = df['FuelType'].replace({"Gasoline Fuel":'Gasoline'
                                        ,'Electric Fuel System':'Gasoline'
                                        ,'E85 Flex Fuel':'Gasoline'
                                        ,'Flex Fuel Capability':'Gasoline'
                                        ,'Gasoline/Mild Electric Hybrid':'Hybrid'
                                        ,'Flexible Fuel':'Gasoline'})
df['FuelType'] = df['FuelType'].replace({"–": 0, "Gasoline":1, "Diesel": 2, "Electric": 3,"Hybrid":4})


#Обработка колонки Цена
df['Price'] = df['Price'].str.replace('$','')
df['Price'] = df['Price'].str.replace(',','')
df['Price'] = pd.to_numeric(df['Price'],errors = 'coerce')
df = df[df['Price'].notna()]


#Обработка колонки Коробка Передач
df['Transmission'] = df['Transmission'].apply(lambda x: "AT" if "AUTOMATIC" in str(x).upper() else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "AT" if "A/T" in str(x) else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "AT" if "AUTO" in str(x).upper() else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "AT" if "VARIABLE" in str(x).upper() else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "AT" if "A" == str(x).upper() else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "AT" if "CTV" in str(x).upper() else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "AT" if "CVT" in str(x).upper() else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "AT" if "7-Speed Double-clutch" in str(x) else x)

df['Transmission'] = df['Transmission'].apply(lambda x: "MT" if "MANUAL" in str(x).upper() else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "MT" if "M/T" in str(x).upper() else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "MT" if "Transmission w/Dual Shift Mode" in str(x) else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "MT" if "PDK" in str(x) else x)
df['Transmission'] = df['Transmission'].apply(lambda x: "MT" if "7-Speed" == str(x) else x)
df['Transmission'] = df['Transmission'].replace({"MT":1, "AT":2,"–":0})

#Обработка колонки Привод
df['Drivetrain'] = df['Drivetrain'].str.replace('Front-wheel Drive','FWD') #Передний привод
df['Drivetrain'] = df['Drivetrain'].str.replace('Front Wheel Drive','FWD') #Передний привод
df['Drivetrain'] = df['Drivetrain'].str.replace('Rear-wheel Drive','RWD') #Задний привод
df['Drivetrain'] = df['Drivetrain'].str.replace('Four wheel drive','4WD') #Полный привод
df['Drivetrain'] = df['Drivetrain'].str.replace('Four-wheel Drive','4WD') #Полный привод
df['Drivetrain'] = df['Drivetrain'].str.replace('AWD','4WD') #Полный привод
df['Drivetrain'] = df['Drivetrain'].str.replace('All-wheel Drive','4WD') #Полный привод
df['Drivetrain'] = df['Drivetrain'].replace({"–":0,"FWD":1, "RWD":2,"4WD":3})

##Обработка колонки Новое/ б/у
df['Used/New'] = df['Used/New'].apply(lambda x: "New" if x != "Used" else x)
df['Used/New'] = df['Used/New'].replace({"New":0, "Used":1})

##Обработка колонки Марка
df['Make'] = df['Make'].replace({"Toyota": 1, 'Ford': 2, 'RAM': 3, 'Lexus': 4, 'Honda': 5, 'Mercedes-Benz': 6, 'Dodge': 7
                                    ,'Subaru': 8, 'BMW': 9, 'Audi': 10, 'Volvo': 11, 'Lincoln': 12, 'Land': 13, 'Acura': 14
                                    ,'Chevrolet': 15, 'Tesla': 16, 'Jeep': 17,'Chrysler': 18, 'Kia': 19, 'Volkswagen': 20
                                    ,'Porsche': 21, 'Nissan': 22, 'Hyundai': 23,'GMC': 24,'Buick': 25,'Genesis': 26
                                    ,'INFINITI': 27, 'Cadillac': 28, 'Alfa': 29,'FIAT': 30, 'Jaguar': 31, 'MINI': 32
                                    ,'Maserati': 33, 'Mitsubishi': 34, 'Bentley': 35, 'Mercury': 36, 'Lamborghini': 37
                                    ,'Scion': 38, 'Saturn': 39, 'Mazda': 40})

##Выбор Целевых параметров модели
X = df[['Year', 'Make', 'Used/New', 'Drivetrain', 'FuelType', 'Transmission', 'Mileage']]
y = df['Price']

# Создание отдельных графиков для каждой переменной:

for col in X.columns:
    plt.figure(figsize=(10,6))
    plt.scatter(X[col], y)
    plt.title(f'Price vs {col}')
    plt.xlabel(col)
    plt.ylabel('price')
    plt.show()

##Использование seaborn для создания парных графиков (scatter plot):
sns.pairplot(df[['Price', 'Year', 'Make', 'Used/New',
                 'Drivetrain', 'FuelType', 'Transmission', 'Mileage']])
plt.show()


##Модуль подготовки модели
X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y, test_size=0.2, random_state=30)

##Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

## Прогон тестовой выборки
predictions = model.predict(X_test)

##оценка качества
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error (MSE): {mse}')
print("Test set score (model.score): {:.2f}".format(model.score(X_test, y_test)))


##Модуль предсказания цены
new_Year = 2022         #Год выпуска
new_Make = 5            #Марка
new_UsedNew = 1         #Новое/неновое
new_Drivetrain = 1      #Привод
new_FuelType = 2        #Тип двигателя
new_Transmission = 3    #Коробка передач
new_Mileage = 10000     #Пробег

## Предсказание цены
new_data = [[new_Year, new_Make, new_UsedNew, new_Drivetrain, new_FuelType, new_Transmission, new_Mileage]]
prediction = model.predict(new_data)

## Пересчет цены в России
k_cur = cr.getCurrency("usd/rub") #Курс валют
k_t = customsDuty.getDuty("200", new_Year) #Таможенные пошлины
k_delivery = delivery.getPrice(2) #Доставка в Россию
carPrice = prediction[0] * k_cur + k_t + k_delivery

print(f'Ожидаемая стоимость машины в Росии: {carPrice.__format__(".2f")} рублей')
