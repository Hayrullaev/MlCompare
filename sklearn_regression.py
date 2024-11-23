
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загрузка данных
df = pd.read_csv('../data/regression_data.csv')

# Разделение на признаки и метки
X = df.drop(['y'], axis=1).values
y = df['y'].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

# Оценка ошибки модели
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')