import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Загрузка данных
df = pd.read_csv('../data/regression_data.csv')

# Разделение на признаки и метки
X = df.drop(['y'], axis=1).values
y = df['y'].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=0)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

# Оценка ошибки модели
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse:.2f}')