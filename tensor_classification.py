import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Загрузка данных
df = pd.read_csv('../data/iris.csv')

# Преобразование целевого столбца в числовой формат
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Разделение на признаки и метки
X = df.drop(['species'], axis=1).values
y = df['species'].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Определение модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=0)

# Прогнозирование на тестовой выборке
y_pred = model.predict_classes(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=-1))
print(f'Accuracy: {accuracy:.2f}')