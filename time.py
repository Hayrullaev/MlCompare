import timeit

setup_code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Загрузка данных
df = pd.read_csv('../data/iris.csv')

# Преобразование целевого столбца в числовой формат
target_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df['species'] = df['species'].map(target_mapping)

# Разделение на признаки и метки
X = df.drop(['species'], axis=1).values
y = df['species'].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""

test_code = """
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
"""

time = timeit.timeit(setup=setup_code, stmt=test_code, number=10)
print(f'Scikit-learn execution time: {time:.2f} seconds')