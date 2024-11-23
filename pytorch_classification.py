import torch
import torch.nn as np
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


# Указываем путь к файлу
file_path = 'data/regression_data.csv'  # Предполагается, что файл находится в поддиректории data

# Читаем данные из CSV-файла
try:
    df = pd.read_csv(file_path)
except FileNotFoundError as e:
    print(f"Не удалось найти файл: {e}")
else:
    # Просматриваем первые несколько строк данных
    print(df.head())

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

# Преобразование данных в тензоры
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.int64))

# Определение модели
class Net(np.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = np.Linear(4, 16)
        self.relu = np.ReLU()
        self.fc2 = np.Linear(16, 3)
        self.softmax = np.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

# Создание экземпляра модели
model = Net()

# Определение оптимизатора и функции потерь
criterion = np.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 200
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Прогнозирование на тестовой выборке
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred.data, 1)

# Оценка точности модели
accuracy = accuracy_score(y_test, predicted.numpy())
print(f'Accuracy: {accuracy:.2f}')