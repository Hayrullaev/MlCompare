
import torch
import torch.nn as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Загрузка данных
df = pd.read_csv('../data/regression_data.csv')

# Разделение на признаки и метки
X = df.drop(['y'], axis=1).values
y = df['y'].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование данных в тензоры
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
X_test_tensor = torch.from_numpy(X_test.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32))

# Определение модели
class Net(np.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = np.Linear(1, 1)

    def forward(self, x):
        out = self.fc1(x)
        return out

# Создание экземпляра модели
model = Net()

# Определение оптимизатора и функции потерь
criterion = np.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

# Оценка ошибки модели
mse = mean_squared_error(y_test, y_pred.detach().numpy())
print(f'MSE: {mse:.2f}')