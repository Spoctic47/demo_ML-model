import numpy as np

# Hours studied
X = np.array([1, 2, 3, 4, 5])

# Scores
y = np.array([20, 40, 60, 80, 100])

def predict(X, m, b):
    return m * X + b

def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
m = 0.0  # weight
b = 0.0  # bias
lr = 0.01  # learning rate

for epoch in range(1000):
    y_pred = predict(X, m, b)

    dm = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    m = m - lr * dm
    b = b - lr * db
hours = 6
predicted_score = predict(hours, m, b)

print("Weight (m):", m)
print("Bias (b):", b)
print("Predicted score for 6 hours:", predicted_score)
