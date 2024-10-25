import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading of the CSV file
df = pd.read_csv("Nairobi Office Price Ex.csv")

# Extracting of relevant columns
size = df['SIZE'].values
price = df['PRICE'].values

# Normalizing the data to ensure gradient descent works effectively
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

normalized_size = normalize(size)
normalized_price = normalize(price)

# Defining Mean Squared Error (MSE) function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Defining Gradient Descent function
def gradient_descent(x, y, m, c, learning_rate):
    n = len(y)
    y_pred = m * x + c
    dm = (-2/n) * sum(x * (y - y_pred))
    dc = (-2/n) * sum(y - y_pred)
    m -= learning_rate * dm
    c -= learning_rate * dc
    return m, c

# Training loop
def train_linear_regression(x, y, learning_rate=0.01, epochs=1000, print_frequency=100):
    m = np.random.rand()  # Random initial slope
    c = np.random.rand()  # Random initial intercept
    for epoch in range(epochs):
        y_pred = m * x + c
        error = mean_squared_error(y, y_pred)
        if (epoch + 1) % print_frequency == 0:
            print(f'Epoch {epoch+1}: MSE = {error}')
        m, c = gradient_descent(x, y, m, c, learning_rate)
    return m, c

m, c = train_linear_regression(normalized_size, normalized_price, learning_rate=0.01, epochs=1000)

# Plotting the line of best fit
def plot_best_fit(x, y, m, c):
    plt.scatter(x, y, color='blue', label='Data points')
    y_pred = m * x + c
    plt.plot(x, y_pred, color='red', label='Best fit line')
    plt.xlabel('Normalized Office Size')
    plt.ylabel('Normalized Office Price')
    plt.legend()
    plt.show()

plot_best_fit(normalized_size, normalized_price, m, c)

# Predicting office price for a normalized size of 100 sq. ft.
size_to_predict = 100
normalized_size_to_predict = (size_to_predict - np.min(size)) / (np.max(size) - np.min(size))
predicted_normalized_price = m * normalized_size_to_predict + c

# Reversing the normalization to get the actual predicted price
predicted_price = predicted_normalized_price * (np.max(price) - np.min(price)) + np.min(price)
print(f'Predicted office price for 100 sq. ft. = {predicted_price}')
