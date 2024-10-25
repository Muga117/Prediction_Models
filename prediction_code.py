import numpy as np
import matplotlib.pyplot as plot

def calculate_msqe(y,predictedy):
    result = np.mean(np.square(y - predictedy))
    return result

def calculate_gradientdescent(x,y,slope,intercept,learning_rate):
    n = len(x)
    predictedy = slope * x + intercept

    dslope = (-2 / n) * np.sum(x * (y - predictedy))
    dintercept = (-2 / n) * np.sum(y - predictedy)
    # Update m and c
    slope -= learning_rate * dslope
    intercept -= learning_rate * dintercept
    return slope,intercept

dataset = np.genfromtxt("Nairobi Office Price Ex.csv",delimiter=",", skip_header=1)
x = dataset[:, 8]
y = dataset[:, 9]

slope = np.random.rand()
intercept = np.random.rand()

for epoch in range(10):
    predictedy = (slope * x) + intercept
    error = calculate_msqe(y,predictedy)
    slope,intercept = calculate_gradientdescent(x,y,slope,intercept,0.0001)

plot.scatter(x, y, color='red', label="Actual data")
plot.plot(x, slope * x + intercept, color='green', label="Best fit line")
plot.xlabel("Office Size")
plot.ylabel("Office Price")
plot.legend()
plot.show()

office_price = (slope * 100) + intercept
print(f"Office Price when size is 100: {office_price}")