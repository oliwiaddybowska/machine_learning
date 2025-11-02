import random


def eval(x):
    result = x**2 + 3*x + 2  # math function
    return result


def deriv(x):
    result = 2*x + 3  # derivative
    return result


def gradientDescent():
    x = random.uniform(-10, 10)
    alpha = 0.01  # learning rate

    while deriv(x) > 0.00001:  # tolerance, very small numer close to 0
        x = x - alpha * deriv(x)
        return x


solution = gradientDescent()
print(f"Our findings: {solution}")
