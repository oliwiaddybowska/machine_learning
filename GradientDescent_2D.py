import random

class GradientDescent2D():

    def __init__(self, alpha, tol, maxIter):
        self.alpha = alpha      # learning rate
        self.tol = tol          # tolerance
        self.maxIter = maxIter  # maximum number of iterations

    def eval(self, x, y):
        value = x**2 + y**2 + 3*x + 2*y + 1
        return value

    def deriv(self, x, y):
        x_value = 2*x + 3
        y_value = 2*y + 2
        return x_value, y_value

    def optimize(self, x_initial, y_initial):
        iter = 0
        x, y = x_initial, y_initial
        x_deriv, y_deriv = self.deriv(x, y)
        while (abs(x_deriv) > self.tol or abs(y_deriv) > self.tol) and iter <= self.maxIter:
            x = x - self.alpha * x_deriv
            y = y - self.alpha * y_deriv
            iter += 1
            x_deriv, y_deriv = self.deriv(x, y)

        return x, y, self.eval(x,y)

# Generate random initial points
initial_x = random.uniform(-10, 10)
initial_y = random.uniform(-10, 10)

# Initialize the GradientDescent2D class
GD = GradientDescent2D(alpha=0.1, tol=1e-6, maxIter=1000)

# Optimize to find the minimum
optimal_x, optimal_y, optimal_value = GD.optimize(initial_x, initial_y)

# Print the results
print(f"Initial point: ({initial_x}, {initial_y})")
print(f"Optimal point: ({optimal_x}, {optimal_y})")
print(f"Function value at optimal point: {optimal_value}")