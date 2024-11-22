import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        
        self.W1 = np.random.uniform(-0.1, 0.1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.uniform(-0.1, 0.1, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    def activation(self, x):
        """Applies the activation function."""
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError("Unsupported activation function")

    def activation_derivative(self, x):
        """Computes the derivative of the activation function."""
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return (x > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function")
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        out = self.activation(self.z2)
        self.activations = {
            "input": X,
            "hidden_pre_activation": self.z1,
            "hidden_activation": self.a1,
            "output_pre_activation": self.z2,
            "output_activation": out
        }
        return out

    def backward(self, X, y):
        n_samples = X.shape[0]

        output_error = self.activations["output_activation"] - y
        output_delta = output_error * self.activation_derivative(self.activations["output_pre_activation"])

        grad_W2 = np.dot(self.activations["hidden_activation"].T, output_delta) / n_samples
        grad_b2 = np.sum(output_delta, axis=0) / n_samples

        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.activation_derivative(self.activations["hidden_pre_activation"])

        grad_W1 = np.dot(X.T, hidden_delta) / n_samples
        grad_b1 = np.sum(hidden_delta, axis=0) / n_samples

        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

        self.gradients = {
            "output_error": output_error,
            "output_delta": output_delta,
            "grad_W2": grad_W2,
            "grad_b2": grad_b2,
            "hidden_error": hidden_error,
            "hidden_delta": hidden_delta,
            "grad_W1": grad_W1,
            "grad_b1": grad_b1,
    }

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    hidden_features = mlp.activations["hidden_activation"]
    gradients = mlp.gradients

    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)

       # Hidden Features Scatter Plot
    ax_hidden.set_title("Hidden Layer Features")
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap="bwr", alpha=0.7
    )
    ax_hidden.set_xlabel("Hidden Neuron 1")
    ax_hidden.set_ylabel("Hidden Neuron 2")
    ax_hidden.set_zlabel("Hidden Neuron 3")

    # Hyperplane Visualization
    # Plot hyperplane boundaries in the hidden space
    # Assume two dimensions of the hidden space for simplicity
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, 50),
        np.linspace(-1, 1, 50)
    )
    hidden_hyperplane = -(mlp.W2[0] * xx + mlp.W2[1] * yy + mlp.b2[0]) / (mlp.W2[2] + 1e-5)
    ax_hidden.plot_surface(xx, yy, hidden_hyperplane, alpha=0.3, color="orange")

    # Input Space Decision Boundary
    ax_input.set_title("Input Space Decision Boundary")
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    input_space = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(input_space)
    zz = predictions.reshape(xx.shape)
    ax_input.contourf(xx, yy, zz, alpha=0.7, cmap="bwr")
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolors="k")

    # Gradient Visualization
    ax_gradient.set_title("Gradient Visualization")
    for i in range(mlp.W1.shape[0]):
        for j in range(mlp.W1.shape[1]):
            gradient_magnitude = gradients["grad_W1"][i, j]
            ax_gradient.plot(
                [i, j],
                [0, gradient_magnitude],
                "k-",
                lw=np.abs(gradient_magnitude) * 5  # Line width proportional to gradient
            )
            ax_gradient.scatter(
                i, 0, s=50, c="blue"
            )  # Node in input layer
            ax_gradient.scatter(
                j, gradient_magnitude, s=50, c="red"
            )  # Node in hidden layer
    
def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)