import numpy as np


class TensorRoot:
    def __init__(self, input_size=784, hidden_sizes=(256, 128), output_size=10, seed=42):

        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.hidden_size = self.hidden_sizes[0]
        self.output_size = output_size
        self.rng = np.random.default_rng(seed)

        layer_sizes = [self.input_size, *self.hidden_sizes, self.output_size]
        self.weights = []
        self.biases = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            weight = self.rng.standard_normal((fan_in, fan_out), dtype=np.float32) * np.sqrt(2.0 / fan_in)
            bias = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(weight)
            self.biases.append(bias)

        print("Initialized weights and biases.")

    def relu(self, x) -> np.ndarray:
        return np.maximum(0, x)

    def sigmoid(self, x) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def softmax(self, x) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X) -> np.ndarray:
        self.activations = [X]
        self.hidden_zs = []

        a = X
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(a, weight) + bias
            self.hidden_zs.append(z)
            a = self.relu(z)
            self.activations.append(a)

        self.logits = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.output = self.softmax(self.logits)
        return self.output

    def relu_derivative(self, output) -> np.ndarray:
        return output > 0

    def sigmoid_derivative(self, output) -> np.ndarray:
        return output * (1 - output)

    def backward(self, X, y, learning_rate=0.1, l2=0.0) -> None:
        m = X.shape[0]
        delta = (self.output - y) / m

        dWs = [np.zeros_like(weight) for weight in self.weights]
        dbs = [np.zeros_like(bias) for bias in self.biases]

        for i in reversed(range(len(self.weights))):
            dWs[i] = np.dot(self.activations[i].T, delta) + l2 * self.weights[i]
            dbs[i] = np.sum(delta, axis=0)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * (self.hidden_zs[i - 1] > 0)

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dWs[i]
            self.biases[i] -= learning_rate * dbs[i]

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.forward(X), axis=1)

    def loss(self, X, y_one_hot) -> float:
        probs = self.forward(X)
        return float(-np.mean(np.sum(y_one_hot * np.log(probs + 1e-15), axis=1)))

    def accuracy(self, X, y) -> float:
        return float(np.mean(self.predict(X) == y))

    def get_parameters(self):
        return [weight.copy() for weight in self.weights], [bias.copy() for bias in self.biases]

    def set_parameters(self, params) -> None:
        weights, biases = params
        self.weights = [weight.copy() for weight in weights]
        self.biases = [bias.copy() for bias in biases]


def one_hot(y, num_classes):
    return np.eye(num_classes, dtype=np.float32)[y]


def iterate_minibatches(X, y, batch_size, rng):
    indices = rng.permutation(X.shape[0])
    for start in range(0, X.shape[0], batch_size):
        batch_indices = indices[start:start + batch_size]
        yield X[batch_indices], y[batch_indices]


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    import matplotlib.pyplot as plt

    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    X_all = x[:20000].astype(np.float32) / 255.0
    y_all = y[:20000].astype(np.int64)
    X_all = X_all.reshape(X_all.shape[0], -1)

    rng = np.random.default_rng(42)
    permutation = rng.permutation(X_all.shape[0])
    X_all = X_all[permutation]
    y_all = y_all[permutation]

    val_size = max(1000, X_all.shape[0] // 10)
    X_val = X_all[:val_size]
    y_val = y_all[:val_size]
    X_train = X_all[val_size:]
    y_train = y_all[val_size:]

    print("Initializing Tensor Root.")
    tensor_root = TensorRoot(hidden_sizes=(256, 128), seed=42)

    print("Preprocessing data.")
    epochs = 100
    learning_rate = 0.1
    batch_size = 128
    l2 = 1e-4
    lr_decay = 0.95
    patience = 5

    best_val_loss = float("inf")
    best_params = tensor_root.get_parameters()
    epochs_without_improvement = 0

    print("Starting training.")
    for epoch in range(epochs):
        for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size, rng):
            y_batch_one_hot = one_hot(y_batch, tensor_root.output_size)
            tensor_root.forward(X_batch)
            tensor_root.backward(X_batch, y_batch_one_hot, learning_rate=learning_rate, l2=l2)

        train_one_hot = one_hot(y_train, tensor_root.output_size)
        val_one_hot = one_hot(y_val, tensor_root.output_size)

        train_probs = tensor_root.forward(X_train)
        train_loss = float(-np.mean(np.sum(train_one_hot * np.log(train_probs + 1e-15), axis=1)))
        train_accuracy = float(np.mean(np.argmax(train_probs, axis=1) == y_train))

        val_probs = tensor_root.forward(X_val)
        val_loss = float(-np.mean(np.sum(val_one_hot * np.log(val_probs + 1e-15), axis=1)))
        val_accuracy = float(np.mean(np.argmax(val_probs, axis=1) == y_val))

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_acc={train_accuracy:.4f} train_loss={train_loss:.4f} | "
            f"val_acc={val_accuracy:.4f} val_loss={val_loss:.4f} | "
            f"lr={learning_rate:.5f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = tensor_root.get_parameters()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

        learning_rate *= lr_decay

    tensor_root.set_parameters(best_params)
    final_val_accuracy = tensor_root.accuracy(X_val, y_val)
    print(f"Best validation accuracy: {final_val_accuracy:.4f}")

    # Visualize some predictions
    num_samples = 10
    sample_indices = rng.choice(X_val.shape[0], num_samples, replace=False)
    X_samples = X_val[sample_indices]
    y_samples = y_val[sample_indices]
    predictions = tensor_root.predict(X_samples)

    for i in range(num_samples):
        test_image = X_samples[i]
        true_label = y_samples[i]

        # forward expects a batch (N, 784), so wrap one image as (1, 784)
        predicted_prob = tensor_root.forward(test_image.reshape(1, -1))[0]
        predicted_label = int(np.argmax(predicted_prob))
        confidence = float(np.max(predicted_prob) * 100)

        image_2d = test_image.reshape(28, 28)

        plt.figure(figsize=(10, 10))
        plt.imshow(image_2d, cmap="gray")

        title_color = "green" if predicted_label == true_label else "red"
        plt.title(f"AI Predict: {predicted_label} (conf: {confidence:.2f}%)\n True Label: {true_label}",
                  color=title_color)

        plt.axis("off")

        print(
            f"Sample {i + 1}: True Label = {true_label}, Predicted Label = {predicted_label}, Confidence = {confidence:.2f}%")
        plt.show()
