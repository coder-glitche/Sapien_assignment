import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve

# Set random seed for reproducibility
np.random.seed(42)

# loading data
def load_cifar10_batch(batch_filename):
    with open(batch_filename, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        X = batch['data']
        y = np.array(batch['labels'])
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return X, y

def load_cifar10_data(dataset_path):
    X_train, y_train = [], []
    for i in range(1, 6):
        X_batch, y_batch = load_cifar10_batch(os.path.join(dataset_path, f'data_batch_{i}'))
        X_train.append(X_batch)
        y_train.append(y_batch)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test, y_test = load_cifar10_batch(os.path.join(dataset_path, 'test_batch'))
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    return X_train, y_train, X_test, y_test

dataset_path = 'cifar-10-python/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_cifar10_data(dataset_path)

# neural network with Adam optimization
class ImprovedNeuralNetworkWithAdam:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim1) * 0.01
        self.b1 = np.zeros((1, hidden_dim1))
        self.W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.01
        self.b2 = np.zeros((1, hidden_dim2))
        self.W3 = np.random.randn(hidden_dim2, output_dim) * 0.01
        self.b3 = np.zeros((1, output_dim))
        
        # Adam parameters
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.mW3, self.vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.mb3, self.vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return Z > 0
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3
    
    def compute_loss(self, A3, y):
        m = y.shape[0]
        log_probs = -np.log(A3[range(m), y])
        loss = np.sum(log_probs) / m
        return loss
    
    def adam_update(self, grads, learning_rate=0.0001):
        self.t += 1
        # Update parameters with Adam optimizer
        for param, grad, m, v in zip([self.W1, self.b1, self.W2, self.b2, self.W3, self.b3],
                                    grads,
                                    [self.mW1, self.mb1, self.mW2, self.mb2, self.mW3, self.mb3],
                                    [self.vW1, self.vb1, self.vW2, self.vb2, self.vW3, self.vb3]):
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def backward(self, X, y, learning_rate=0.0001):
        m = y.shape[0]
        y_one_hot = np.zeros_like(self.A3)
        y_one_hot[np.arange(m), y] = 1
        
        dZ3 = self.A3 - y_one_hot
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        grads = [dW1, db1, dW2, db2, dW3, db3]
        self.adam_update(grads, learning_rate)

    def predict(self, X):
        A3 = self.forward(X)
        return np.argmax(A3, axis=1)

# training
input_dim = 32 * 32 * 3  # 32x32 RGB images
hidden_dim1 = 256
hidden_dim2 = 128
output_dim = 10  # 10 classes
model = ImprovedNeuralNetworkWithAdam(input_dim, hidden_dim1, hidden_dim2, output_dim)

num_epochs = 30
learning_rate = 0.0001
batch_size = 64

csv_filename = "training_metrics.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Loss', 'Precision', 'Recall', 'F1'])

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

for epoch in range(num_epochs):
    permutation = np.random.permutation(X_train_flattened.shape[0])
    X_train_shuffled = X_train_flattened[permutation]
    y_train_shuffled = y_train[permutation]
    
    for i in range(0, X_train_flattened.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]
        A3 = model.forward(X_batch)
        loss = model.compute_loss(A3, y_batch)
        model.backward(X_batch, y_batch, learning_rate=learning_rate)
    
    y_pred = model.predict(X_test_flattened)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch + 1, loss, precision, recall, f1])
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Plotting
for class_index in range(10):
    class_mask = (y_test == class_index)
    precisions, recalls, _ = precision_recall_curve(class_mask, y_pred == class_index)
    plt.plot(recalls, precisions, label=f'Class {class_index}')
    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for CIFAR-10 Classes')
plt.legend()
plt.show()
