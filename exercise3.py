import numpy as np

class SimplifiedLogisticRegression: 
    def __init__(self, n_order=3, n_max_iter=1000):
        """Linear regression using a first order polynomial model for 2D data."""
        self.n_order = n_order
        self.n_max_iterations = n_max_iter
        self.loss = []

        self._theta = np.zeros(3)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _soft_predict(self, X):
        """σ(θ_0 + θ_1 * x_0 + θ_2 * x_1)"""
        #assert self._theta.ndim == 3, "θ should be a 3D array"

        nr_of_samples = X.shape[0]
        y_hat = np.zeros(nr_of_samples)
        for sample_i in range(nr_of_samples):
            y_hat[sample_i] = self._sigmoid(self._theta[0] + self._theta[1] * X[sample_i,0] + self._theta[2] * X[sample_i,1])

        return y_hat

    def _loss(self, X, y):
        """Returns the sum of squares residual for the given inputs and outputs."""
        return np.sum(np.square(y - self._soft_predict(X)))

    def _fit(self, X, y):
        old_theta = self._theta.copy() # gotcha moment! np arrays point to the same memory unless copied
        old_score = self._loss(X, y)

        self._theta += np.random.normal(size=self._theta.shape)
        new_score = self._loss(X, y)

        if new_score > old_score:
            self._theta = old_theta
            return old_score

        return new_score
        
    def fit(self, X, y):
        for _ in range(self.n_max_iterations):
            loss = self._fit(X, y)
            self.loss.append(loss)

        return self # scikit-learn convention that "fit" returns itself
    
    def predict(self, X):
        """
        Hard predict, y ϵ {1, 0}
        The prediction function for logistic regression takes the data X of shape (n_samples, n_features), where n_features=2, and predicts crisp classifications as a flat vector.
        """
        return np.round(self._soft_predict(X), 0).ravel()

    def score(self, X, y): 
        """The score function should output the relative number of correct classifications as a value between 0 and 1."""
        """np.mean(y == y_hat)"""
        y_hat = self.predict(X)
        count = 0
        for i in y:
            if y[i] == y_hat[i]:
                count += 1
            
        return count / len(y) 
    
if __name__ == '__main__':
   
    # # Example data
    X = np.random.randn(10, 2)
    y = np.random.randint(0, 2, size=10)

    import matplotlib.pyplot as plt
    
    model = SimplifiedLogisticRegression()
    
    # Plot before training
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    y_pred_before = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', cmap='coolwarm', s=100, edgecolors='black', label='True labels')
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_before, marker='x', cmap='coolwarm', s=100, label='Predictions')
    plt.title('Before Training')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    # Plot after training
    plt.subplot(1, 2, 2)
    model.fit(X, y)
    y_pred_after = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', cmap='coolwarm', s=100, edgecolors='black', label='True labels')
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_after, marker='x', cmap='coolwarm', s=100, label='Predictions')
    plt.title(f'After Training (Score: {model.score(X, y):.2f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.tight_layout()
    plt.show()
