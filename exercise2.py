import numpy as np

class LinearRegression: 
    def __init__(self, n_order, n_max_iter=1000):
        """Linear regression using a first order polynomial model for 2D data."""
        self.n_order = n_order
        self.n_max_iter = n_max_iter
        self.loss_ = []

        self._theta = np.zeros(n_order + 1)

    def predict(self, X):
        y_hat = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            for m in range(0, self.n_order + 1):
                y_hat[i] += self._theta[m] * (X[i,0] ** m) # zero because there's one column of data (not necessary, but clearer)
        return y_hat

    def score(self, X, y): 
        """Returns the sum of squares residual for the given inputs and outputs."""
        return np.sum(np.square(y - self.predict(X)))

    def _fit(self, X, y):
        old_theta = self._theta.copy() # gotcha moment! np arrays point to the same memory unless copied
        old_score = self.score(X, y)

        self._theta += np.random.normal(size=self._theta.shape)
        new_score = self.score(X, y)

        if new_score > old_score:
            self._theta = old_theta
            return old_score

        print(new_score)
        return new_score
        
    def fit(self, X, y):
        for _ in range(self.n_max_iter):
            loss = self._fit(X, y)
            self.loss_.append(loss)

        return self # scikit-learn convention that "fit" returns itself

if __name__ == '__main__':
   
    # # Example data
    X = np.vstack(np.linspace(0, 10, num=10))
    y = X.ravel() ** 2 + X.ravel() * 3 + 4 * np.random.normal(0, 1, size=len(X))
    z = X.ravel() ** 2 + X.ravel() * 3

    import matplotlib.pyplot as plt
    # from matplotlib.animation import FuncAnimation

    # Plot setup
    #fig, ax = plt.subplots(figsize=(6, 4))
    #ax.plot(X.ravel(), y, 'o', label="Data")
    # ax.plot(X.ravel(), z, '-', label="Optimal model y = x**2 + 3x")

    # model = LinearRegression(2, 10000)

    # # Initial model line
    # line, = ax.plot(X.ravel(), model.predict(X), '-', label="Model (iterating)")
    # ax.legend()

    # # ---- ANIMATION UPDATE FUNCTION ----
    # def update(frame):
    #     model._fit(X, y)                 # ONE iteration
    #     line.set_ydata(model.predict(X)) # Update curve
    #     ax.set_title(f"Iteration {frame}")
    #     return line,

    # ani = FuncAnimation(
    #     fig,
    #     update,
    #     frames=model.n_max_iter,
    #     interval=20,     # ms between frames
    #     blit=True
    # )

    # plt.show()
    
    # Plot before training
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(X.ravel(), y, 'o', label="Data")
    model = LinearRegression(2, 10000)
    plt.plot(X.ravel(), model.predict(X), '-', label="Untrained model")
    plt.plot(X.ravel(), z, '-', label="Optimal model y = x**2 + 3x")
    plt.legend()

    # # Plot after training
    plt.subplot(1, 2, 2)
    plt.plot(X.ravel(), y, 'o', label="Data")
    model.fit(X, y)
    plt.plot(X.ravel(), model.predict(X), '-', label="Trained model")
    plt.plot(X.ravel(), z, '-', label="Optimal model y = x**2 + 3x")
    plt.legend()
    plt.show()
