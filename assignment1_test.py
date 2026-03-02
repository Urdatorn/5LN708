import numpy as np

regularizer_dampening = 0.001
_omega = np.zeros(X_test.shape[0])
_omega += np.random.normal(size=X_test.shape[0])

def score(X, y):
    '''Hinge loss with l2 regularization'''
    regularization_term = regularizer_dampening/2 * np.sum(np.square(_omega))
    hinge_loss = np.sum(np.maximum(0, 1 - y * (_omega @ X))) # numpy.maximum takes two input arrays a and b with the same shape and returns an array c where c_ij = max(a_ij, b_ij)
    loss = regularization_term + hinge_loss

    assert isinstance(loss, float)
    assert loss >= 0
    return loss

y_hat = np.sign(_omega @ X_test)
print(score(X_test, y_hat))