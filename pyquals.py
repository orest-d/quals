import numpy as np


class LeastSquares:
    def __init__(self):
        self.a = None

    def fit(self, X, y):
        n = X.shape[1]
        A = np.zeros((n, n), float)
        b = np.zeros(n, float)
        for i in range(n):
            b[i] = np.dot(X[:, i], y)
            for j in range(n):
                A[i, j] = np.dot(X[:, i], X[:, j])
        self.a = np.dot(np.linalg.inv(A), b)

    def predict(self, X):
        return np.dot(X, self.a)


class CubicQuantile1D:
    def __init__(self, gamma=0.0):
        self.a = None
        self.gamma = gamma

    def fit(self, x, y):
        gamma = self.gamma
        A = 3*gamma*np.sum(x*x*x)
        B = 2*np.sum(x*x*(1.0-3*gamma*y))
        C = np.sum(x*y*(3.0*gamma*y-2.0))
        print(f"A={A} B={B} C={C}")
        if np.abs(A) < 1e-6:
            self.a = -C/B
        else:
            self.a = (-B+np.sqrt(B*B-4.0*A*C))/(2*A)

    def predict(self, x):
        return self.a*x


class ExpQuantile:
    def __init__(self, gamma=0.0):
        self.a = None
        self.gamma = gamma

    def fit_ls(self, X, y):
        n = X.shape[1]
        A = np.zeros((n, n), float)
        b = np.zeros(n, float)
        for i in range(n):
            b[i] = np.dot(X[:, i], y)
            for j in range(n):
                A[i, j] = np.dot(X[:, i], X[:, j])
        self.a = np.dot(np.linalg.inv(A), b)
        return self.a

    def gradient(self, X, y, a=None):
        if a is None:
            a = self.a
        gamma = self.gamma
        epsilon = np.dot(X, a)-y
        gs = (2.0 + gamma*epsilon)*epsilon*np.exp(gamma*epsilon)
        return np.dot(gs, X)

    def gradient_descent_step(self, X, y, step=1e-2):
        g = self.gradient(X, y)
        gn = np.sqrt(np.sum(g*g))
        self.a -= step*g/gn

    def fit(self, X, y):
        if self.a is None:
            self.fit_ls(X, y)
        for i in range(10000):
            self.gradient_descent_step(X, y)

    def predict(self, X):
        return np.dot(X, self.a)


def ls_test():
    model = LeastSquares()
    N = 100
    X = []
    y = []
    for i in range(N):
        x1 = np.random.uniform(0, 100)
        x2 = np.random.uniform(0, 100)
        yy = x1+x2 + 0.1*np.random.uniform(0, 100)
        X.append([x1, x2])
        y.append(yy)
    X = np.array(X)
    y = np.array(y)
    model.fit(X, y)
    print(model.a)


def cubic1d_test():
    N = 200
    x = np.random.uniform(0, 100, N)
    y = x+np.random.uniform(0, 30, N)
    plt.scatter(x, y)

    model = LeastSquares()
    X = np.zeros((len(x), 1), float)
    X[:, 0] = x
    model.fit(X, y)
    x_mesh = np.linspace(0, 100, 50)
    X_mesh = np.zeros((len(x_mesh), 1), float)
    X_mesh[:, 0] = x_mesh
    y_mesh = model.predict(X_mesh)
    plt.plot(x_mesh, y_mesh, label=f"LS a={model.a}")

    x_mesh = np.linspace(0, 100, 50)
    for gamma in (-0.027, -0.025, 0, 0.025, 0.035):
        model = CubicQuantile1D(gamma)
        model.fit(x, y)
        print(f"gamma={gamma} a={model.a}")
        y_mesh = model.predict(x_mesh)
        plt.plot(x_mesh, y_mesh, label=f"gamma={gamma} a={model.a}")
    plt.legend()
    plt.show()


def exp2d_test():
    N = 1000
    x = np.random.uniform(0, 100, N)
    y = x+np.random.uniform(0, 30, N)
    plt.scatter(x, y)
    X = np.zeros((len(x), 2), float)
    X[:, 0] = x
    X[:, 1] = 1

    model = LeastSquares()
    model.fit(X, y)

    x_mesh = np.linspace(0, 100, 50)
    X_mesh = np.zeros((len(x_mesh), 2), float)
    X_mesh[:, 0] = x_mesh
    X_mesh[:, 1] = 1
    y_mesh = model.predict(X_mesh)
    plt.plot(x_mesh, y_mesh, label=f"LS a={model.a}")

#    for gamma in (-0.1,-0.05, -0.025, 0.0, 0.025, 0.05, 0.1):
    for gamma in (-0.08, 0.0, 0.08):
        model = ExpQuantile(gamma)
        model.fit(X, y)
        print(f"gamma={gamma} a={model.a}")
        y_mesh = model.predict(X_mesh)
        plt.plot(x_mesh, y_mesh, label=f"gamma={gamma} a={model.a}")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    exp2d_test()
