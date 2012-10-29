import numpy as np
from scipy.linalg import inv
from lgbopt import fmin_gd, fmin_lbfgs

class Objective(object):
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c

    def __call__(self, x):
        return 0.5 * np.inner(x, np.dot(A,x)) + np.inner(b,x) + c

    def grad(self, x):
        return np.dot(A,x) + b

if __name__ == '__main__':
    
    np.random.seed(8714)
    A = np.random.randn(2,2)
    A = np.dot(A,A.T)
    b = np.random.randn(2)
    c = np.random.randn(1)[0]

    obj = Objective(A, b , c)

    x0 = np.random.randn(2)
    
    xopt, fopt = fmin_gd(obj, obj.grad, x0, c=1e-4, verbose=True)
    print xopt, fopt
    xtrue = -np.dot(inv(A),b)
    print xtrue, obj(xtrue)
    print
    print "testing l-bfgs"
    xopt, fopt = fmin_lbfgs(obj, obj.grad, x0, c=1e-4, verbose=True, gtol=1e-5, rho_lo=1e-3, m=5)
    grad = obj.grad(xopt)
    print xopt, fopt, np.inner(grad, grad)
    print xtrue, obj(xtrue)
    
