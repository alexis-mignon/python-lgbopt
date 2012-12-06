import numpy as np
from scipy.linalg import inv
from lgbopt import fmin_gd, fmin_lbfgs, fmin_cg

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
    from time import time

    def timer(func):
        """ Decorator for timing a function.
        """
        def inner(*args, **kwargs):
            t0 = time()
            r = func(*args, **kwargs)
            t1 = time()
            print "function %s ran in %0.3f ms"%(func.__name__, ((t1-t0)*1000))
            return r
        return inner

    #creating objective function
    np.random.seed(8714)
    A = np.random.randn(2,2)
    A = np.dot(A,A.T)
    b = np.random.randn(2)
    c = np.random.randn(1)[0]
    obj = Objective(A, b , c)

    # starting point
    x0 = np.random.randn(2)

    # computing the true optimum
    print "expected result"
    xtrue = -np.dot(inv(A),b)
    print xtrue, obj(xtrue)

    # testing the different algorithms
    print "testing steapest descent"
    xopt, fopt = timer(fmin_gd)(obj, obj.grad, x0, verbose=True)
    print "found    x:", xopt, "; fval:", fopt
    print "expected x:", xtrue, "; fval:", obj(xtrue)
    print 
    print "testing l-bfgs"
    xopt, fopt = timer(fmin_lbfgs)(obj, obj.grad, x0, verbose=True)
    print "found    x:", xopt, "; fval:", fopt
    print "expected x:", xtrue, "; fval:", obj(xtrue)
    print 
    print "testing conjugate gradient"
    xopt, fopt = timer(fmin_cg)(obj, obj.grad, x0, verbose=True)
    grad = obj.grad(xopt)
    print "found    x:", xopt, "; fval:", fopt
    print "expected x:", xtrue, "; fval:", obj(xtrue)
    print 
