""" Light Gradient Based Optimization.

    This module is dedicated to gradient based optimization schemes when
    the gradient is expensive to compute. This means that the gradient
    computation is avoided as much as possible.

    Specifically, the line search procedure do not recomputed the
    gradient during the process and looks for a point verifying the
    following sufficient decrease condition (aka Armijo condition):
    
        f(x + alpha * p) <= f(x) + c * alpha * < df(x), p >

    where f is the objective to minimize, x is the current point,
    p is the descent direction, c is a constant, df(x) is the gradient
    at point x, < ., .> represents the inner product and alpha is the
    descent step we want to determine.

    Two minimization routines are provided:
    * fmin_gd: is the steepest gradient descent algorithm.
    * fmin_lgfbs : uses l-gfbs quasi-Newton method (sparse approximation
        of the hessian matrix).

    Note: The implementation is based on the description of the
        algorithms found in:

    J. Nocedal and S. Wright. Numerical Optimization.

"""
import numpy as np

def line_search(f, x0, df0, p=None, f0=None, alpha_0=1, c=1e-4, inner=np.inner, maxiter=100, rho_lo=1e-3, rho_hi=0.9):
    """ Interpolation Line search for the steapest gradient descent.

    Finds the a step length in the descending direction -df0 verifying
    the Amijo's sufficient decrease conditions.

    Arguments:
    ----------
    * f  : the function to minimize
    * x0 : the starting point
    * df0 : the gradient value at x0
    * p : the descent direction. If None (default) -df0 is taken.
    * f0 : the function value at x0. If None (default) it is computed.
    * alpha_0 : the initial descent step (default = 1.0).
    * c : the constant used for the sufficient decrease (Armijo),
        condition:
            f(x + alpha * p) <= f(x) + c * alpha * < df(x), p>
        (default = 1e-4)
    * inner: the function used to compute the inner product. The default
       is the ordinary dot product.

    * maxiter: maximum number of iterations allowed. (default=100)
    * rho_lo : lowest ratio valued allowed between steps coefficient
               in concecutive iterations. (default=1e-3)
    * rho_hi : lowest ratio valued allowed between steps coefficient
               in concecutive iterations. (default=0.9)
               If not rho_lo <= alpha_[t+1]/alpha_[t] <= rho_hi, then
               alpha_[t+1] = 0.5 * alpha_[t] is taken.

    Returns:
    --------
    * xopt: the optimal point
    * fval: the optimal value found

    NB: Adapted for Gradient Descent from the interpolation procedure
    described in:

    J. Nocedal and S. Wright. Numerical Optimization. Chap.3 p56

    NB2: In this implementation x0 (and fd0) can be any object supporting
        addition, multiplication by a scalar and for which the inner
        product is defined (through the 'inner' function).
    """
    if f0 is None:
        f0 = f(x0)

    if p is None:
        p = -df0
    
    dphi0 = inner(df0,p)
    
    x1 = x0 + alpha_0 * p
    f1 = f(x1)
    
    if f1 <= f0 + c * alpha_0 * dphi0:
        return x1, f1

    # perfoms quadratic interpolation
    alpha_0_2 = alpha_0 * alpha_0
    alpha_1 = - dphi0* alpha_0_2 / ( 2 * (f1 - f0 - dphi0 * alpha_0) )

    x2 = x0 + alpha_1 * p

    f2 = f(x2)

    if f2 <= f0 + c * alpha_1 * dphi0:
        return x2, f2

    alpha_0_3 = alpha_0_2 * alpha_0

    iter = 0
    while True:
        # performs cubic interpolation
        alpha_1_2 = alpha_1 * alpha_1
        ff1 = f2 - f0 - dphi0 * alpha_1
        ff0 = f1 - f0 - dphi0 * alpha_0

        den = 1/(alpha_0_2 * alpha_1_2 * (alpha_1 - alpha_0))
        _3a = 3 * (alpha_0_2 * ff1 - alpha_1_2 * ff0) / den
        b = (alpha_1_2 * alpha_1 * ff0 - alpha_0_3 * ff1) / den

        alpha_2 = (-b + np.sqrt(b*b - _3a * dphi0))/ _3a

        if not  rho_lo <= alpha_2/alpha_1 <= rho_hi:
            alpha_2 = alpha_1 / 2.

        x3 = x0 + alpha_2 * p
        f3 = f(x3)

        if f3 <= f0 + c * alpha_2 * dphi0:
            return x3, f3

        iter += 1
        if iter >= maxiter:
            print "Maximum number of iteration reached before a good step size was found!"
            return x3, f3

        x1 = x2
        x2 = x3
        f1 = f2
        f2 = f3
        alpha_0 = alpha_1
        alpha_1 = alpha_2

def _print_info(iter, fval, grad_norm):
    print "iter:", iter, "fval:", fval, "|grad|:", grad_norm

def fmin_gd(f, df, x0, alpha_0=1.0, gtol=1e-6, maxiter=100,
            maxiter_line_search=100, c=1e-4, inner=np.inner,
            rho_lo=1e-3, rho_hi=0.9, 
            verbose=False, callback=None):
    """ Steepest gradient descent optimization.

    Arguments:
    ----------

    * f : the function to minimize
    * df : the function that computed the gradient.
    * x0 : the starting point.
    * alpha_0 : starting value for the descent step.
    * gtol : the value of the gradient norm under which we consider
         the optimization as converged.
    * maxiter: maximum number of iterations allowed.
    * maxiter_line_search: maximum number of iteration allowed for the
       inner line_search process.
    * c : the constant used for the sufficient decrease (Armijo)
        condition:
            f(x + alpha * p) <= f(x) + c * alpha * < df(x), p >
        (default = 1e-4)
    * inner: the function used to compute the inner product. The default
       is the ordinary dot product.
    * verbose : (boolean) If True, displays information about the
        convergence of the algorithm.
    * rho_lo : lowest ratio valued allowed between steps coefficient
               in concecutive iterations. (default=1e-3)
    * rho_hi : lowest ratio valued allowed between steps coefficient
               in concecutive iterations. (default=0.9)
               If not rho_lo <= alpha_[t+1]/alpha_[t] <= rho_hi, then
               alpha_[t+1] = 0.5 * alpha_[t] is taken.
    * callback: A function called after each iteration. The function
        is called as callback(x).

    Returns:
    --------
    * xopt: the optimal point
    * fval: the optimal value found


    """

    f0 = f(x0)
    dfx = df(x0)

    alpha_start = alpha_0
    norm_dfx = inner(dfx, dfx)

    if verbose:
        _print_info(0, f0, norm_dfx)

    if norm_dfx <= gtol:
        return x0, f0

    iter = 0
    while True:
        x1, f1 = line_search(f, x0, dfx, f0=f0, alpha_0=alpha_start,
                                c=c, inner=inner,
                                maxiter=maxiter_line_search,
                                rho_lo=rho_lo, rho_hi=rho_hi)
        if f1 >= f0:
            print "Could not minimize in the descent direction"
            return x0, f0

        if callback is not None:
            callback(x1)
        iter += 1
        if iter >= maxiter:
            print "Maximum number of iteration reached."
            return x1, f1

        dfx = df(x1)
        norm_dfx = inner(dfx, dfx)
        if verbose:
            _print_info(iter, f1, norm_dfx)
        
        if norm_dfx <= gtol:
            return x1, f1

        alpha_start = 2*(f0 - f1)/norm_dfx
        x0 = x1
        f0 = f1

def fmin_lbfgs(f, df, x0, alpha_0=1.0, m=5, gtol=1e-6, maxiter=100,
                maxiter_line_search=10, c=1e-4, inner=np.inner,
                verbose=False, rho_lo=1e-3, rho_hi=0.9, callback=None):
    """ Optimization with the Low-memory Broyden, Fletcher, Goldfarb,
    and Shanno (l-BFGS) quasi-Newton method.

    Arguments:
    ----------

    * f : the function to minimize
    * df : the function that computed the gradient.
    * x0 : the starting point.
    * alpha_0 : starting value for the descent step.
    * m : Number of points used to approximate the inverse of the
        Hessian matrix.
    * gtol : the value of the gradient norm under which we consider
         the optimization as converged.
    * maxiter: maximum number of iterations allowed.
    * maxiter_line_search: maximum number of iteration allowed for the
       inner line_search process.
    * c : the constant used for the sufficient decrease (Armijo)
        condition:
            f(x + alpha * p) <= f(x) + c * alpha * < df(x), p >
        (default = 1e-4)
    * inner: the function used to compute the inner product. The default
       is the ordinary dot product.
    * verbose : (boolean) If True, displays information about the
        convergence of the algorithm.
    * rho_lo : lowest ratio valued allowed between steps coefficient
               in concecutive iterations. (default=1e-3)
    * rho_hi : lowest ratio valued allowed between steps coefficient
               in concecutive iterations. (default=0.9)
               If not rho_lo <= alpha_[t+1]/alpha_[t] <= rho_hi, then
               alpha_[t+1] = 0.5 * alpha_[t] is taken.
    * callback: A function called after each iteration. The function
        is called as callback(x).

    Returns:
    --------
    * xopt: the optimal point
    * fval: the optimal value found

    NB: In this implementation x0 (and fd0) can be any object supporting
        addition, multiplication by a scalar and for which the inner
        product is defined (through the 'inner' function).

    Note:
        Implemented from:
        J. Nocedal and S. Wright. Numerical Optimization.
    """
    sy = []

    f0 = f(x0)
    dfx = df(x0)

    norm_dfx = inner(dfx, dfx)

    if verbose:
        _print_info(0, f0, norm_dfx)

    if norm_dfx <= gtol:
        return x0, f0

    p = -dfx

    iter = 0
    gamma = 1.0
    
    while True:
        x1, f1 = line_search(f, x0, dfx, p=p, f0=f0, alpha_0=alpha_0,
                            c=c, inner=inner, maxiter=maxiter_line_search,
                            rho_lo=rho_lo, rho_hi=rho_hi)
        if f1 >= f0:
            print "Could not minimize in the descent direction, try steapest direction"
            return x0, f0

        if callback is not None:
            callback(x1)
        iter += 1
        if iter >= maxiter:
            print "Maximum number of iteration reached."
            return x1, f1

        dfx1 = df(x1)
        norm_dfx1 = inner(dfx1, dfx1)

        if verbose:
            _print_info(iter, f1, norm_dfx1)
        
        if norm_dfx1 <= gtol:
            return x1, f1

        s = (x1-x0)
        y = (dfx1 - dfx)
        rho = 1.0/inner(y,s)
        gamma1 = inner(y,s)/inner(y,y)
        
        sy.append((y,s,rho))
        if len(s) > m:
            sy.pop(0)

        q = dfx1.copy()
        a = []
        for s,y,rho in sy[-2::-1]:
            ai = rho * inner(s,q)
            q -= ai * y
            a.insert(0,ai)

        r = gamma * q
        for (s,y,rho), ai in zip(sy[:-1],a):
            b = rho * inner(y,r)
            r += s * (ai - b)
        p = -r

        x0 = x1
        f0 = f1
        dfx = dfx1
        gamma = gamma1
