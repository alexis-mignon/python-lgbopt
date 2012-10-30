python-lgbopt
=============

Light Gradient Based Optimization.

Author: Alexis Mignon (c) Oct. 2012
E-mail: alexis.mignon@gmail.com


This module provides routines for gradient based optimization.
It focuses on the case where gradient computation is expensive
and should be avoided as much as possible.

To achieve this, the inner line search procedure uses only one
gradient estimation. The line search is then performed using
quadratic and cubic interpolation as described in:

J. Nocedal and S. Wright. Numerical Optimization. Chap.3 p56

Two optimization schemes are provided:
- steepest gradient descent (since sometimes it's still the
   most practicle way to do it).
- low-memory GFBS Quasi-Newton method.

Why writing optimization code while there exists optimized 
packages to do so ?

Because the problems I had to deal with had the following
properties:
* the gradient computation is expensive (don't even think of
  computing the Hessian),
* the domains on which I have to optimize may not be Euclidean.

The only  assumptions made on the optimization variables (and
gradient values) is that they support some basic operations
in normed vector spaces:
- addition,
- multiplication by a scalar,
- inner product (a custom inner product function can be provided).


<script type="text/javascript">

  var _gaq = _gaq || [];
  _gaq.push(['_setAccount', 'UA-35983614-1']);
  _gaq.push(['_trackPageview']);

  (function() {
    var ga = document.createElement('script'); ga.type = 'text/javascript'; ga.async = true;
    ga.src = ('https:' == document.location.protocol ? 'https://ssl' : 'http://www') + '.google-analytics.com/ga.js';
    var s = document.getElementsByTagName('script')[0]; s.parentNode.insertBefore(ga, s);
  })();

</script>

