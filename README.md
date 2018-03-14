# integral
Vectorized Adaptive Quadrature in Python. This is just one function that can be used to evaluate a 1D numerical integral. It is a line-by-line translation of the quadgk function that is implemented in matlab and that was published As:  L.F. Shampine, "Vectorized Adaptive Quadrature in Matlab", Journal of Computational and Applied Mathematics 211, 2008, pp.131-140.

It is an alternative to the quad function in numpy and other implementations for integrals available in python. because it is vecorized it can be more efficient in some cases. It also is more convenient for integrating over a path through the complex plane.  
