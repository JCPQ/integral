import numpy as np
import warnings

def quadgk(FUN,a,b,*args,**kwargs):
    """Numerically evaluate integral, adaptive Gauss-Kronrod quadrature.
    
    This is a translation of the quadva integration from Matlab, which is 
    based on the original code published by Lawrence F. Shampine:
    Ref: L.F. Shampine, "Vectorized Adaptive Quadrature in Matlab",
    Journal of Computational and Applied Mathematics 211, 2008, pp.131-140.

    Q = quadgk(FUN,A,B,*args,**kwargs) attempts to approximate the integral of
    scalar-valued function FUN from A to B using high order global adaptive
    quadrature and default error tolerances. The function Y=FUN(X) should
    accept a vector argument X and return a vector result Y, the integrand
    evaluated at each element of X. FUN must be of type 'function'. A and B
    can be -np.Inf or np.Inf. If both are finite, they can be complex. If at
    least one is complex, the integral is approximated over a straight line
    path from A to B in the complex plane.
    
    Parameters
    ----------
    FUN : function
        Description of arg1
    a : float
        start of integral
    b : float
        end of integral

    Returns
    -------
    Q : float
        value of integral
    ERRBND : float
        ERRBND is an approximate upper bound on the
        absolute error, |Q - I|, where I denotes the exact value of the
        integral.
        
        Description of return value

    Examples
    --------
    
    note that since the code is a translation from matlab not everything
    is entirely pythonic.


    [Q,ERRBND] = QUADGK(FUN,A,B,*args,**kwargs) allows for passing the additional
    parameters that FUN(x,*args) might have via the variable length arguments.
   
    [Q,ERRBND] = QUADGK(FUN,A,B,*args,**kwargs) allows for passing specific
    keyworded arguments via kwargs. the available options are:
       AbsTol the absolute error tolerance (f.i. AbsTol=10e-3)
       RelTol the relative error tolerance (f.i. RelTol=10e-4)
       MaxIntervalCount (f.i. MaxIntervalCount=1000)
       Waypoints vector of integration waypoints (f.i. np.array([1+j,j])) NOT TESTED!

       QUADGK attempts to satisfy ERRBND <= max(AbsTol,RelTol*|Q|). This
       is absolute error control when |Q| is sufficiently small and
       relative error control when |Q| is larger. A default tolerance
       value is used when a tolerance is not specified. The default value
       of 'AbsTol' is 1.e-10 (double). The default value
       of 'RelTol' is 1.e-6 (double). For pure absolute
       error control use
         Q = quadgk(FUN,A,B,AbsTol=ATOL,RelTol=0) NOT TESTED!
       where ATOL > 0. For pure relative error control use
         Q = quadgk(FUN,A,B,RelTol=RTOL,AbsTol=0) NOT TESTED!
       Except when using pure absolute error control, the minimum relative
       tolerance is 100*spacing(1).

   Waypoints, vector of integration waypoints

       If FUN(X) has discontinuities in the interval of integration, the
       locations should be supplied as a 'Waypoints' vector. When A, B,
       and the waypoints are all real, only the waypoints between A and B
       are used, and they are used in sorted order.  Note that waypoints
       are not intended for singularities in FUN(X). Singular points
       should be handled by making them endpoints of separate integrations
       and adding the results.

       If A, B, or any entry of the waypoints vector is complex, the
       integration is performed over a sequence of straight line paths in
       the complex plane, from A to the first waypoint, from the first
       waypoint to the second, and so forth, and finally from the last
       waypoint to B.

   MaxIntervalCount, maximum number of intervals allowed

       The 'MaxIntervalCount' parameter limits the number of intervals
       that QUADGK will use at any one time after the first iteration. A
       warning is issued if QUADGK returns early because of this limit.
       The default value is 650. Increasing this value is not recommended,
       but it may be appropriate when ERRBND is small enough that the
       desired accuracy has nearly been achieved.

   Notes:
   QUADGK may be most efficient for oscillatory integrands and any smooth
   integrand at high accuracies. It supports infinite intervals and can
   handle moderate singularities at the endpoints. It also supports
   contour integration along piecewise linear paths.
   
   Example:
   Integrate f(x) = exp(-x^2)*log(x)^2 from 0 to infinity:
      def f(x):
          return exp(-x.^2).*log(x).^2
      Q = quadgk(f,0,Inf)

   Example:
   To use a parameter in the integrand:
      def f(x,ac):
          return 1./(x.^3-2*x-c)
      c=3
      Q = quadgk(f,0,2,c);

   Example:
   Integrate f(z) = 1/(2z-1) in the complex plane over the triangular
   path from 0 to 1+1i to 1-1i to 0:
      def fnuc(z):
          return 1/(2*z-1)
      Q = quadgk(fnuc,0,0,Waypoints=np.array([1+1i,1-1i]))

   Class support for inputs A, B, and the output of FUN : float


   Based on "quadva" by Lawrence F. Shampine.
   Ref: L.F. Shampine, "Vectorized Adaptive Quadrature in Matlab",
   Journal of Computational and Applied Mathematics 211, 2008, pp.131-140.

    """

    global FIRSTFUNEVAL,RTOL,ATOL    
    # Args refers to the other input parameters Fun might have
    # while ** kwargs are all keyword arguments that are used
    # as parameters in the integral
    
    # Variable names in all caps are referenced in nested functions.

    # Validate the first three inputs.
    #error(nargchk(3,inf,nargin,'struct'));
    if type(FUN) != type(lambda :None):
        raise TypeError('First input argument must be a function type.')
    # validdate a as a float or an int (and then covert it to float)
    if isinstance(a,int):
        a=a*1.0
    elif not isinstance(a,float):
        raise TypeError("Start value integration range (a) is not numeric")
    # validdate b as a float or an int (and then covert it to float)
    if isinstance(b,int):
        b=b*1.0
    elif not isinstance(b,float):
        raise TypeError("End value integration range (b) is not numeric")

    # Default values for keyworded inputs
    AbsTol=1e-10
    RelTol=1e-6
    MaxIntervalCount=650
    Waypoints=np.array([])
    # Pick up the keywords, and check their validity, might not catch all exceptions here
    for key in kwargs:
        if key=='AbsTol':
            if isinstance(kwargs[key],float) and kwargs[key]>=0 :
                AbsTol=kwargs[key]
            else:
                raise TypeError("input type AbsTol should be positive real float")
        elif key=='RelTol':
            if isinstance(kwargs[key],float) and kwargs[key]>=0 :
                RelTol=kwargs[key]
            else:
                raise TypeError("input type RelTol should be positive real float")
        elif key=='MaxIntervalCount':
            if isinstance(kwargs[key],int) and kwargs[key]>0 :
                MaxIntervalCount=kwargs[key]
            else:
                raise TypeError("input type MaxIntervalCount should be int > 0")
        elif key=='Waypoints':
            if isinstance(kwargs[key],np.ndarray) and np.all(np.isfinite(kwargs[key])):
                Waypoints=kwargs[key]
            else:
                raise TypeError("input type Waypoints should be ndarray with finite values")
        else:
            # unknow keyword used
            raise ValueError("This is not a valid keyworded option input"
            "please use AbsTol, RelTol, MaxIntervalCount or Waypoints" )
            
    # this is a historical statement, might be made simpler/ more direct
    ATOL=AbsTol
    RTOL=RelTol
    MAXINTERVALCOUNT=MaxIntervalCount
    WAYPOINTS=Waypoints
    
    # Fixed parameters. might be partially obsolete
    MININTERVALCOUNT = 10 # Minimum number subintervals to start.
     
    # usefull abreviations    
    zero=np.array([0])
    one=np.array([1])
        
    # Gauss-Kronrod (7,15) pair. Use symmetry in defining nodes and weights.
    pnodes = np.array([ 0.2077849550078985, 0.4058451513773972, 0.5860872354676911,
        0.7415311855993944, 0.8648644233597691, 0.9491079123427585,
        0.9914553711208126])
    pwt = np.array([ 0.2044329400752989, 0.1903505780647854, 0.1690047266392679,
        0.1406532597155259, 0.1047900103222502, 0.06309209262997855,
        0.02293532201052922])
    pwt7 = np.array([0,0.3818300505051189,0,0.2797053914892767,0,0.1294849661688697,0])
    NODES = np.concatenate([-pnodes[::-1],zero,pnodes])
    WT = np.concatenate([pwt[::-1], np.array([0.2094821410847278]), pwt])
    EWT = WT - np.concatenate([pwt7[::-1], np.array([0.4179591836734694]), pwt7])
    
    # Initialize the FIRSTFUNEVAL variable.  Some checks will be done just
    # after the first evaluation.
    FIRSTFUNEVAL = True

    #--------Nested functions--------------------------------------    
    def midpArea(f,a,b):
        # Return q = (b-a)*f((a+b)/2). Although formally correct as a low
        # order quadrature formula, this function is only used to return
        # nan or zero of the appropriate class when a == b, isnan(a), or
        # isnan(b). \\ this looks like a test for finiteness of results...
        x = (a+b)/2
        if np.isfinite(a) and np.isfinite(b) and  not np.isfinite(x):
            # Treat overflow, e.g. when finite a and b > realmax/2
            x = a/2 + b/2 
        fx = f(x)
        if not np.isfinite(fx):
            warnings.warn('quadgk:midpArea ....Infinite or Not-a-Number value encountered.');
        q = (b-a)*fx
        return q

    def evalFun(x):
        # Evaluate the integrand, here the *args are used.
        global FIRSTFUNEVAL,RTOL,ATOL
        if FIRSTFUNEVAL:
            # Don't check the closeness of the mesh on the first iteration.
            too_close = False
            fx = FUN(x,*args)
            RTOL=finalInputChecks(x,fx,RTOL,ATOL)
            FIRSTFUNEVAL = False
        else:
            too_close = checkSpacing(x)
            if too_close:
                fx = np.array([])
            else:
                fx = FUN(x,*args)
        return fx,too_close

    def f1(t):
        # Transform to weaken singularities at both ends: [a,b] -> [-1,1]
        tt = 0.25*(B-A)*t*(3 - t**2) + 0.5*(B+A)
        y,too_close = evalFun(tt)
        if not too_close:
            y = 0.75*(B-A)*y*(1 - t**2)
        return y,too_close

    def f2(t):
        # Transform to weaken singularity at left end: [a,Inf) -> [0,Inf).
        # Then transform to finite interval: [0,Inf) -> [0,1].
        tt = t / (1 - t)
        t2t = A + tt**2
        y,too_close = evalFun(t2t)
        if not too_close:
            y =  2*tt * y / (1 - t)**2
        return y,too_close

    def f3(t):
        # Transform to weaken singularity at right end: (-Inf,b] -> (-Inf,b].
        # Then transform to finite interval: (-Inf,b] -> (-1,0].
        tt = t / (1 + t)
        t2t = B - tt**2
        y,too_close = evalFun(t2t)
        if  not too_close:
            y = -2*tt * y / (1 + t)**2;
        return y,too_close

    def f4(t):
        # Transform to finite interval: (-Inf,Inf) -> (-1,1).
        tt = t / (1 - t**2)
        y,too_close = evalFun(tt);
        if not too_close:
            y = y * (1 + t**2) / (1 - t**2)**2;
        return y,too_close

    def checkSpacing(x):
        ax = np.abs(x)
        tcidx = np.where(np.abs(np.diff(x))<= 100*np.spacing(1)*np.maximum(ax[0:-1],ax[1:])) 
        # find the indices of x where the step to the next x is smaller then 100 times the 
        # smallest number available, maybe this can be polished using:
        # spacing(np.maximum(ax[0:-1],ax[1:]))
        too_close = np.bool(np.sum(tcidx));
        if too_close:
            warnings.warn('quadgk:CheckSpacing'
                'Minimum step size reached near x[tcidx] singularity possible.')
        return too_close

    def split(x,minsubs):
        # Split subintervals in the interval vector X so that, to working
        # precision, no subinterval is longer than 1/MINSUBS times the
        # total path length. Removes subintervals of zero length, except
        # that the resulting X will always has at least two elements on
        # return, i.e., if the total path length is zero, X will be
        # collapsed into a single interval of zero length.  Also returns
        # the integration path length.
        # this was not an easy strecht to translate might contain an error
        
        absdx = np.abs(np.diff(x))
        if np.all(np.isreal(x)):
            pathlen = x[-1] - x[0]
        else:
            pathlen = np.sum(absdx)
        if pathlen > 0:
            udelta = minsubs/pathlen
            nnew = np.ceil(absdx*udelta) - 1
            idxnew = np.where(nnew > 0)
            nnew = nnew[idxnew]
            for jj in reversed(range(len(idxnew[0]))): #np.len(idxnew):-1:1 translation should be oke j now to 0
                k = idxnew[0][jj]
                nnj = nnew[jj]
                # Calculate new points.
                newpts = x[k] + np.linspace(1,nnj,nnj)/(nnj+1)*(x[k+1]-x[k]) #' so how does this go with indices?
                # Insert the new points.
                x = np.concatenate([x[0:k+1],newpts,x[k+1:]]) # uncertain about index k here                
        # Remove useless subintervals.
        x=x[np.r_[1,np.abs(np.diff(x))]!=0]
        if len(x)==1:
            # Return at least two elements.
            x = np.array([x[0],x[0]])
        return x,pathlen

    def finalInputChecks(x,fx,RTOL,ATOL):
        # Do final input validation with sample input and outputs to the
        # integrand function.
        # Check classes.
        if not (isinstance(x,np.ndarray) and isinstance(fx,np.ndarray)):
            raise ValueError('quadgk:finalInputChecks, UnsupportedClass'
                'Supported classes are ''double'' and ''single''.')
        # Check sizes.
        if not np.all(np.shape(x)==np.shape(fx)):
            raise ValueError('quadgk:finalInputChecks'
                'Output of the function must be the same size as the input.')
        # Make sure that RTOL >= 100*eps(outcls) except when
        # using pure absolute error control (ATOL>0 && RTOL==0).
        if RTOL < 100*np.spacing(1) and not (ATOL > 0 and RTOL == 0):
            RTOL = 100*np.spacing(1);
            warnings.warn('quadgk:finalInputChecks, increasedRelTol'
                'RelTol was increased to 100*spacing.') 
                #'RelTol was increased to 100*eps(''%s'') = %g.',outcls,RTOL
        return RTOL# function has no output

    def vadapt(f,tinterval):
        # Iterative routine to perform the integration.
        # Compute the path length and split tinterval if needed.
        tinterval,pathlen = split(tinterval,MININTERVALCOUNT)      
        if pathlen == 0:
            # Test case: quadgk(@(x)x,1+1i,1+1i);
            q = midpArea(f,tinterval[0],tinterval[-1])
            errbnd = q
            return q,errbnd
        # Initialize array of subintervals of [a,b].
        subs = np.array([tinterval[0:-1],tinterval[1:]]) # create n by two array  for sub intervals [tinterval(1:end-1);tinterval(2:end)]; ' weird statement
        # Initialize partial sums.
        q_ok = 0
        err_ok = 0
        # Initialize main loop
        while True:
            # SUBS contains subintervals of [a,b] where the integral is not
            # sufficiently accurate. The first row of SUBS holds the left end
            # points and the second row, the corresponding right endpoints.
            midpt = np.sum(subs,0)/2   # midpoints of the subintervals
            halfh = (subs[1,:]-subs[0,:])/2  # half the lengths of the subintervals
            x = NODES[:,np.newaxis]*halfh+midpt #bsxfun(@plus,NODES*halfh,midpt); 'mysterious plus function?'
            x = np.reshape(x,-1)   # function f expects a row vector
            fx,too_close = f(x)
            # Quit if mesh points are too close.
            if too_close:
                break
            fx = np.reshape(fx,(len(WT),-1)) #reshape(fx,numel(WT),[]); '???
            # Quantities for subintervals.
            qsubs = np.dot(WT,fx) * halfh
            errsubs = np.dot(EWT,fx) * halfh
            # Calculate current values of q and tol.
            q = np.sum(qsubs) + q_ok
            tol = max(ATOL,RTOL*np.abs(q))
            # Locate subintervals where the approximate integrals are
            # sufficiently accurate and use them to update the partial
            # error sum.
            ndx = np.where(np.abs(errsubs) <= (2*tol/pathlen)*halfh)
            err_ok = err_ok + np.sum(errsubs[ndx]);
            # Remove errsubs entries for subintervals with accurate
            # approximations.
            errsubs=np.delete(errsubs,ndx)
            # The approximate error bound is constructed by adding the
            # approximate error bounds for the subintervals with accurate
            # approximations to the 1-norm of the approximate error bounds
            # for the remaining subintervals.  This guards against
            # excessive cancellation of the errors of the remaining
            # subintervals.
            errbnd = np.abs(err_ok) + np.linalg.norm(errsubs,1)# no clue ???
            # Check for nonfinites.
            if not (np.isfinite(q) and np.isfinite(errbnd)):
                warnings.warn('quadgk:NonFiniteValue ... Infinite or Not-a-Number value encountered.');
                break
            # Test for convergence.
            if errbnd <= tol:
                break
            # Remove subintervals with accurate approximations.
            subs=np.delete(subs,ndx,1) #subs(:,ndx) = []; '???
            if subs.size==0:
                break
            # Update the partial sum for the integral.
            q_ok = q_ok + np.sum(qsubs[ndx]);
            # Split the remaining subintervals in half. Quit if splitting
            # results in too many subintervals.
            nsubs = 2*subs.shape[1]
            if nsubs > MAXINTERVALCOUNT:
                warnings.warn('MaxIntervalCountReached'
                    'Reached the limit on the maximum number of intervals in use.\n'
                    'Approximate bound on error is%9.1e. The integral may not exist, or\n'
                    'it may be difficult to approximate numerically. Increase MaxIntervalCount\n'
                    'to %d to enable QUADGK to continue for another iteration.')#], ... errbnd,nsubs)
                break
            midpt=np.delete(midpt,ndx)#midpt(ndx) = []; '??# Remove unneeded midpoints.
            subs = np.reshape(np.array([subs[0,:], midpt, midpt, subs[1,:]]),(2,-1),order='F')
        return q,errbnd
        
    # Handle contour integration, this acts if a,b or any of the waypoints is complex 
    if not (np.isreal(a) and np.isreal(b) and np.all(np.isreal(WAYPOINTS))):
        tinterval = np.concatenate([np.array([a]),WAYPOINTS,np.array([b])])
        if np.any(np.isinf(tinterval)):
            raise ValueError('quadgk:nonFiniteContourError'
                             'Contour endpoints and waypoints must be finite.')
        # A and B should not be needed, so we do not define them here.
        # Perform the contour integration.
        q,errbnd = vadapt(evalFun,tinterval)
        return q,errbnd   # first instance to escape from function with succes
    
    # Define A and B and note the direction of integration on real axis.
    if b < a:
        # Integrate left to right and change sign at the end.
        reversedir = True
        A = b
        B = a
    else:
        reversedir = False
        A = a
        B = b
    
    # Trim and sort the waypoints vector. The use of waypoints is to deal with
    # discontinuities in the function
    WAYPOINTS = WAYPOINTS[WAYPOINTS>A and WAYPOINTS<B]
    WAYPOINTS.sort
    
    # Construct interval vector with waypoints.
    interval = np.concatenate([np.array([A]), WAYPOINTS, np.array([B])])
    
    # Extract A and B from interval vector to regularize possible mixed
    # single/double inputs. //no clue what this means
    A = interval[0]
    B = interval[-1]
    # Identify the task and perform the integration. Not unlikely that translatio went wrong here
    if A == B:
        # Handles both finite and infinite cases.
        # Return zero or nan of the appropriate class.
        q = midpArea(evalFun,A,B)
        errbnd = q
    elif np.isfinite(A) and np.isfinite(B):
        if len(interval) > 2:
            # Analytical transformation suggested by K.L. Metlov:
            alpha = 2*np.sin( np.arcsin((A + B - 2*interval[1:-1])/(A - B))/3 )
            interval = np.concatenate([np.array([-1]),alpha,np.array([1])])
        else:
            interval = np.array([-1,1])
        q,errbnd = vadapt(f1,interval)
    elif np.isfinite(A) and np.isinf(B):
        if len(interval) > 2:
            alpha = np.sqrt(interval[1:-1] - A)
            interval = np.concatenate([zero,alpha/(1+alpha),one])
        else:
            interval = np.array([0,1])
        q,errbnd = vadapt(f2,interval)
    elif np.isinf(A) and np.isfinite(B):
        if len(interval) > 2:
            alpha = np.sqrt(B - interval[1:-1])
            interval = np.concatenate([-one,-alpha/(1+alpha),zero])
        else:
            interval = np.array([-1,0])
        q,errbnd = vadapt(f3,interval)
    elif np.isinf(A) and np.isinf(B):
        if len(interval) > 2:
            # Analytical transformation suggested by K.L. Metlov:
            alpha = np.tanh( np.arcsinh(2*interval[1:-1]/2 ))
            interval = np.concatenate([-one,alpha,one])
        else:
            interval = np.array([-1,1])
        q,errbnd = vadapt(f4,interval)
    else: # i.e., if isnan(a) || isnan(b)
        q = midpArea(evalFun,A,B)
        errbnd = q
    
    # Account for integration direction.
    if reversedir:
        q = -q;

    return q,errbnd # finally the output of quadgk

# ------end quadgk------------------- 

# test
#
#def fnuc1(z):
#    return 1/(2*z-1)
#
#Q = quadgk(fnuc1,0,0,Waypoints=np.array([1+1j,1-1j]))
#print(Q[0])
