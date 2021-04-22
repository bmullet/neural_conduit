import numpy as np

def mcmc_engine(func,loglikelihood,data,x0,xstep,xbnds,sigma,Niter,*args):
  """subroutine for MCMC sampling using Metropolis-Hasting w/ normal
  distribution.
   Inputs:
        func = function name eg:   mcmc(@travel_time,....)
                 parameter required for function in varargin
        loglikelihood = function for computing log(likelihood)
        data = vector of observations
        x0   = initial estimate of parameter vector
        xstep = step size in all parameter directions
        xbnds = bounds (vector of upper and lower bounds)
        sigma = sigma of normal distribution
        Niter = number of iterations
   Outputs:
        x_keep = array of samples
        L_keep = likelihood of samples
        count  = number of accepted. Acceptance ratio is count/Niter
  """

  #  Initialize
  x_keep = []
  L_keep = []
  N = max(x0.shape)

  # First model is just initial first guess
  x_keep.append(x0)
  L_keep.append(loglikelihood(func,x0,data,sigma,args))

  xi = x0
  count = 0

  for i in range(1,Niter):
    lxi = L_keep[i-1]

    # Generate proposed model
    xp = xi + xstep*np.random.randn(N)

    # check bounds
    if (np.all(xp > xbnds[:,0])) & (np.all(xp < xbnds[:,1])):

      # Calculate liklelihoods 
      lxp = loglikelihood(func,xp,data,sigma,args)

      # Decide to keep or reject proposed model 
      alpha = min(np.float(0),lxp-lxi)

      t = np.log(np.random.rand(1))

      if alpha > t:
        count = count + 1
        xi = xp # Keep proposed model, otherwise xi stays the same
        lxi = lxp              

    # Store accepted model
    x_keep.append(xi)
    L_keep.append(lxi)

  # change likelihoods back from log
  L_keep = np.exp(np.array(L_keep))
  x_keep = np.array(x_keep)

  return x_keep, L_keep, count