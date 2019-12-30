gp_online <- function(xtrain, ytrain, xtest, sigma_n, sigmasq){
  # Find kernel matrix
  K <- kernel_matrix(xtrain, xtrain, sigmasq) 
  # Cholesky factor for K + sigma^2 * I
  L <- chol(K + sigma_n^2*diag(ntrain))  ## Upper triangular
  alpha <- backsolve(L, forwardsolve(t(L), ytrain))
  # Allocate memory first
  gpmean <- rep(0, ntest)
  gpvar  <- rep(0, ntest)
  # Loop through all test rows of do online regression
  for (row_ix in 1:nrow(xtest)){
    # Find kernel evaluation against all training points
    kstar <- kernel_matrix(matrix(xtest[row_ix, ]), xtrain, sigmasq)
    # GP mean for current test point
    gpmean[row_ix] <- t(kstar) %*% alpha
    # GP variance for current test point
    gpvar[row_ix] <- 1.0 - crossprod(forwardsolve(t(L), kstar))  ## as SE(x*, x*) = 1.0
  }
  return(list(gpmean, gpvar))
}