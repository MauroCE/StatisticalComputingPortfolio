# uses vectorized kernel matrix
gp_online_vect <- function(xtrain, ytrain, xtest, sigma_n, sigmasq){
  # Find kernel matrix
  K <- kernel_matrix_vectorized(xtrain, sigmasq) 
  # Cholesky factor for K + sigma^2 * I
  L <- chol(K + sigma_n^2*diag(ntrain))  ## Upper triangular
  alpha <- backsolve(L, forwardsolve(t(L), ytrain))
  # Allocate memory first
  gpmean <- rep(0, ntest)
  gpvar  <- rep(0, ntest)
  # Loop through all test rows of do online regression
  for (row_ix in 1:nrow(xtest)){
    # Find kernel evaluation against all training points
    kstar <- kernel_matrix_vectorized(matrix(xtest[row_ix, ]), sigmasq, xtrain)
    # GP mean for current test point
    gpmean[row_ix] <- kstar %*% alpha
    # GP variance for current test point
    gpvar[row_ix] <- 1.0 - crossprod(forwardsolve(t(L), t(kstar)))  ## as SE(x*, x*) = 1.0
  }
  return(list(gpmean, gpvar))
}