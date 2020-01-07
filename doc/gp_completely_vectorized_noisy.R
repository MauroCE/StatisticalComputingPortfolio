gp_completely_vectorized_noisy <- function(xtrain, ytrain, xtest, sigma_n, sigmasq){
  # Find kernel matrices in vectorized form
  K  <-  kernel_matrix_vectorized(xtrain, sigmasq) 
  Ks <-  kernel_matrix_vectorized(xtest,  sigmasq, xtrain)
  Kss <- kernel_matrix_vectorized(xtest, sigmasq)
  # Cholesky factorization
  L <- chol(K + sigma_n^2*diag(ntrain))  ## Upper triangular
  alpha <- backsolve(L, forwardsolve(t(L), ytrain))
  # Solve by forward and backward substitution
  gpmean <- Ks %*% alpha
  gpvcov <- Kss - Ks %*% backsolve(L, forwardsolve(t(L), t(Ks))) + sigma_n^2*diag(nrow(xtest))
  return(list(gpmean, gpvcov))
}
