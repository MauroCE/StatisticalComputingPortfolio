gp_naive <- function(xtrain, ytrain, xtest, sigma_n, sigmasq){
  # Find kernel matrices in vectorized form
  K  <-  kernel_matrix(xtrain, xtrain, sigmasq) 
  Ks <-  kernel_matrix(xtest,  xtrain, sigmasq)
  Kss <- kernel_matrix(xtest,  xtest, sigmasq)
  # Find the inverse of K + sigma_n^2I directly
  inverse <- solve(K + sigma_n^2 * diag(ntrain))
  # GP mean
  gpmean <- Ks %*% (inverse %*% ytrain)
  # GP variance-covariance matrix
  gpvcov <- Kss - Ks %*% inverse %*% t(Ks)
  return(list(gpmean, gpvcov))
}
