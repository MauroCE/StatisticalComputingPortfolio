# Get data from the weblink
library(readr)
url <- "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data"
df <- read.table(url)[1:9]

# Recall X must have a feature in each row and an observation in each column
X <- t(as.matrix(df[1:8]))  # (d+1) * n
# We add an extra row of 1s for the bias
X <- rbind(X, rep(1, dim(X)[2]))
y <- t(as.vector(df$lpsa)) # 1 * n

least_squares <- function(X, y){
  # X must have dim (d+1)*n where d = number of features, and n = number of observations
  # y must be a row vector of dim 1*n
  A <- X%*%t(X)
  b <- X%*%t(y)
  return(solve(A, b))
}

linear_ls <- function(x, w){
  # This function simply implements the linear version of f(x, w) = w^t * x
  # w will be returned as a column vector from least_squares().
  x <- as.matrix(x)
  w <- as.matrix(w)
  # Want this function to do both matrix multiplication and dot product,
  # so check for dimension and if necessary transpose
  if ((dim(x)[2] == 1) && (dim(x)[2] != dim(w)[1])) {
    x <- t(x)
  }
  return(x%*%w)
}

# for testing purposes let's set a seed
cv <- function(X, y, k){
  # K-fold cross validation. For leave-one-out set k to num of observations.
  # X should be a (d+1)*n matrix with d = num of features, n = numb of samples.
  # y should be 1*n target vector.
  # Create a vector where errors will be stored. Length will be k
  errors <- rep(0, k)
  # First stack together X and y. y will be last row of X. Then transpose.
  xy <- t(rbind(X, y))
  # shuffle the rows
  xyshuffled <- xy[sample(nrow(xy)), ]
  # Create flags for the rows of xyshuffled to be divided into k folds
  folds <- cut(seq(1, nrow(xyshuffled)), breaks=k, labels=FALSE)
  # Go through each fold and calculate train and test stuff
  for(i in 1:k){
    # Find indeces of rows in the hold-out (test) group
    test_ind <- which(folds==i, arr.ind=TRUE)
    # Use such indeces to grab test data
    test_x <- xyshuffled[test_ind, -dim(xyshuffled)[2]]
    test_y <- xyshuffled[test_ind, dim(xyshuffled)[2]]
    # Use the remaining indeces to grab training data
    train_x <- xyshuffled[-test_ind, -dim(xyshuffled)[2]]
    train_y <- xyshuffled[-test_ind, dim(xyshuffled)[2]]
    # Now use train_x and train_y data to find the parameters. Recall that we want them
    # with num of observations as the second dimension
    w <- least_squares(t(train_x), t(train_y))
    # Now use these parameters to find the fitted value for the current test data
    f <- linear_ls(test_x, w)
    # Now compare the function value with test_y with a suitable error function.
    e <- sum((f - test_y)^2)
    # Add error to error list
    errors[i] <- e
  }
  # Finally return the average error
  return(mean(errors))
}

# There's dim(X)[1] features if we count also the bias feature
errors <- rep(0, dim(X)[1])
# Remove one feature at a time
for (i in 1:dim(X)[1]){
  X <- X[-i, ]
  errors[i] <- cv(X, y, dim(X)[2])
}

# Plot errors and save it as a PNG file
png(filename="output/SM1_HW1.png")
plot(1:length(errors), errors, xlab="Features", xaxt = "n", main="CV errors with feature dropping", ylab="CV errors")
labels <- names(df[1:8])
labels[9] <- "bias"
axis(1, at=1:length(errors), labels=labels)
# Save the plot
dev.off()
