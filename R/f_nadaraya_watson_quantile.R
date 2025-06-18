#' Estimation of quantiles using the Nadaraya-Watson estimator with a product kernel
#'
#' @param qt_trgt Numeric vector, dim k, of k quantiles for different qt-estimations
#' @param v_dep Numeric vector of the dependent variable
#' @param v_expl Numeric vector or matrix of the (k) explanatory covariate(s)
#' @param bandwidth Numeric value specifying the bandwidth for the Gaussian kernel
#'
#' @return Numeric matrix with all the predicted values based on each quantile regression, where each column corresponds to a quantile target. 
#' @description
#' This function performs quantile regression using the Nadaraya-Watson estimator with a product kernel. It computes the weights using a Gaussian kernel for each dimension of the explanatory variables and then estimates the quantile using a weighted average of the observed responses.
#'
#' @export
#' @examples
#' # Data process
#' set.seed(123)
#' Y <- as.vector(rnorm(100))
#' X <- matrix(rnorm(200), ncol = 2)
#' quantile_target <- c(0.1, 0.5, 0.9)
#' bandwidth_value <- 0.5
#'
#' results_qt <- f_nadaraya_watson_quantile(v_dep=Y, 
#' v_expl=X, 
#' qt_trgt=quantile_target, 
#' bandwidth=bandwidth_value)
#'
f_nadaraya_watson_quantile <- function(v_dep, v_expl, qt_trgt, bandwidth){
  
  f_gaussian_kernel <- function(u, bandwidth) {
    exp(-0.5 * (u / bandwidth)^2) / (sqrt(2 * pi) * bandwidth)
  }
  n <- length(v_dep)
  v_expl <- as.matrix(v_expl)
  p <- ncol(v_expl)
  results_qt <- matrix(0, nrow = n, ncol = length(qt_trgt))
  
  for (ct_qt in seq_along(qt_trgt)) {
    qt <- qt_trgt[ct_qt]
    for (i in 1:n) {
      # Compute the weights using the Gaussian kernel
      weights <- apply(v_expl, 1, function(x) {
        prod(f_gaussian_kernel(x - v_expl[i, ], bandwidth))
      })
      
      # Compute the weighted quantile
      sorted_indices <- order(v_dep)
      sorted_weights <- weights[sorted_indices]
      cumulative_weights <- cumsum(sorted_weights)
      total_weight <- sum(sorted_weights)
      
      # Find the quantile index
      quantile_index <- which.min(abs(cumulative_weights / total_weight - qt))
      results_qt[i, ct_qt] <- v_dep[sorted_indices[quantile_index]]
    }
  }
  return(results_qt)
}
