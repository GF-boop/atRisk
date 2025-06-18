#' Estimation of quantiles
#'
#' @param qt_trgt Numeric vector, dim k, of k quantiles for different qt-estimations
#' @param v_dep Numeric vector of the dependent variable
#' @param v_expl Numeric vector or matrix of the (k) explanatory covariate(s)
#' @param newdata Numeric optional vector of the (k) out of sample explanatory covariate(s)
#'
#' @return A list with the following elements:
#' \item{quantile_target}{Numeric vector, dim k, of k quantiles for different qt-estimations.}
#' \item{results_qt}{Numeric matrix with all the predicted values based on each quantile regression, where each column corresponds to a quantile target. This matrix includes out-of-sample values of the dependent variable if `newdata` is specified.}
#' @importFrom stats predict
#' @importFrom quantreg rq
#' @description
#' Predicted values based on each quantile regression (Koenker and Basset, 1978), at time=t_trgt, for each quantile in qt_trgt.
#'
#' @references Koenker, Roger, and Gilbert Bassett Jr. "Regression quantiles." Econometrica: journal of the Econometric Society (1978): 33-50.
#'
#' @export
#' @examples
#' # Import data
#' data("data_euro")
#'
#' # Data process
#' PIB_euro_forward_4 = data_euro["GDP"][c(5:length(data_euro["GDP"][,1])),]
#' FCI_euro_lag_4 = data_euro["FCI"][c(1:(length(data_euro["GDP"][,1]) - 4)),]
#' CISS_euro_lag_4 = data_euro["CISS"][c(1:(length(data_euro["GDP"][,1]) - 4)),]
#'
#' quantile_target <- as.vector(c(0.10,0.25,0.75,0.90))
#' results_quantile_reg <- f_compile_quantile(qt_trgt=quantile_target,
#' v_dep=PIB_euro_forward_4,
#' v_expl=as.matrix(cbind(FCI_euro_lag_4, CISS_euro_lag_4)))
#'
f_compile_quantile <- function(qt_trgt, v_dep, v_expl, newdata=NULL){

  # Check validity of target quantiles
  if (any(qt_trgt < 0 | qt_trgt > 1)) {
    stop("Target quantiles must be between 0 and 1.")
  }else{
  }

  # Check Validity of input arguments
  if (!is.numeric(qt_trgt) && !is.numeric(v_dep) && !is.numeric(v_expl)) {
    stop("'qt_trgt', 'v_dep' and 'v_expl' must be numeric.")
  } else if (!is.numeric(qt_trgt) && !is.numeric(v_dep)){
    stop("'qt_trgt' and 'v_dep' must be numeric.")
  } else if (!is.numeric(qt_trgt) && !is.numeric(v_expl)){
    stop("'qt_trgt' and 'v_expl' must be numeric.")
  } else if (!is.numeric(v_dep) && !is.numeric(v_dep)){
    stop("'v_dep' and 'v_expl' must be numeric.")
  } else if (!is.numeric(qt_trgt)){
    stop("'qt_trgt' must be numeric.")
  } else if (!is.numeric(v_dep)){
    stop("'v_dep' must be numeric.")
  }else if (!is.numeric(v_expl)){
    stop("'v_expl' must be numeric.")
  } else{}

  # Check missing values
  if (anyNA(v_dep) || anyNA(v_expl)) {
    message("Missing values found in the dataset.")
  } else {
  }

  # number of quantile regressions (for k quantile regressions)
  nb_qt <- length(qt_trgt)
  v_expl <- as.matrix(v_expl)
  t_trgt <- nrow(v_expl)

  results_qt <- matrix(data=0, ncol=nb_qt, nrow=t_trgt)
  if(!is.null(newdata)){
    newdata <- as.matrix(newdata)
    results_qt_new <- matrix(data=0, ncol=nb_qt, nrow=nrow(newdata))
  }

  # loop on each quantile regression
  for (ct_qt in 1:nb_qt){
    reg_qt <- rq(v_dep ~ cbind(v_expl), tau=qt_trgt[ct_qt]) # quantile regression
    for(t in t_trgt){
      pred_qt <- reg_qt$fitted.values
      results_qt[,ct_qt] <- pred_qt
    }
    if(!is.null(newdata)){
      pred_qt_new_data <- reg_qt$coefficients[1] + reg_qt$coefficients[-1] %*% t(newdata)
      results_qt_new[,ct_qt] <- pred_qt_new_data
    }
  }
  if(!is.null(newdata)){
    results_qt <- rbind(results_qt, results_qt_new)
  }
  results <- list(quantile_target=qt_trgt, res_qt = results_qt)
  return(results)

}
