#' Historical parameters
#'
#' @param param_histo Dataframe with the parameters of the distribution for each period.
#' @param type_function String argument : "gaussian" for Normal Distribution, "skew-gaussian" for Skew-Normal Distribution  or "skew-t" for t-student distribution
#' @param alpha Numeric argument for Expected-Shortfall, between 0 and 1
#'
#' @description
#' This function allows to calculate historical historical parameters and the VaR and ES for each historical period.
#'
#' @references
#'
#' @return A list with historical estimated coefficients, VaR(alpha) and ES(alpha)
#' @export
#'
#' @examples
#' \donttest{
#' data("data_euro")
#'
#' # Data process
#' PIB_euro_forward_4 = data_euro["GDP"][c(5:length(data_euro["GDP"][,1])),]
#' FCI_euro_lag_4 = data_euro["FCI"][c(1:(length(data_euro["GDP"][,1]) - 4)),]
#' CISS_euro_lag_4 = data_euro["CISS"][c(1:(length(data_euro["GDP"][,1]) - 4)),]
#'
#' results_quantile_reg <- f_compile_quantile(qt_trgt=as.vector(c(0.10,0.25,0.75,0.90)),
#' v_dep=PIB_euro_forward_4,
#' v_expl=cbind(FCI_euro_lag_4, CISS_euro_lag_4))
#'
#' histo_param <- f_distrib(type_function="skew-t",
#' compile_qt=results_quantile_reg,
#' starting_values=c(0, 1, -0.5, 1.3))
#'
#' # for a skew-t
#' results_s <- f_histo_RM(param_histo = histo_param,
#' type_function="skew-t",
#' alpha=0.95)
#' }
#'
#'
f_histo_RM <- function(param_histo, type_function, alpha){

  nb_T <- nrow(param_histo)

  VaR_res <- as.vector(rep(0,nb_T))
  ES_res <- as.vector(rep(0,nb_T))
  if(type_function=="gaussian"){
    for (count in 1:nb_T){
      VaR_res[count] <- f_VaR(alpha, "gaussian", params=as.matrix(param_histo[count,]))
      ES_res[count] <- f_ES(alpha, "gaussian", params=as.matrix(param_histo[count,]))
    }
  }else if(type_function=="skew-gaussian"){
    for (count in 1:nb_T){
      VaR_res[count] <- f_VaR(alpha, "skew-gaussian", params=as.matrix(param_histo[count,]))
      ES_res[count] <- f_ES(alpha, "skew-gaussian", params=as.matrix(param_histo[count,]))
    }
  }else if(type_function=="skew-t"){
    for (count in 1:nb_T){
      VaR_res[count] <- f_VaR(alpha, "skew-t", params=as.matrix(param_histo[count,]))
      ES_res[count] <- f_ES(alpha, "skew-t", params=as.matrix(param_histo[count,]))
    }
  }else{
    stop("the 'type_function' has to be 'gaussian', 'skew-gaussian' or 'skew-t'")
  }
  results <- list("param_histo"=param_histo, "VaR"=VaR_res, "ES"=ES_res)
  return(results)
}
