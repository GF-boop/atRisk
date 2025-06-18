#' Distribution
#'
#' @param type_function String argument : "gaussian" for Normal Distribution, "skew-gaussian" for Skew-Normal Distribution or "skew-t" for t-student Distribution
#' @param compile_qt List containing the results of f_compile_quantile function
#' @param starting_values Numeric vector with initial values for optimization
#' @param tolerance Numeric optional for the convergence tolerance. Iteration is terminated when the absolute difference in function value between successive iteration is below tol (default value = 1.e-06).
#'
#' @return Dataframe with the parameters of the distribution for each period. This dataframe includes out of sample values of the paramateres if newdata has been specified in f_compile_quantile.
#' @importFrom dfoptim nmkb
#' @importFrom stats qnorm
#' @importFrom sn qst qsn
#' @description
#' This function is used to estimate the parameters of the distribution for each period (mean and standard deviation for Gaussian Distribution, xi, omega and alpha for Skew-Normal Distribution, and xi, omega, alpha, and nu for t-student Distribution) based on the quantile regression results (Koenker and Basset, 1978). See Adrian et al. (2019) and Adrian et al. (2022) for more details on the estimation steps.
#'
#' @references Adrian, Tobias, Nina Boyarchenko, and Domenico Giannone. "Vulnerable growth." American Economic Review 109.4 (2019): 1263-89.
#' @references Adrian, Tobias, et al. "The term structure of growth-at-risk." American Economic Journal: Macroeconomics 14.3 (2022): 283-323.
#' @references Koenker, Roger, and Gilbert Bassett Jr. "Regression quantiles." Econometrica: journal of the Econometric Society (1978): 33-50.
#' @references Azzalini, Adelchi. "The skew-normal and related families." Vol. 3. Cambridge University Press (2013).
#'
#' @export
#'
#' @examples
#' \donttest{
#' # Import data
#' data("data_euro")
#'
#' # Data process
#' PIB_euro_forward_4 = data_euro["GDP"][c(5:length(data_euro["GDP"][,1])),]
#' FCI_euro_lag_4 = data_euro["FCI"][c(1:(length(data_euro["GDP"][,1]) - 4)),]
#' CISS_euro_lag_4 = data_euro["CISS"][c(1:(length(data_euro["GDP"][,1]) - 4)),]
#'
#' # for a gaussian
#' quantile_target <- as.vector(c(0.25,0.75))
#' results_quantile_reg <- f_compile_quantile(qt_trgt=quantile_target,
#' v_dep=PIB_euro_forward_4,
#' v_expl=cbind(FCI_euro_lag_4, CISS_euro_lag_4))
#'
#' results_g <- f_distrib(type_function="gaussian",
#' compile_qt=results_quantile_reg,
#' starting_values=c(0, 1))
#'
#' # for a skew-t
#' quantile_target <- as.vector(c(0.10,0.25,0.75,0.90))
#' results_quantile_reg <- f_compile_quantile(qt_trgt=quantile_target,
#' v_dep=PIB_euro_forward_4,
#' v_expl=cbind(FCI_euro_lag_4, CISS_euro_lag_4))
#'
#' results <- f_distrib(type_function="skew-t",
#' compile_qt=results_quantile_reg,
#' starting_values=c(0, 0.5, 0, 2), tolerance=1e-05)
#' }
#'
f_distrib <- function(type_function, compile_qt, starting_values, tolerance=1e-06){

  # Check validity of type_function
  if (type_function != "gaussian" && type_function != "skew-t" && type_function != "skew-gaussian") {
    stop("the 'type_function' has to be 'gaussian', 'skew-t' or 'skew-gaussian'")
  }else{
  }
  if (is.matrix(compile_qt$res_qt) == FALSE) {
    stop("'compile_qt' should be a matrix")
  }else{
  }

  # for a gaussian function
  if(type_function=="gaussian"){
    # error management
    if (length(starting_values)!=2){
      stop("for a gaussian function, 'starting_values' has to be of dimension 2")
    }else{
    }
    if(ncol(compile_qt$res_qt)<2){
      stop("for a gaussian function, 'compile_qt$res_qt' has to be a matrix with a minimum of 2 rows")
    } else{
      # objective function
      f_objective <- function(X, par){
        # initialization
        sum <- 0
        # Loop on each elements of X
        for (compteur in 1:nrow(X)){
          sum <- sum + (qnorm(X[compteur,1], mean=par[1], sd=par[2]) - X[compteur,2])^2
        }
        return(sum)
      }
      # optimization
      t_trgt <- nrow(compile_qt$res_qt)
      results_par <- matrix(data=0, ncol=2 , nrow=t_trgt)
      for(cpt_t in 1:t_trgt){
        mat_optim <- cbind(compile_qt$quantile_target,compile_qt$res_qt[cpt_t,])
        param <-nmkb(par=starting_values, fn=f_objective,
                     lower=c(-Inf,0),
                     upper=c(+Inf, +Inf), X=mat_optim)
        results_par[cpt_t,] <- param$par
      }
      results <- data.frame("mean"=results_par[,1], "sd"=results_par[,2])
      return(results)
    }
    # for a skew-t function
  }else if(type_function=="skew-t"){
    # error management
    if (length(starting_values)!=4){
      stop("for a skew-t function, 'starting_values' has to be of dimension 4")
    }else{
    }
    if(ncol(compile_qt$res_qt) < 4){
      stop("for a skew-t function, 'compile_qt$res_qt' has to be a matrix with a minimum of 4 rows")
    } else{
      # objective function
      f_objective <- function(X, par){
        # initialization
        sum <- 0
        # Loop on each elements of X
        for (compteur in 1:nrow(X)){
          sum <- sum + (qst(X[compteur,1], xi=par[1], omega=par[2], alpha=par[3], nu=par[4], tol=tolerance, method=0) - X[compteur,2])^2
        }
        return(sum)
      }
      # optimization
      t_trgt <- nrow(compile_qt$res_qt)
      results_par <- matrix(data=0, ncol=4 , nrow=t_trgt)
      for(cpt_t in 1:t_trgt){
        mat_optim <- cbind(compile_qt$quantile_target,compile_qt$res_qt[cpt_t,])
        param <-nmkb(par=starting_values, fn=f_objective,
                     lower=c(-Inf,10e-6, -1, 10e-6),
                     upper=c(+Inf, +Inf, 1, +Inf), X=mat_optim, control = list(tol = tolerance))

        results_par[cpt_t,] <- param$par
      }
      results <- data.frame("xi"=results_par[,1], "omega"=results_par[,2], "alpha"=results_par[,3], "nu"=results_par[,4])
      return(results)
    }
  }else if(type_function=="skew-gaussian"){
    # error management
    if (length(starting_values)!=3){
      stop("for a skew-gaussian function, 'starting_values' has to be of dimension 3")
    }else{
    }
    if(ncol(compile_qt$res_qt) < 3){
      stop("for a skew-gaussian function, 'compile_qt$res_qt' has to be a matrix with a minimum of 3 rows")
    } else{
      # objective function
      f_objective <- function(X, par){
        # initialization
        sum <- 0
        # Loop on each elements of X
        for (compteur in 1:nrow(X)){
          sum <- sum + (qsn(X[compteur,1], xi=par[1], omega=par[2], alpha=par[3], tol=tolerance, method=0) - X[compteur,2])^2
        }
        return(sum)
      }

      # optimization
      t_trgt <- nrow(compile_qt$res_qt)
      results_par <- matrix(data=0, ncol=3 , nrow=t_trgt)
      for(cpt_t in 1:t_trgt){
        mat_optim <- cbind(compile_qt$quantile_target,compile_qt$res_qt[cpt_t,])
        param <-nmkb(par=starting_values, fn=f_objective,
                     lower=c(-Inf,10e-2, -1),
                     upper=c(+Inf, +Inf, 1), X=mat_optim, control = list(tol = tolerance))
        results_par[cpt_t,] <- param$par
      }
      results <- data.frame("xi"=results_par[,1], "omega"=results_par[,2], "alpha"=results_par[,3])
      return(results)
    }
  }else{}
}
