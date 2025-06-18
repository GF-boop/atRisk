#' Plot of historical distributions in 2D
#'
#' @param m_param_histo Numeric matrix containing the parameters of the f_param_histo function
#' @param type_function String argument specifying the distribution type (gaussian, skew-gaussian or skew-t)
#' @param v_date Vector optional of dates containing the full sample's dates (default value : daily dates starting from "1970-01-01")
#' @param v_var_dep Numeric vector containing the realization of the dependent variable
#' @param x_lab String optional argument for the x axis title (default value = x)
#' @param y_lab String optionalargument for the y axis title (default value = y)
#' @param x_min Numeric optional argument (default value = VaR 97.5)
#' @param x_max Numeric optional argument (default value = VaR 2.5)
#' @param color_theme  A character vector specifying the color theme to use (default value c("#bd8e42","gray30","#876b3a","khaki1"))
#'
#'
#' @return A plot of historical distributions with the median, four quantiles (5th, 25th, 75th, 95th) and the realized dependent variable.
#' @import ggplot2
#' @export
#' @description
#' This function allows to create a plot in 2D of historical distributions.
#'
#' @examples
#' # Import data
#' data(data_US)
#' data(data_param_histo_US)
#'
#' results_plot_2D <- f_plot_distrib_2D(m_param_histo=data_param_histo_US,
#' type_function="skew-t",
#' v_date=data_US[,1],
#' v_var_dep=data_US[,2],
#' x_lab="US GDP variation",
#' y_lab="Year")

f_plot_distrib_2D <- function(m_param_histo, type_function, v_date=NULL, v_var_dep, x_lab, y_lab, x_min=NULL, x_max=NULL,  color_theme=c("#bd8e42","gray30","#876b3a","khaki1")){

  # quantiles for the plot 2D
  v_quantile <- c(0.05,0.25,0.50,0.75,0.95)

  # Calculation of VaR for quantiles
  VaR_results <- matrix(data=0,nrow=nrow(m_param_histo), ncol=length(v_quantile))
  if(type_function=="gaussian"){
    for(count in 1:nrow(m_param_histo)){
      VaR_results[count,] <- sapply(v_quantile, f_VaR, type_function="gaussian", params=as.matrix(m_param_histo[count,]),simplify=TRUE)
    }
  }else if(type_function=="skew-gaussian"){
    for(count in 1:nrow(m_param_histo)){
      VaR_results[count,] <- sapply(v_quantile, f_VaR, type_function="skew-gaussian", params=as.matrix(m_param_histo[count,]),simplify=TRUE)
    }
  }else if(type_function=="skew-t"){
    for(count in 1:nrow(m_param_histo)){
      VaR_results[count,] <- sapply(v_quantile, f_VaR, type_function="skew-t", params=as.matrix(m_param_histo[count,]),simplify=TRUE)
    }
  }else{
    stop("the 'type_function' has to be 'gaussian', 'skew-gaussian' or 'skew-t'")
  }

  # initialization of parameters
  if(is.null(v_date)){
    dates <- seq(1:nrow(m_param_histo))
    dates <- as.Date(dates, origin = "1970-01-01")
  }else{
    dates <- as.Date(v_date,format="%d/%m/%Y")
  }

  results <- data.frame(VaR_results, dates[-c(1)], v_var_dep[-c(1)])
  colnames(results)=c("Q05","Q25","Q50","Q75","Q95","t","real")



  # Calculation of x_min and x_max if missing
  if (is.null(x_min) && is.null(x_max)){
    v_quantile <- c(0.025,0.975)
    VaR_results <- matrix(data=0,nrow=nrow(m_param_histo), ncol=length(v_quantile))
    if(type_function=="gaussian"){
      for(count in 1:nrow(m_param_histo)){
        VaR_results[count,] <- sapply(v_quantile, f_VaR, type_function="gaussian", params=as.matrix(m_param_histo[count,]),simplify=TRUE)
      }
    }else if(type_function=="skew-gaussian"){
      for(count in 1:nrow(m_param_histo)){
        VaR_results[count,] <- sapply(v_quantile, f_VaR, type_function="skew-gaussian", params=as.matrix(m_param_histo[count,]),simplify=TRUE)
      }
    }else if(type_function=="skew-t"){
      for(count in 1:nrow(m_param_histo)){
        VaR_results[count,] <- sapply(v_quantile, f_VaR, type_function="skew-t", params=as.matrix(m_param_histo[count,]),simplify=TRUE)
      }
    }else{
    }
    x_min <- floor(min(VaR_results[,2]))
    x_max <- ceiling(max(VaR_results[,1]))
  }

  if (missing(x_lab)){
    x_lab <- "x"
  }
  if (missing(y_lab)){
    y_lab <- "y"
  }
  p_0 <- ggplot()+
    geom_ribbon(data=results, aes(xmin=results$Q05, xmax=results$Q95, y=results$t),fill=color_theme[2], alpha=0.2) +
    geom_ribbon(data=results, aes(xmin=results$Q25, xmax=results$Q75, y=results$t),fill=color_theme[2], alpha=0.2) +
    geom_path(data=results,aes(x=results$real,y=t, color="realized"),group=2, linewidth = 0.5) +
    geom_path(data=results,aes(x=results$Q50,y=results$t, color="median"), linewidth = 0.4) +
    geom_path(data=results,aes(x=results$Q95,y=results$t, color="Q5th & Q95th"), linewidth = 0.05) +
    geom_path(data=results,aes(x=results$Q05,y=results$t, color="Q5th & Q95th"), linewidth = 0.05) +
    geom_path(data=results,aes(x=results$Q25,y=results$t, color="Q25th & Q75th"), linewidth = 0.4) +
    geom_path(data=results,aes(x=results$Q75,y=results$t, color="Q25th & Q75th"), linewidth = 0.4) +
    scale_color_manual(values=c(color_theme[4], color_theme[1], "black", color_theme[3]))+
    labs(x = x_lab,
         y = y_lab,
         color = "") +
    theme(legend.title = element_text(color = "black", size = 8),
          legend.text = element_text(color = "black", size = 6),
          legend.key.size=unit(0.25, 'cm'))+
    xlim(x_min, x_max)

  return(p_0)
}
