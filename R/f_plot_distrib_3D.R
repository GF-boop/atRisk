#' Plot of historical distributions in 3D
#'
#' @param m_param_histo Numeric matrix containing the parameters of the f_param_histo function
#' @param type_function String argument specifying the distribution type ("gaussian", "skew-gaussian" or "skew-t")
#' @param v_date Vector optional of dates containing the full sample's dates (default value : daily dates starting from "1970-01-01")
#' @param n_samples Number optional of samples for the plot (default value = 1000)
#' @param x_lab String optional argument for the x axis title (default value = x)
#' @param y_lab String optional argument for the y axis title (default value = y)
#' @param x_min Numeric optional argument (default value = VaR 97.5)
#' @param x_max Numeric optional argument (default value = VaR 2.5)
#' @param color_theme  A character vector specifying the color theme to use (default value c("#bd8e42", "#bebfbf"))
#'
#'
#' @return A plot in 3D of historical distributions
#' @import ggplot2
#' @importFrom sn rsn rst
#' @importFrom stats rnorm
#' @importFrom ggridges geom_density_ridges_gradient
#' @export
#' @description
#' This function allows to create a plot in 3D of historical distributions.
#'
#' @examples
#' # Import data
#' data(data_US)
#'
#' data(data_param_histo_US)
#'
#' results_plot_3D <- f_plot_distrib_3D(m_param_histo=data_param_histo_US,
#' type_function="skew-t",
#' v_date=data_US[,1],
#' x_lab="US GDP variation",
#' y_lab="Year")

f_plot_distrib_3D <- function(m_param_histo, type_function, v_date=NULL, n_samples=10000, x_min=NULL, x_max=NULL, x_lab, y_lab, color_theme=c("#bd8e42","#bebfbf")){

  # initialization of parameters
  if(is.null(v_date)){
    dates <- seq(1:nrow(m_param_histo))
    dates <- as.Date(dates, origin = "1970-01-01")
  }else{
    dates <- as.Date(v_date,format="%d/%m/%Y")
    dates <- dates[-length(dates)]
  }

  n_dates <- length(dates)
  samples_3D <- data.frame(matrix(data=0, nrow=n_dates*n_samples, ncol = 4))
  colnames(samples_3D) <- c("date", "month", "X", "mean_value")
  samples_3D[,1] <- rep(dates[c(1 : n_dates)], n_samples)
  samples_3D[,2] <- as.Date(as.Date(samples_3D[,1], format="%Y-%m-%d"))

  if (type_function=="skew-t"){
    for (count_date in c(1:n_dates)){
      samples_3D[seq(count_date, n_samples*n_dates, n_dates), 3] <- rst(n_samples, dp = c(m_param_histo[count_date, 1], m_param_histo[count_date, 2], m_param_histo[count_date, 3], m_param_histo[count_date, 4]))
    }
  }else if (type_function=="gaussian"){
    for (count_date in c(1:n_dates)){
      samples_3D[seq(count_date, n_samples*n_dates, n_dates), 3] <- rnorm(n_samples, m_param_histo[count_date, 1], m_param_histo[count_date, 2])
    }
  }else if (type_function=="skew-gaussian"){
    for (count_date in c(1:n_dates)){
      samples_3D[seq(count_date, n_samples*n_dates, n_dates), 3] <- rsn(n_samples, dp = c(m_param_histo[count_date, 1], m_param_histo[count_date, 2], m_param_histo[count_date, 3]))
    }
  }else{
    stop("the 'type_function' has to be 'gaussian' or 'skew-t'")
  }

  mean_values = matrix(data = NA, nrow=n_dates)
  for (count_date in c(1:n_dates)){
    mean_values[count_date] <- mean(samples_3D[which(samples_3D$date == dates[count_date]),3], na.rm = TRUE)
  }

  for (count_date in c(1:n_dates)){
    samples_3D[seq(count_date, n_samples*n_dates, n_dates), 4] <- rep(mean_values[count_date], n_samples)
  }

  # Calculation of x_min and x_max if missing
  if (is.null(x_min) && is.null(x_max)){
    v_quantile <- c(0.025,0.975)
    VaR_results <- matrix(data=0,nrow=nrow(m_param_histo), ncol=length(v_quantile))
    if(type_function=="gaussian"){
      for(count in 1:nrow(m_param_histo)){
        VaR_results[count,] <- sapply(v_quantile, f_VaR, type_function="gaussian", params=as.matrix(m_param_histo[count,]),simplify=TRUE)
      }
    }else if(type_function=="skew-t"){
      for(count in 1:nrow(m_param_histo)){
        VaR_results[count,] <- sapply(v_quantile, f_VaR, type_function="skew-t", params=as.matrix(m_param_histo[count,]),simplify=TRUE)
      }
    }else if(type_function=="skew-gaussian"){
      for(count in 1:nrow(m_param_histo)){
        VaR_results[count,] <- sapply(v_quantile, f_VaR, type_function="skew-gaussian", params=as.matrix(m_param_histo[count,]),simplify=TRUE)
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

  #p_1 <- ggplot(samples_3D, aes(x = X, y= date, group= date, fill=after_stat(x))) +
  #  geom_density_ridges_gradient(scale = 20, size = 0.1, rel_min_height = 0.05) +
  #  scale_fill_gradient(low = color_theme[1], high = color_theme[2]) +
  #  xlim(x_min, x_max) +
  #  theme(plot.title = element_text(hjust = 0.5),
  #        legend.key.size = unit(0.25, 'cm'),
  #        legend.title = element_text(size=8)) +
  #  labs(x=x_lab, y = y_lab) +
  #  guides(fill=guide_legend(""))

  p_1 <- ggplot(samples_3D, aes(x = .data$X, y = .data$date, group = .data$date, fill=after_stat(.data$x))) +
    geom_density_ridges_gradient(scale = 20, size = 0.1, rel_min_height = 0.05) +
    scale_fill_gradient(low = color_theme[1], high = color_theme[2]) +
    xlim(x_min, x_max) +
    theme(plot.title = element_text(hjust = 0.5),
          legend.key.size = unit(0.25, 'cm'),
          legend.title = element_text(size = 8)) +
    labs(x = x_lab, y = y_lab) +
    guides(fill = guide_legend(""))
  return(p_1)

}
