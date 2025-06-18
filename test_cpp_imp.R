# test_skewt_timing.R - UPDATED FOR EXPLICIT ARGUMENTS

library(rstudioapi)
library(microbenchmark) # For timing comparisons
library(sn)             # For comparing against sn package
library(e1071)          # For skewness/kurtosis calculations
library(ggplot2)        # For autoplot (if desired)

# Set working directory to the script's location
# IMPORTANT: Adjust "cpp/sn_cpp.cpp" to the actual path of your C++ file
file_path <- getSourceEditorContext()$path
file_dir <- dirname(file_path)
setwd(file_dir)

# Source your C++ implementation (which has defaults in declarations, not definitions)
Rcpp::sourceCpp("cpp/sn_cpp.cpp")

cat("--- Comparing sn_cpp (Simpson's Rule) to sn package (R) ---\n")

# --- Common Parameters for Testing ---
# Using parameters that will challenge numerical integration and show performance differences
# High alpha (skewness) and moderate nu (degrees of freedom)
params_common <- list(xi = 0.5, omega = 1.5, alpha = 8, nu = 4.5) # Fractional nu for integration tests

# --- DST (Density) Comparison ---
cat("\n--- Timing dst_cpp vs sn::dst ---\n")
x_vals_dst <- seq(-5, 5, length.out = 1000)

bench_dst <- microbenchmark(
  cpp = dst_cpp(x_vals_dst, xi = params_common$xi, omega = params_common$omega,
                alpha = params_common$alpha, nu = params_common$nu), # ADDED log_d = FALSE
  sn = sn::dst(x_vals_dst, xi = params_common$xi, omega = params_common$omega,
               alpha = params_common$alpha, nu = params_common$nu), # sn::dst has its own defaults
  times = 100
)
print(bench_dst)
cat("Mean dst_cpp time:", mean(bench_dst$time[bench_dst$expr == "cpp"]) / 1e6, "ms\n")
cat("Mean sn::dst time:", mean(bench_dst$time[bench_dst$expr == "sn"]) / 1e6, "ms\n")
cat("Speedup (sn / cpp):", mean(bench_dst$time[bench_dst$expr == "sn"]) / mean(bench_dst$time[bench_dst$expr == "cpp"]), "x\n")


# --- RST (Random Generation) Comparison ---
cat("\n--- Timing rst_cpp vs sn::rst ---\n")
n_samples_rst <- 1000000

set.seed(42)
rst_cpp_val <- rst_cpp(n_samples_rst, xi = params_common$xi, omega = params_common$omega,
                       alpha = params_common$alpha, nu = params_common$nu) # nu explicitly passed
set.seed(42)
rst_sn_val <- sn::rst(n_samples_rst, xi = params_common$xi, omega = params_common$omega,
                      alpha = params_common$alpha, nu = params_common$nu)

cat("\nStatistical moments for rst_cpp:\n")
print(c(mean = mean(rst_cpp_val), sd = sd(rst_cpp_val),
        skewness = e1071::skewness(rst_cpp_val), kurtosis = e1071::kurtosis(rst_cpp_val)))
cat("\nStatistical moments for sn::rst:\n")
print(c(mean = mean(rst_sn_val), sd = sd(rst_sn_val),
        skewness = e1071::skewness(rst_sn_val), kurtosis = e1071::kurtosis(rst_sn_val)))
cat("\nKolmogorov-Smirnov Test (rst_cpp vs sn::rst):\n")
print(ks.test(rst_cpp_val, rst_sn_val))

set.seed(123)
bench_rst <- microbenchmark(
  cpp = rst_cpp(n_samples_rst, xi = params_common$xi, omega = params_common$omega,
                alpha = params_common$alpha, nu = params_common$nu), # nu explicitly passed
  sn = sn::rst(n_samples_rst, xi = params_common$xi, omega = params_common$omega,
               alpha = params_common$alpha, nu = params_common$nu),
  times = 10
)
print(bench_rst)
cat("Mean rst_cpp time:", mean(bench_rst$time[bench_rst$expr == "cpp"]) / 1e6, "ms\n")
cat("Mean sn::rst time:", mean(bench_rst$time[bench_rst$expr == "sn"]) / 1e6, "ms\n")
cat("Speedup (sn / cpp):", mean(bench_rst$time[bench_rst$expr == "sn"]) / mean(bench_rst$time[bench_rst$expr == "cpp"]), "x\n")


# --- PST (Cumulative Distribution) Comparison ---
cat("\n--- Timing pst_cpp vs sn::pst (Fractional nu - using numerical integration) ---\n")
x_vals_pst <- seq(-5, 5, length.out = 50)

bench_pst_frac <- microbenchmark(
  cpp = pst_cpp(x_vals_pst, xi = params_common$xi, omega = params_common$omega,
                alpha = params_common$alpha, nu = params_common$nu),
  sn = sn::pst(x_vals_pst, xi = params_common$xi, omega = params_common$omega,
               alpha = params_common$alpha, nu = params_common$nu),
  times = 10
)
print(bench_pst_frac)
cat("Mean pst_cpp (frac nu) time:", mean(bench_pst_frac$time[bench_pst_frac$expr == "cpp"]) / 1e6, "ms\n")
cat("Mean sn::pst (frac nu) time:", mean(bench_pst_frac$time[bench_pst_frac$expr == "sn"]) / 1e6, "ms\n")
cat("Speedup (sn / cpp, frac nu):", mean(bench_pst_frac$time[bench_pst_frac$expr == "sn"]) / mean(bench_pst_frac$time[bench_pst_frac$expr == "cpp"]), "x\n")

cat("\nValues Comparison (pst, fractional nu):\n")
val_cpp_pst_frac <- pst_cpp(x_vals_pst, xi = params_common$xi, omega = params_common$omega,
                            alpha = params_common$alpha, nu = params_common$nu, method = 0, lower_tail = TRUE, log_p = FALSE) # ADDED method, lower_tail, log_p
val_sn_pst_frac <- sn::pst(x_vals_pst, xi = params_common$xi, omega = params_common$omega,
                           alpha = params_common$alpha, nu = params_common$nu)
print(data.frame(x = x_vals_pst, cpp = val_cpp_pst_frac, sn = val_sn_pst_frac,
                 abs_diff = abs(val_cpp_pst_frac - val_sn_pst_frac)))


# --- QST (Quantile) Comparison ---
cat("\n--- Timing qst_cpp vs sn::qst (Fractional nu - relies on numerical integration) ---\n")
p_vals_qst <- c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)

bench_qst_frac <- microbenchmark(
  cpp = qst_cpp(p_vals_qst, xi = params_common$xi, omega = params_common$omega,
                alpha = params_common$alpha, nu = params_common$nu),
  sn = sn::qst(p_vals_qst, xi = params_common$xi, omega = params_common$omega,
               alpha = params_common$alpha, nu = params_common$nu),
  times = 5
)
print(bench_qst_frac)
cat("Mean qst_cpp (frac nu) time:", mean(bench_qst_frac$time[bench_qst_frac$expr == "cpp"]) / 1e6, "ms\n")
cat("Mean sn::qst (frac nu) time:", mean(bench_qst_frac$time[bench_qst_frac$expr == "sn"]) / 1e6, "ms\n")
cat("Speedup (sn / cpp, frac nu):", mean(bench_qst_frac$time[bench_qst_frac$expr == "sn"]) / mean(bench_qst_frac$time[bench_qst_frac$expr == "cpp"]), "x\n")

cat("\nValues Comparison (qst, fractional nu):\n")
val_cpp_qst_frac <- qst_cpp(p_vals_qst, xi = params_common$xi, omega = params_common$omega,
                            alpha = params_common$alpha, nu = params_common$nu, tol = 1e-8, method = 0)
val_sn_qst_frac <- sn::qst(p_vals_qst, xi = params_common$xi, omega = params_common$omega,
                           alpha = params_common$alpha, nu = params_common$nu, method=0)
print(data.frame(p = p_vals_qst, cpp = val_cpp_qst_frac, sn = val_sn_qst_frac,
                 abs_diff = abs(val_cpp_qst_frac - val_sn_qst_frac)))


# --- Additional comparison for PST/QST with integer nu (method 4, no integration) ---
cat("\n--- Timing pst_cpp vs sn::pst (Integer nu - using recursive formula) ---\n")
params_int_nu <- list(xi = 0.5, omega = 1.5, alpha = 8, nu = 5) # Integer nu

bench_pst_int <- microbenchmark(
  cpp = pst_cpp(x_vals_pst, xi = params_int_nu$xi, omega = params_int_nu$omega,
                alpha = params_int_nu$alpha, nu = params_int_nu$nu, method = 4, lower_tail = TRUE, log_p = FALSE), # ADDED method, lower_tail, log_p
  sn = sn::pst(x_vals_pst, xi = params_int_nu$xi, omega = params_int_nu$omega,
               alpha = params_int_nu$alpha, nu = params_int_nu$nu),
  times = 100
)
print(bench_pst_int)
cat("Mean pst_cpp (int nu) time:", mean(bench_pst_int$time[bench_pst_int$expr == "cpp"]) / 1e6, "ms\n")
cat("Mean sn::pst (int nu) time:", mean(bench_pst_int$time[bench_pst_int$expr == "sn"]) / 1e6, "ms\n")
cat("Speedup (sn / cpp, int nu):", mean(bench_pst_int$time[bench_pst_int$expr == "sn"]) / mean(bench_pst_int$time[bench_pst_int$expr == "cpp"]), "x\n")

cat("\n--- Timing qst_cpp vs sn::qst (Integer nu - using recursive formula) ---\n")
bench_qst_int <- microbenchmark(
  cpp = qst_cpp(p_vals_qst, xi = params_int_nu$xi, omega = params_int_nu$omega,
                alpha = params_int_nu$alpha, nu = params_int_nu$nu, tol = 1e-8, method = 4), # ADDED tol, method
  sn = sn::qst(p_vals_qst, xi = params_int_nu$xi, omega = params_int_nu$omega,
               alpha = params_int_nu$alpha, nu = params_int_nu$nu, tol=1e-8),
  times = 50
)
print(bench_qst_int)
cat("Mean qst_cpp (int nu) time:", mean(bench_qst_int$time[bench_qst_int$expr == "cpp"]) / 1e6, "ms\n")
cat("Mean sn::qst (int nu) time:", mean(bench_qst_int$time[bench_qst_int$expr == "sn"]) / 1e6, "ms\n")
cat("Speedup (sn / cpp, int nu):", mean(bench_qst_int$time[bench_qst_int$expr == "sn"]) / mean(bench_qst_int$time[bench_qst_int$expr == "cpp"]), "x\n")

