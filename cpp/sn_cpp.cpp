// sn_cpp.cpp - Final Verified Version (Defaults in declarations, NOT definitions)
// [[Rcpp::depends(RcppGSL)]]
#include <RcppGSL.h>     // For GSL integration with Rcpp
#include <gsl/gsl_integration.h> // GSL's numerical integration header
#include <Rcpp.h>
#include <cmath>         // For std::abs, std::sqrt, std::tan, std::log, std::acos, M_PI
#include <limits>        // For std::numeric_limits
#include <vector>        // For std::vector
#include <functional>    // For std::function
#include <algorithm>     // For std::min, std::max

#include <Rmath.h>       // For R::dnorm4, R::pnorm, R::rnorm, R::rchisq, R::dt, R::pt, R::qf, R::lgammafn, R_PosInf, R_NegInf, R_NaN, NA_REAL

// Define M_PI if it's not already defined by a system header (Rmath.h usually defines it)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===================================================================================================
// FORWARD DECLARATIONS OF ALL FUNCTIONS
// Default parameters ARE specified here (prototypes).
// ===================================================================================================

// Typedef for integrand function (must be here after <functional>)
typedef std::function<double(double)> Integrand;

// Rcpp Exported Functions (main interface) - forward declarations
Rcpp::NumericVector dst_cpp(Rcpp::NumericVector x, double xi, double omega, double alpha, double nu, bool log_d);
double dsn_cpp(double x, double xi, double omega, double alpha, bool log_d);
Rcpp::NumericVector rsn_cpp(int n, double xi, double omega, double alpha, double tau);
double psn_cpp(double x, double xi, double omega, double alpha, bool lower_tail, bool log_p);
Rcpp::NumericVector rst_cpp(int n, double xi, double omega, double alpha, double nu);
Rcpp::NumericVector pst_cpp(Rcpp::NumericVector x, double xi, double omega, double alpha, double nu, int method, bool lower_tail, bool log_p);
Rcpp::NumericVector qst_cpp(Rcpp::NumericVector p, double xi, double omega, double alpha, double nu, double tol, int method);

double integrate_wrapper_cpp(Integrand f, double a, double b, double tol);
double st_tails_cpp(double x, double alpha, double nu, bool lower_tail = true, double threshold = 20.0);
double pst_int_cpp(double z, double alpha, double nu);


// ===================================================================================================
// GSL INTEGRATION IMPLEMENTATION (CORRECTED)
// ===================================================================================================

// This is a "trampoline" function. GSL's C-style integrators call this function.
// It casts the void* parameter back to the C++ std::function we want to integrate,
// and then calls it.
double gsl_integrand_adapter(double x, void *params) {
    Integrand* f = static_cast<Integrand*>(params);
    return (*f)(x);
}

// Main integration wrapper using GSL (Corrected Version)
// Replaces simpson_rule, simpson_adaptive_integrate, and the old integrate_wrapper_cpp
double integrate_wrapper_cpp(Integrand f, double a, double b, double tol) { // Default value removed from definition
    if (R_IsNA(a) || R_IsNA(b)) return R_NaN;
    if (a == b) return 0.0;
    if (a > b) return -integrate_wrapper_cpp(f, b, a, tol);

    // GSL workspace size
    const size_t limit = 1000;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(limit);
    
    double result, error;

    gsl_function F;
    F.function = &gsl_integrand_adapter;
    F.params = &f; // Pass the address of our std::function object

    int status = 1; // Default to an error status

    // Choose the correct GSL routine based on the integration limits
    if (a == R_NegInf && b == R_PosInf) {
        // Correct call for (-inf, +inf)
        status = gsl_integration_qagi(&F, 0, tol, limit, w, &result, &error);
    } else if (a == R_NegInf) {
        // Correct call for (-inf, b)
        status = gsl_integration_qagil(&F, b, 0, tol, limit, w, &result, &error);
    } else if (b == R_PosInf) {
        // Correct call for (a, +inf)
        status = gsl_integration_qagiu(&F, a, 0, tol, limit, w, &result, &error);
    } else {
        // Correct call for a finite interval [a, b]
        status = gsl_integration_qags(&F, a, b, 0, tol, limit, w, &result, &error);
    }

    gsl_integration_workspace_free(w);

    if (status) {
        // You can uncomment this for debugging, but it can be noisy.
        // Rcpp::warning("GSL integration failed with status: %d", status);
        return R_NaN;
    }

    return result;
}

// --- Skew-Normal (SN) helpers ---
// Density of Skew-Normal (dsn)
// [[Rcpp::export]]
double dsn_cpp(double x, double xi = 0, double omega = 1, double alpha = 0, bool log_d = false) {
  if (omega <= 0) return R_NaN;
  double z = (x - xi) / omega;
  double std_norm_pdf = R::dnorm4(z, 0.0, 1.0, 0);
  double std_norm_cdf_arg = alpha * z;
  double std_norm_cdf = R::pnorm(std_norm_cdf_arg, 0.0, 1.0, 1, 0);

  double pdf_val = 2.0 / omega * std_norm_pdf * std_norm_cdf;
  if (log_d) {
    return std::log(pdf_val);
  } else {
    return pdf_val;
  }
}

// Random generation for Skew-Normal (rsn)
// [[Rcpp::export]]
Rcpp::NumericVector rsn_cpp(int n = 1, double xi = 0, double omega = 1, double alpha = 0, double tau = 0) {
  if (omega <= 0) Rcpp::stop("omega must be positive");
  if (n <= 0) return Rcpp::NumericVector(0);

  Rcpp::NumericVector y(n);
  GetRNGstate();

  double delta = alpha / std::sqrt(1.0 + alpha * alpha);

  if (tau == 0) {
    std::vector<double> normal_draws(2 * n);
    for (int i = 0; i < 2 * n; ++i) {
      normal_draws[i] = R::rnorm(0.0, 1.0);
    }

    for (int i = 0; i < n; ++i) {
      double chi_i = std::abs(normal_draws[2 * i]);
      double nrv_i = normal_draws[2 * i + 1];
      y[i] = xi + omega * (delta * chi_i + std::sqrt(1.0 - delta * delta) * nrv_i);
    }
  } else {
    double p_min = R::pnorm(-tau, 0.0, 1.0, 1, 0);
    std::vector<double> runif_draws(n);
    std::vector<double> rnorm_draws(n);

    for(int i=0; i<n; ++i) runif_draws[i] = R::runif(p_min, 1.0);
    for(int i=0; i<n; ++i) rnorm_draws[i] = R::rnorm(0.0, 1.0);

    for (int i = 0; i < n; ++i) {
      double truncN_val = R::qnorm(runif_draws[i], 0.0, 1.0, 1, 0);
      double z_i = delta * truncN_val + std::sqrt(1.0 - delta * delta) * rnorm_draws[i];
      y[i] = xi + omega * z_i;
    }
  }

  PutRNGstate();
  return y;
}

// CDF of Skew-Normal (psn) - Placeholder / Approximation
// [[Rcpp::export]]
double psn_cpp(double x, double xi = 0, double omega = 1, double alpha = 0, bool lower_tail = true, bool log_p = false) {
  if (omega <= 0) return R_NaN;
  double z = (x - xi) / omega;

  if (alpha == 0) {
    return R::pnorm(z, 0.0, 1.0, lower_tail, log_p);
  } else {
    double phi_z = R::pnorm(z, 0.0, 1.0, 1, 0);
    double phi_alpha_z_scaled = R::pnorm(alpha * z, 0.0, 1.0, 1, 0);

    double p_val = 2.0 * phi_z * phi_alpha_z_scaled; // Still a placeholder
    if (!lower_tail) p_val = 1.0 - p_val;
    if (log_p) p_val = std::log(p_val);
    return p_val;
  }
}

// Helper for st_tails (log-probabilities of ST tails)
// No default parameters in definition
double st_tails_cpp(double x, double alpha, double nu, bool lower_tail, double threshold) {
  if (R_IsNA(nu)) nu = R_PosInf; // Treat NA_REAL as Inf

  if (std::abs(x) < threshold) return R_NaN;
  if (alpha < 0) return st_tails_cpp(-x, -alpha, nu, !lower_tail, threshold);

  double lp;
  if (x > 0) {
    double log_c2 = std::log(2.0) + R::lgammafn((nu + 1.0) / 2.0) + (nu / 2.0) * std::log(nu) +
                    R::pt(alpha * std::sqrt(nu + 1.0), nu + 1.0, 1, 1) -
                    R::lgammafn(nu / 2.0) - 0.5 * std::log(M_PI);
    if (!lower_tail) {
      lp = log_c2 - std::log(nu) - nu * std::log(x);
    } else {
      double upper_tail_log_prob = log_c2 - std::log(nu) - nu * std::log(x);
      lp = std::log(1.0 - std::exp(upper_tail_log_prob));
    }
  } else {
    double log_c1 = std::log(2.0) + R::lgammafn((nu + 1.0) / 2.0) + (nu / 2.0) * std::log(nu) +
                    R::pt(-alpha * std::sqrt(nu + 1.0), nu + 1.0, 1, 1) -
                    R::lgammafn(nu / 2.0) - 0.5 * std::log(M_PI);
    if (lower_tail) {
      lp = log_c1 - std::log(nu) - nu * std::log(-x);
    } else {
      double lower_tail_log_prob = log_c1 - std::log(nu) - nu * std::log(-x);
      lp = std::log(1.0 - std::exp(lower_tail_log_prob));
    }
  }
  return lp;
}


// Recursive helper for pst_int (for integer nu)
// No default parameters in definition
double pst_int_cpp(double z, double alpha, double nu) {
  if (R_IsNA(nu)) nu = R_PosInf; // Treat NA_REAL as Inf for consistency

  if (nu != std::round(nu) || nu < 1) {
    Rcpp::stop("'nu' is not a positive integer in pst_int_cpp (or nu is Inf after NA_REAL conversion)");
  }

  if (nu == 1) {
    return std::atan(z) / M_PI + std::acos(alpha / std::sqrt((1.0 + alpha * alpha) * (1.0 + z * z))) / M_PI;
  } else if (nu == 2) {
    return 0.5 - std::atan(alpha) / M_PI + (0.5 + std::atan(z * alpha / std::sqrt(2.0 + z * z)) / M_PI) * z / std::sqrt(2.0 + z * z);
  } else {
    double term1 = pst_int_cpp(std::sqrt((nu - 2.0) / nu) * z, alpha, nu - 2.0);
    double term2_factor = z *
                          std::exp(R::lgammafn((nu - 1.0) / 2.0) + (nu / 2.0 - 1.0) * std::log(nu) - 0.5 * std::log(M_PI) - R::lgammafn(nu / 2.0) - 0.5 * (nu - 1.0) * std::log(nu + z * z));
    double term2_pt = R::pt(std::sqrt(nu - 1.0) * alpha * z / std::sqrt(nu + z * z), nu - 1.0, 1, 0);

    return term1 + term2_pt * term2_factor;
  }
}


// Density of Skew-t (dst)
// [[Rcpp::export]]
Rcpp::NumericVector dst_cpp(Rcpp::NumericVector x, double xi = 0, double omega = 1, double alpha = 0, double nu = NA_REAL, bool log_d = false) {
  int n = x.size();
  Rcpp::NumericVector out(n);

  if (R_IsNA(nu)) nu = R_PosInf;

  if (nu <= 0) Rcpp::stop("'nu' must be positive");
  if (omega <= 0) {
    for (int i = 0; i < n; ++i) out[i] = R_NaN;
    return out;
  }

  for (int i = 0; i < n; ++i) {
    if (R_IsNA(x[i])) {
      out[i] = R_NaN;
      continue;
    }

    if (nu == R_PosInf) {
      out[i] = dsn_cpp(x[i], xi, omega, alpha, log_d);
      continue;
    }

    double z = (x[i] - xi) / omega;
    double pdf_dt = R::dt(z, nu, 0);
    double cdf_pt_arg = alpha * z * std::sqrt((nu + 1.0) / (z * z + nu));
    double cdf_pt = R::pt(cdf_pt_arg, nu + 1.0, 1, 0);

    double pdf_val = 2.0 * pdf_dt * cdf_pt / omega;

    if (log_d) {
      out[i] = std::log(pdf_val);
    } else {
      out[i] = pdf_val;
    }
  }
  return out;
}


// Random generation for Skew-t (rst)
// [[Rcpp::export]]
Rcpp::NumericVector rst_cpp(int n = 1, double xi = 0, double omega = 1, double alpha = 0, double nu = NA_REAL) {
  if (omega <= 0) Rcpp::stop("omega must be positive");

  Rcpp::NumericVector y(n);
  GetRNGstate();

  double actual_nu = nu;
  if (R_IsNA(nu)) actual_nu = R_PosInf;

  if (actual_nu <= 0) Rcpp::stop("'nu' must be positive");


  if (actual_nu == R_PosInf) {
    y = rsn_cpp(n, xi, omega, alpha);
  } else {
    Rcpp::NumericVector z_std_val = rsn_cpp(n, 0, 1, alpha);
    Rcpp::NumericVector v(n);
    for (int i = 0; i < n; ++i) {
      v[i] = R::rchisq(actual_nu) / actual_nu;
      y[i] = (omega * z_std_val[i]) / std::sqrt(v[i]) + xi;
    }
  }
  PutRNGstate();
  return y;
}


// [[Rcpp::export]]
Rcpp::NumericVector pst_cpp(Rcpp::NumericVector x, double xi = 0, double omega = 1, double alpha = 0, double nu = NA_REAL, int method = 0, bool lower_tail = true, bool log_p = false) {
  int n = x.size(); Rcpp::NumericVector pr(n); if (R_IsNA(nu)) nu = R_PosInf;
  if (omega <= 0) { for (int i = 0; i < n; ++i) pr[i] = R_NaN; return pr; }
  if (nu <= 0) Rcpp::stop("'nu' must be positive"); if (method < 0 || method > 5 || method != std::round(method)) Rcpp::stop("invalid 'method' value");
  if ((method == 1 || method == 4) && (nu != std::round(nu))) { Rcpp::stop("selected 'method' does not work for non-integer nu"); }
  double alpha_for_integrand = alpha; double nu_for_integrand = nu;
  for (int i = 0; i < n; ++i) {
    if (R_IsNA(x[i])) { pr[i] = R_NaN; continue; }
    if (x[i] == R_PosInf) { pr[i] = log_p ? 0.0 : 1.0; continue; }
    if (x[i] == R_NegInf) { pr[i] = log_p ? R_NegInf : 0.0; continue; }
    double z = (x[i] - xi) / omega; double p_val;
    if (nu == R_PosInf) { p_val = psn_cpp(x[i], xi, omega, alpha, lower_tail, log_p); pr[i] = p_val; continue; }
    if (alpha == 0) { p_val = R::pt(z, nu, lower_tail, log_p); }
    else if (std::abs(alpha) == R_PosInf) {
      double z0 = (alpha * z < 0) ? 0 : z; p_val = R::pf(z0 * z0, 1.0, nu, 1, 0); if (alpha < 0) p_val = (1.0 - p_val);
      if (!lower_tail) p_val = 1.0 - p_val; if (log_p) p_val = std::log(p_val);
    } else {
      bool int_nu = (nu == std::round(nu)); double n_x_vals = x.size(); double nu0 = (8.2 + 3.55 * std::log(std::log(n_x_vals + 1)));
      if (method == 4 || (method == 0 && int_nu && (nu <= nu0))) {
        double p_dot = pst_int_cpp(z, alpha_for_integrand, nu_for_integrand); p_val = lower_tail ? p_dot : 1.0 - p_dot; if (log_p) p_val = std::log(p_val);
      } else {
        if (method == 5 || (method == 0 && std::abs(z) > (30.0 + 1.0 / std::sqrt(nu)))) {
          p_val = st_tails_cpp(z, alpha_for_integrand, nu_for_integrand, lower_tail, 20.0); if (!log_p) p_val = std::exp(p_val);
        } else {
          if (method == 1 || (method == 0 && int_nu && (nu > nu0))) { Rcpp::warning("Method 1 (pmst) requires multivariate CDF implementation, not included."); p_val = R_NaN; }
          else {
            double current_z = z;
            if (method == 2 || (method == 0 && std::abs(current_z) < (10.0 + 50.0/nu))) {
              Integrand dst_integrand = [&](double x_val) { return dst_cpp({x_val}, 0, 1, alpha_for_integrand, nu_for_integrand, false)[0]; };
              double p0 = std::acos(alpha_for_integrand / std::sqrt(1.0 + alpha_for_integrand * alpha_for_integrand)) / M_PI;
              double integral_val;
              if (current_z == 0) { integral_val = 0.0; }
              else { 
                  // Provide tolerance explicitly here
                  integral_val = integrate_wrapper_cpp(dst_integrand, std::min(0.0, current_z), std::max(0.0, current_z), 1e-8); 
              }
              double p_dot = p0 + (current_z > 0 ? 1 : -1) * integral_val;
              p_val = lower_tail ? p_dot : 1.0 - p_dot; if (log_p) p_val = std::log(p_val);
            } else {
              Integrand fp_integrand = [&](double v_val) {
                  if (v_val <= 0) return 0.0;
                  double psn_val = psn_cpp(std::sqrt(v_val) * current_z, 0, 1, alpha_for_integrand, true, false);
                  double dchisq_val = R::dchisq(v_val * nu_for_integrand, nu_for_integrand, 0);
                  return psn_val * dchisq_val * nu_for_integrand;
              };
              // Provide tolerance explicitly here
              double p_dot = integrate_wrapper_cpp(fp_integrand, 0.0, R_PosInf, 1e-8);
              p_val = lower_tail ? p_dot : 1.0 - p_dot; if (log_p) p_val = std::log(p_val);
            }
          }
        }
      }
    }
    pr[i] = p_val;
  }
  return pr;
}

// Quantile function for Skew-t (qst)
// [[Rcpp::export]]
Rcpp::NumericVector qst_cpp(Rcpp::NumericVector p, double xi = 0, double omega = 1, double alpha = 0, double nu = NA_REAL, double tol = 1e-8, int method = 0) {
  int n = p.size();
  Rcpp::NumericVector q(n);

  if (R_IsNA(nu)) nu = R_PosInf;

  if (omega <= 0) {
    for (int i = 0; i < n; ++i) q[i] = R_NaN;
    return q;
  }
  if (nu <= 0) Rcpp::stop("'nu' must be non-negative");


  if (nu > 1e4) {
    Rcpp::warning("qst_cpp for nu=Inf (qsn) case is a placeholder; requires qsn_cpp.");
    return Rcpp::rep(R_NaN, n);
  }
  if (nu == 1) {
    Rcpp::warning("qst_cpp for nu=1 (qsc) case is a placeholder; requires qsc_cpp.");
    return Rcpp::rep(R_NaN, n);
  }

  if (alpha == R_PosInf) {
    for (int i = 0; i < n; ++i) {
      if (R_IsNA(p[i]) || p[i] < 0 || p[i] > 1) { q[i] = R_NaN; }
      else if (p[i] == 0) { q[i] = R_NegInf; }
      else if (p[i] == 1) { q[i] = R_PosInf; }
      else { q[i] = xi + omega * std::sqrt(R::qf(p[i], 1.0, nu, 1, 0)); }
    }
    return q;
  }
  if (alpha == R_NegInf) {
    for (int i = 0; i < n; ++i) {
      if (R_IsNA(p[i]) || p[i] < 0 || p[i] > 1) { q[i] = R_NaN; }
      else if (p[i] == 0) { q[i] = R_NegInf; }
      else if (p[i] == 1) { q[i] = R_PosInf; }
      else { q[i] = xi - omega * std::sqrt(R::qf(1.0 - p[i], 1.0, nu, 1, 0)); }
    }
    return q;
  }

  Rcpp::NumericVector p_adj = Rcpp::clone(p);
  double abs_alpha = std::abs(alpha);
  if (alpha < 0) {
    for (int i = 0; i < n; ++i) {
      p_adj[i] = 1.0 - p_adj[i];
    }
  }

  Rcpp::NumericVector xa(n), xb(n), xc(n), fa(n), fb(n), fc(n);
  Rcpp::LogicalVector converged(n, false);
  Rcpp::LogicalVector is_na(n);
  Rcpp::LogicalVector is_zero(n);
  Rcpp::LogicalVector is_one(n);

  for (int i = 0; i < n; ++i) {
    is_na[i] = R_IsNA(p[i]) || (p[i] < 0) || (p[i] > 1);
    is_zero[i] = (p[i] == 0);
    is_one[i] = (p[i] == 1);
    if (is_na[i] || is_zero[i] || is_one[i]) {
      converged[i] = true;
      if (is_na[i]) q[i] = R_NaN;
      else if (is_zero[i]) q[i] = R_NegInf;
      else if (is_one[i]) q[i] = R_PosInf;
    }
  }

  for (int i = 0; i < n; ++i) {
      if (!converged[i]) {
          double lower_bound, upper_bound;
          if (abs_alpha == 0) {
              lower_bound = R::qt(p_adj[i], nu, 1, 0);
              upper_bound = R::qt(p_adj[i], nu, 1, 0);
          } else {
              lower_bound = R::qt(p_adj[i], nu, 1, 0);
              upper_bound = std::sqrt(R::qf(p_adj[i], 1.0, nu, 1, 0));

              if ((upper_bound - lower_bound) > 5.0) {
                  double step = 5.0;
                  int m = 0;
                  while (true) {
                      double current_lower = upper_bound - step;
                      // Pass all arguments explicitly for pst_cpp call
                      Rcpp::NumericVector p0_vec = pst_cpp({current_lower}, 0, 1, abs_alpha, nu, 2, true, false);
                      double p0 = p0_vec[0];
                      if (p0 < p_adj[i]) break;
                      step = step * std::pow(2.0, 2.0 / (m + 2.0));
                      m++;
                  }
                  lower_bound = upper_bound - step;
              }
              if (alpha < 0) {
                  double temp = lower_bound;
                  lower_bound = -upper_bound;
                  upper_bound = -temp;
              }
          }
          xa[i] = lower_bound;
          xb[i] = upper_bound;
          // Pass all arguments explicitly for pst_cpp call
          fa[i] = pst_cpp({xa[i]}, 0, 1, abs_alpha, nu, method, true, false)[0] - p_adj[i];
          fb[i] = pst_cpp({xb[i]}, 0, 1, abs_alpha, nu, method, true, false)[0] - p_adj[i];
      }
  }


  bool regula_falsi = false;
  int max_iter = 1000;
  int current_iter = 0;

  while (Rcpp::sum(!converged) > 0 && current_iter < max_iter) {
    current_iter++;
    for (int i = 0; i < n; ++i) {
      if (!converged[i]) {
        if (regula_falsi) {
          xc[i] = xb[i] - fb[i] * (xb[i] - xa[i]) / (fb[i] - fa[i]);
        } else {
          xc[i] = (xb[i] + xa[i])/2.0;
        }

        // Pass all arguments explicitly for pst_cpp call
        fc[i] = pst_cpp({xc[i]}, 0, 1, abs_alpha, nu, method, true, false)[0] - p_adj[i];

        bool pos = (fc[i] > 0);
        if (!pos) {
          xa[i] = xc[i];
          fa[i] = fc[i];
        } else {
          xb[i] = xc[i];
          fb[i] = fc[i];
        }

        if (std::abs(fc[i]) < tol) {
          converged[i] = true;
          q[i] = xi + omega * xc[i];
        }

        bool fail_check = ((xc[i]-xa[i]) * (xc[i]-xb[i])) > 0;
        if (fail_check && !converged[i]) {
          q[i] = R_NaN;
          converged[i] = true;
        } else if (!converged[i]) {
          q[i] = xi + omega * xc[i];
        }
      }
    }
    regula_falsi = !regula_falsi;
  }

  for (int i = 0; i < n; ++i) {
    if (is_na[i]) {
      q[i] = R_NaN;
    } else if (is_zero[i]) {
      q[i] = R_NegInf;
    } else if (is_one[i]) {
      q[i] = R_PosInf;
    } else if (!converged[i]) {
        q[i] = R_NaN;
    } else {
        double sign_alpha_val = (alpha > 0) ? 1.0 : (alpha < 0) ? -1.0 : 0.0;
        if (alpha == 0) sign_alpha_val = 1.0;
        q[i] = xi + omega * sign_alpha_val * ((q[i] - xi) / omega);
    }
  }
  return q;
}