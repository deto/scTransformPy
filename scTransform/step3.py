import numpy as np


def compute_residuals(umi, model_matrix, model_pars_fit, res_clip_range, min_var):
    model_pars_fit_sub = model_pars_fit[model_matrix.design_info.column_names]
    theta_fit = model_pars_fit['theta']

    mu = np.exp(model_pars_fit_sub.dot(model_matrix.T))
    mu.columns = umi.columns

    res = pearson_residual(umi, mu, theta_fit, min_var)

    # Clip values
    res[res < res_clip_range[0]] = res_clip_range[0]
    res[res > res_clip_range[1]] = res_clip_range[1]

    return res


def pearson_residual(y, mu, theta, min_var=float('-inf')):

    model_var = mu + (mu**2).divide(theta, axis=0)
    model_var[model_var < min_var] = min_var

    return (y - mu) / np.sqrt(model_var)
