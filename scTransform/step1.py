"""
Steps of the algorithm:

0. Subset to 2k genes using density-based weightings
1. For each gene, fit a Poisson model to estimate B coefficients
   - Then estimate the theta parameter via ML
2. Regularize parameters (offset and slope) via kernal function
    - Find gene means (geometric mean)
    - Choose kernel bandwidth
3. Transform UMI counts into Pearson residuals
    - Clip residuals to maximum value of sqrt N_Cells
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import digamma, polygamma
from tqdm import tqdm


def estimate_parameters_all_genes(umi, model_matrix):
    """
    Estimate the model parameters for every gene
    """
    results = []
    for gene, row in tqdm(umi.iterrows()):

        gene_counts = row.values.reshape((-1, 1))
        coefs, theta = estimate_parameters(gene_counts, model_matrix)

        results.append(
            [gene, theta] + coefs.tolist()
        )

    model_pars = pd.DataFrame(
        results, columns=['gene', 'theta'] + model_matrix.design_info.column_names
    ).set_index("gene")

    return model_pars


def estimate_parameters(gene_counts, model_matrix):
    """
    Estimate model coefficients and theta given gene_counts

    First estimate coefficients using Poisson distribution, then
    use ML to estimate theta
    """

    coefs, mu = estimate_parameters_poisson(gene_counts, model_matrix)

    theta = estimate_theta_ML(gene_counts, mu)

    return coefs, theta


def estimate_parameters_poisson(gene_counts, model_matrix):
    """
    estimate parameters from gene_counts assuming Poisson distribution
    """

    model = sm.GLM(gene_counts, model_matrix, family=sm.families.Poisson())
    res = model.fit()
    return res.params, res.mu


def trigamma(x):
    return polygamma(n=1, x=x)


def estimate_theta_ML(y, mu):
    """
    Find the Maximum Likelihood estimate of theta given response (y) and
    fitted response (mu)
    """

    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]

    if mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu[:, 0]

    assert y.ndim == 1
    assert mu.ndim == 1

    n = len(y)
    limit = 10
    eps = sys.float_info.epsilon**.25

    def score(n, th, mu, y):
        r = digamma(th + y) - digamma(th) + np.log(th) + 1 - np.log(th + mu) - (y + th) / (mu + th)
        return r.sum()

    def info(n, th, mu, y):
        r = -1 * trigamma(th + y) + trigamma(th) - 1 / th + 2 / (mu + th) - \
            (y + th) / (mu + th)**2
        return r.sum()

    t0 = n / np.sum((y / mu - 1)**2)
    it = 1
    delta = 1
    while it < limit:
        t0 = abs(t0)

        i = info(n, t0, mu, y)
        delta = score(n, t0, mu, y) / i

        t0 = t0 + delta

        if abs(delta) <= eps:
            break

        it = it + 1

    if t0 < 0:
        t0 = 0

    # if it == limit:
    #     print("Max iterations reached")

    return t0
