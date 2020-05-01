import sys
import numpy as np
import pandas as pd
from numba import njit

M_1_SQRT_2PI = 1 / np.sqrt(2 * np.pi)


@njit
def bw_phi4(sn, sd, cnt, sh):

    sum = 0.0
    nbin = len(cnt)
    x = cnt  # Integer
    h = sh  # Real
    d = sd  # Real
    n = sn  # Integer
    DELMAX = 1000

    for i in range(nbin):
        delta = i * d / h;
        delta *= delta

        if delta >= DELMAX:
            break

        term = np.exp(-1 * delta / 2) * (delta * delta - 6 * delta + 3);
        sum = sum + term * x[i]

    sum = 2 * sum + n * 3
    u = sum / (n * (n - 1) * np.power(h, 5) * np.sqrt(2 * np.pi))

    return u


@njit
def bw_phi6(n, d, cnt, h):

    sum = 0.0

    nbin = len(cnt)
    x = cnt
    DELMAX = 1000

    for i in range(nbin):
        delta = i * d / h
        delta *= delta

        if delta >= DELMAX:
            break

        term = np.exp(-1 * delta / 2) * \
            (delta * delta * delta - 15 * delta * delta + 45 * delta - 15)

        sum += term * x[i]

    sum = 2.0 * sum - 15.0 * n
    u = sum / (n * (n - 1) * np.power(h, 7.0)) * M_1_SQRT_2PI

    return u


@njit
def bw_den(nb, x):

    n = len(x)

    xmin = x.min()
    xmax = np.max(x)

    rang = (xmax - xmin) * 1.01
    dd = rang / nb

    cnt = np.zeros(nb)

    for i in range(1, n):
        ii = int(x[i] / dd)

        for j in range(0, i):
            jj = int(x[j] / dd)
            cnt[abs(ii - jj)] += 1.0

    return dd, cnt


def bwSJ(x, nb=1000, lower=None, upper=None, method="ste"):

    n = len(x)

    d, cnt = bw_den(nb, x)

    @njit
    def SDh(h):
        return bw_phi4(n, d, cnt, h)

    @njit
    def TDh(h):
        return bw_phi6(n, d, cnt, h)

    IQRx = np.quantile(x, .75) - np.quantile(x, .25)
    scale = min(x.std(ddof=1), IQRx / 1.349)

    a = 1.24 * scale * n**(-1 / 7)
    b = 1.23 * scale * n**(-1 / 9)

    c1 = 1 / (2 * np.sqrt(np.pi) * n)

    TD = -1 * TDh(b)

    if (np.isinf(TD) or TD <= 0):
        raise Exception("sample is too sparse to find TD")

    if (method == "dpi"):
        res = (c1 / SDh((2.394 / (n * TD))**(1 / 7))) ** (1 / 5)
    else:

        bndMiss = False
        if lower is None or upper is None:
            bndMiss = True
            hmax = 1.144 * scale * n**(-1 / 5)
            lower = 0.1 * hmax
            upper = hmax

        tol = 0.1 * lower

        alph2 = 1.357 * (SDh(a) / TD)**(1 / 7)
        if np.isinf(alph2):
            raise Exception("sample is too sparse to find alph2")

        itry = 1

        @njit
        def fSD(h):
            return (
                c1 / SDh(alph2 * h**(5 / 7))
            )**(1 / 5) - h

        while fSD(lower) * fSD(upper) > 0:
            if itry > 99 or not bndMiss:
                raise Exception("no solution in the specified range of bandwidths")

            if itry % 2 == 1:
                upper = upper * 1.2
            else:
                lower = lower / 1.2

            itry = itry + 1

        res = uniroot(fSD, lower, upper, tol=tol)

    return res


@njit
def uniroot(f, lower, upper, tol):

    f_lower = f(lower)
    f_upper = f(upper)

    if f(lower) * f(upper) > 0:
        raise Exception("f() values at end points not of opposite sign")

    mid = (lower + upper) / 2
    f_mid = f(mid)

    if f_mid * f_lower < 0:
        return uniroot_inner(f, lower, mid, f_lower, f_mid, tol)
    else:
        return uniroot_inner(f, mid, upper, f_mid, f_upper, tol)


@njit
def uniroot_inner(f, lower, upper, f_lower, f_upper, tol):

    mid = (lower + upper) / 2

    f_mid = f(mid)

    if f_mid * f_lower < 0:
        new_mid = (lower + mid) / 2

        if abs(new_mid - mid) < tol:  # odd way to do it, but that's how it's described in R
            return new_mid

        return uniroot_inner(f, lower, mid, f_lower, f_mid, tol)
    else:
        new_mid = (mid + upper) / 2

        if abs(new_mid - mid) < tol:
            return new_mid

        return uniroot_inner(f, mid, upper, f_mid, f_upper, tol)


@njit
def ksmooth(x, y, xp, bw):
    """
    Applies the gaussian kernel on values in x/y at points xp
    Derived from ksmooth.c in R (includes BDRksmooth)
    """

    # Values are not sorted, so we must sort then unsort later
    ii = np.argsort(x)
    iip = np.argsort(xp)

    x = x[ii]
    y = y[ii]
    xp = xp[iip]

    n = len(x)
    assert len(x) == len(y)
    n_p = len(xp)
    yp = np.zeros(n_p)

    imin = 0;
    cutoff = 0.0

    bw *= 0.3706506
    cutoff = 4 * bw

    while x[imin] < xp[0] - cutoff and imin < n:
        imin += 1

    for j in range(n_p):
        num = 0.0
        den = 0.0

        x0 = xp[j]

        for i in range(imin, n):
            if x[i] < x0 - cutoff:
                imin = i;
            else:
                if x[i] > x0 + cutoff:
                    break;

                w = np.exp(
                    -0.5 * (abs(x[i] - x0) / bw) ** 2
                )
                num += w * y[i];
                den += w;

        if den > 0:
            yp[j] = num / den
        else:
            yp[j] = np.nan

    xp[iip] = xp
    yp[iip] = yp
    return xp, yp


def reg_model_pars(
        model_pars, genes_log_gmean_step1, genes_log_gmean, cell_attr,
        batch_var, bw_adjust, gmean_eps):

    model_pars = model_pars.copy()
    model_pars['theta'] = np.log10(model_pars['theta'])

    print("Detecting outliers...")
    outliers_all = pd.DataFrame(False, index=model_pars.index, columns=model_pars.columns)
    for col in model_pars.columns:
        col_outliers = is_outlier(model_pars[col].values, genes_log_gmean_step1.values)
        outliers_all[col] = col_outliers

    outliers = outliers_all.any(axis=1).values
    if outliers.sum() > 0:
        print("Found {} outliers - ignoring for the fitting/regularization step".format(outliers.sum()))

        model_pars = model_pars.loc[~outliers]
        genes_log_gmean_step1 = genes_log_gmean_step1[~outliers]

    print("Computing Bandwidth...")
    bw = bwSJ(genes_log_gmean_step1.values) * bw_adjust
    print("  Bandwidth: {:.6f}".format(bw))

    x_points = genes_log_gmean.copy()
    x_points[x_points < genes_log_gmean_step1.min()] = genes_log_gmean_step1.min()
    x_points[x_points > genes_log_gmean_step1.max()] = genes_log_gmean_step1.max()

    model_pars_fit = pd.DataFrame(
        0.0, index=genes_log_gmean.index, columns=model_pars.columns
    )

    print("Regularizing Parameters...")
    if batch_var is None:

        for col in model_pars.columns:

            print("   {}...".format(col))
            model_pars_fit[col] = ksmooth(
                x=genes_log_gmean_step1.values, y=model_pars[col].values,
                xp=x_points.values, bw=bw)[1]

    else:

        # fit / regularize theta

        model_pars_fit["theta"] = ksmooth(
            x=genes_log_gmean_step1.values, y=model_pars["theta"].values,
            xp=x_points.values, bw=bw)[1]

        raise NotImplementedError("batch_var is not None - batch handling not implemented")

    model_pars_fit['theta'] = 10**model_pars_fit['theta']

    return model_pars_fit, outliers_all


def is_outlier(y, x, th=10):
    """
    Identify outliers
    Form bins using x values
    Bin y values.  Scale values (robust) within bin.
    Mark values that are > 10 after scaling as outliers.
    Uses two different break schemes
    """

    xmin = x.min()
    xmax = x.max()
    bin_width = (xmax - xmin) * bwSJ(x) / 2

    eps = np.finfo('float').eps * 10

    breaks1 = np.arange(xmin - eps, xmax + bin_width, bin_width)
    breaks2 = np.arange(xmin - eps - bin_width / 2, xmax + bin_width, bin_width)

    score1 = robust_scale_binned(y, x, breaks1)
    score2 = robust_scale_binned(y, x, breaks2)

    score = np.minimum(np.abs(score1), np.abs(score2))

    return score > th


def robust_scale_binned(y, x, breaks):

    bins = pd.cut(x, breaks)
    score = np.zeros(len(y))

    for b in bins.categories:

        ii = bins == b
        y_bin = y[ii]

        if len(y_bin) > 0:
            y_scaled = robust_scale(y_bin)
            score[ii] = y_scaled

    return score


def robust_scale(x):
    medx = np.median(x)
    madx = np.median(np.abs(x - medx)) * 1.4826

    return (x - medx) / (madx + np.finfo('float').eps)
