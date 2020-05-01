import numpy as np
import pandas as pd
import patsy

from . import step1
from . import step2
from . import step3


def vst(
    umi,
    cell_attr=None,
    latent_var=["log_umi"],
    batch_var=None,
    latent_var_nonreg=None,
    n_genes=None,
    n_cells=None,
    method="poisson",
    do_regularize=True,
    res_clip_range=None,
    bin_size=None,
    min_cells=5,
    residual_type="pearson",
    return_cell_attr=False,
    return_gene_attr=False,
    return_corrected_umi=False,
    min_variance=float("-inf"),
    bw_adjust=3,
    gmean_eps=1,
    theta_given=None,
):

    if batch_var is not None:
        raise NotImplementedError("batch_var is not supported")

    if latent_var_nonreg is not None:
        raise NotImplementedError("latent_var_nonreg is not supported")

    if n_genes is not None:
        raise NotImplementedError("n_genes is not supported")

    if n_cells is not None:
        raise NotImplementedError("n_cells is not supported")

    if method != "poisson":
        raise NotImplementedError("Only method=\"poisson\" is supported")

    if do_regularize is False:
        raise NotImplementedError("Only do_regularize=True is supported")

    if res_clip_range is None:
        res_clip_range = (-1 * np.sqrt(umi.shape[1]), np.sqrt(umi.shape[1]))

    if bin_size is not None:
        raise NotImplementedError("bin_size is not supported")

    if residual_type != "pearson":
        raise NotImplementedError("Only residual_type=\"pearson\" supported")

    if return_cell_attr is not False:
        raise NotImplementedError("return_cell_attr is not supported")

    if return_gene_attr is not False:
        raise NotImplementedError("return_gene_attr is not supported")

    if return_corrected_umi is not False:
        raise NotImplementedError("return_corrected_umi is not supported")

    if theta_given is not None:
        raise NotImplementedError("theta_given is not supported")

    # Populate the cell attr with known attributes

    if cell_attr is None:
        cell_attr = pd.DataFrame(index=umi.columns)
    else:
        assert (cell_attr.index == umi.columns).all(), "cell_attr rows must match umi columns"
        cell_attr = cell_attr.copy()

    if "umi" in latent_var and "umi" not in cell_attr.columns:
        cell_attr["umi"] = umi.sum(axis=0)

    if "gene" in latent_var and "gene" not in cell_attr.columns:
        cell_attr["gene"] = (umi > 0).sum(axis=0)

    if "log_umi" in latent_var and "log_umi" not in cell_attr.columns:
        cell_attr["log_umi"] = np.log10(umi.sum(axis=0))

    if "log_gene" in latent_var and "log_gene" not in cell_attr.columns:
        cell_attr["log_gene"] = np.log10((umi > 0).sum(axis=0))

    if "umi_per_gene" in latent_var and "umi_per_gene" not in cell_attr.columns:
        cell_attr["umi_per_gene"] = umi.sum(axis=0) / (umi > 0).sum(axis=0)

    if "log_umi_per_gene" in latent_var and "log_umi_per_gene" not in cell_attr.columns:
        cell_attr["log_umi_per_gene"] = np.log10(umi.sum(axis=0) / (umi > 0).sum(axis=0))

    assert all([x in cell_attr.columns for x in latent_var]), "not all latent_var in cell_attr"

    # Subset genes using min_cells
    genes_cell_count = (umi > 0).sum(axis=1)
    umi = umi.loc[genes_cell_count >= min_cells, ]

    genes_log_gmean = np.log10(
        np.exp(np.log(umi + gmean_eps).mean(axis=1)) - gmean_eps
    )

    # Create model matrix
    model_matrix = patsy.dmatrix(" + ".join(latent_var), cell_attr)

    # Substitute cells for step1
    genes_log_gmean_step1 = genes_log_gmean

    model_pars = step1.estimate_parameters_all_genes(umi, model_matrix)

    model_pars_fit, outliers = step2.reg_model_pars(
        model_pars, genes_log_gmean_step1, genes_log_gmean, cell_attr,
        batch_var, umi, bw_adjust, gmean_eps)

    residuals = step3.compute_residuals(
        umi, model_matrix, model_pars_fit, res_clip_range, min_variance)

    return residuals
