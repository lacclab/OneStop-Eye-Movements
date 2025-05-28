import pandas as pd
from tqdm import tqdm

from onestop_paper_analyses.utils.mixed_effects_wrapper.julia_linear_mm import (
    jl,
    run_linear_mm,
)


def calc_mm_means(
    df: pd.DataFrame,
    outcome_variable: str,
    explanatory_variable: str,
    re_columns: list[str],
    link_dist=jl.Distributions.Normal(),
    mode: str = "subset_mean",
):
    """
    This function calculates the means of outcome_variable for each value of explanatory_variable

    Args:
        df: the data frame to run the model on
        outcome_variable: the outcome variable name
        explanatory_variable: the explanatory variable name
        re_cols: the random effects columns
        link_dist: the link distribution to use for the model
        mode: the mode to use for calculating the means. Can be either "subset_mean" or "sum_contrast_all_mean". See note
    Returns:
        means_df: a data frame with the means of outcome_variable for each value of explanatory_variable

    Note: - **subset_mean: If I want to compute the mean of reread==1,
            I take the reread column, create a contrast variable which is 1 where reread==1 and 0 otherwise.
            Then I fit the lmm with the formula *outcome ~ 1 + contrast + (1|sub) + (1|item)*.
            The final mean is the intercept + the slope**
            - **sum_contrast_all_mean:  If I want to compute the mean of reread==1,
            I take the subset of the dataset where reread==1,
            Then I fit the lmm with the formula *outcome ~ 1 + (1|sub) + (1|item).*
            The final mean is the intercept**
    """
    assert mode in ["subset_mean", "sum_contrast_all_mean"]
    # remove rows where explanatory_variable is nan or 'nan'
    df = df.loc[lambda x: ~pd.isnull(x[explanatory_variable])]
    df = df.loc[lambda x: x[explanatory_variable] != "nan"]

    u_vals = df[explanatory_variable].unique()
    # remove nan from u_vals
    u_vals = sorted(u_vals[~pd.isnull(u_vals)])

    print(f"u_vals: {u_vals}")

    means_dict = {}
    for val in tqdm(u_vals):
        if mode == "subset_mean":
            subset_df = df[df[explanatory_variable] == val]
            df_for_lmm = subset_df
            formula = f"{outcome_variable} ~ 1 " + " ".join(
                [f" + (1 | {x})" for x in re_columns]
            )
        if mode == "sum_contrast_all_mean":
            raise NotImplementedError
            # df["sum_contrast"] = df[explanatory_variable].apply(
            #     lambda x: 1 if x == val else -1
            # )
            # df_for_lmm = df
            # formula = f"{outcome_variable} ~ 1 + sum_contrast + (1 | subject_id) + (1 | unique_paragraph_id)"

        coeff_table, _ = run_linear_mm(
            df_for_lmm,
            outcome_variable,
            re_columns,
            formula,
            model_res_var_name="j_model",
            link_dist=link_dist,
            centralize_covariates=False,
            centralize_outcome=False,
            z_outcome=False,
            print_model_res=False,
        )
        mean = round(coeff_table["Coef."].sum(), 3)  # beta_0 + beta_1
        if mode == "subset_mean":
            p_val = coeff_table["Pr(>|z|)"].values[0]
        elif mode == "sum_contrast_all_mean":
            raise NotImplementedError
            # p_val = coeff_table["Pr(>|z|)"].values[
            #     1
            # ]  # Hypothesis: the mean of the outcome for this value of the explanatory variable is different from the grand mean
        se = round(
            (coeff_table["Std. Error"].sum() * 1.96),
            3,  # this is actually a single value since the formula supports only 'y ~ 1 + re_terms'
        )
        means_dict[val] = {"mean": mean, "p_val": p_val, "2se": se}
    means_df = (
        pd.DataFrame(means_dict)
        .T.reset_index()
        .rename(columns={"index": "explanatory_variable_value"})
    )
    # To means_df add a column which is always outcome_variable
    means_df["outcome_variable"] = outcome_variable
    means_df["explanatory_variable"] = explanatory_variable
    return means_df


def choose_link_dist(outcome_variable):
    if outcome_variable == "IA_SKIP":
        link_dist = jl.Distributions.Bernoulli()
    elif outcome_variable in [
        "IA_RUN_COUNT",
        "IA_FIXATION_COUNT",
        "IA_REGRESSION_OUT_FULL_COUNT",
    ]:
        link_dist = jl.Distributions.Poisson()
    else:
        link_dist = jl.Distributions.Normal()
    return link_dist


def calc_mm_means_for_all_outcomes(
    df: pd.DataFrame,
    explanatory_variable_list: list[str],
    re_columns: list[str],
    outcomes: list[str],
    mean_mode: str = "subset_mean",
):
    """
    This function calculates the means of outcome_variable for each value of explanatory_variable

    Args:
        df: the data frame to run the model on
        outcome_variable: the outcome variable name
        explanatory_variable: the explanatory variable name
        re_cols: the random effects columns
        link_dist: the link distribution to use for the model

    Returns:
        means_df: a data frame with the means of outcome_variable for each value of explanatory_variable
    """
    means_dfs_lst = []
    df["explanatory_variable"] = df[explanatory_variable_list].apply(
        lambda x: "__".join([str(y) for y in x]), axis=1
    )
    for outcome_variable in outcomes:
        df_m = df.copy()
        # in df_m create a column from explanatory_variable_list which is the concatenation of all explanatory variables devided by __

        # drop all rows where outcome_variable is nan
        # print the number of rows dropped
        print(f"outcome_variable: {outcome_variable}")
        print(f"df_m shape before dropping nan: {df_m.shape}")
        df_m = df_m[~pd.isnull(df_m[outcome_variable])]
        print(f"df_m shape after dropping nan: {df_m.shape}")

        means_dfs_lst.append(
            calc_mm_means(
                df_m,
                outcome_variable,
                "explanatory_variable",
                re_columns,
                mode=mean_mode,
            )
        )

    means_df = pd.concat(means_dfs_lst)

    # now split the column "explanatory_variable" to the different explanatory variable according to explanatory_variable_list
    for i, explanatory_variable in enumerate(explanatory_variable_list):
        means_df[explanatory_variable] = means_df["explanatory_variable_value"].apply(
            lambda x: x.split("__")[i]
        )
    # delete the column "explanatory_variable"
    means_df.drop(
        ["explanatory_variable_value", "explanatory_variable"], axis=1, inplace=True
    )

    return means_df
