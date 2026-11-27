"""Statistical and machine-learning models for E1-E3 (Section V-B), fitted on the aggregated tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Weather terms (reference: dry hour, visibility >= 5 km, temperature 10-20 C)
WEATHER_TERMS = ["rain", "snow", "vis_lt1", "vis_1to5",
                 "C(temp_bin, Treatment('10_20'))", "snow_on_ground", "np.log1p(hours_since_precip)"]
CALENDAR_TERMS = ["C(hour)", "C(dow)", "C(month)", "C(year)", "holiday", "school_day", "daylight", "covid"]


def formula(y: str, terms: list[str]) -> str:
    return f"{y} ~ " + " + ".join(terms)


# ----------------------------------------------------------------------------- count models
def fit_count_models(df: pd.DataFrame, y: str, weather=WEATHER_TERMS, calendar=CALENDAR_TERMS, offset=None):
    """Poisson and negative binomial (NB2) with weather + calendar terms, and NB with calendar only.

    Returns a dict with the fitted results, the dispersion check and the likelihood-ratio tests.
    """
    off = None if offset is None else np.log(df[offset])
    f_full, f_time = formula(y, weather + calendar), formula(y, calendar)
    pois = smf.glm(f_full, df, family=sm.families.Poisson(), offset=off).fit()
    pois_time = smf.glm(f_time, df, family=sm.families.Poisson(), offset=off).fit()
    nb = fit_nb(f_full, df, off, pois)
    nb_time = fit_nb(f_time, df, off, pois_time)
    lr_weather = 2 * (nb.llf - nb_time.llf)
    df_weather = nb.df_model - nb_time.df_model
    lr_alpha = 2 * (nb.llf - pois.llf)            # boundary test: p-value halved
    return {
        "poisson": pois, "nb": nb, "nb_time": nb_time,
        "pearson_dispersion": float(pois.pearson_chi2 / pois.df_resid),
        "alpha": float(nb.params["alpha"]),
        "aic": {"poisson": pois.aic, "nb": nb.aic, "nb_time": nb_time.aic},
        "lr_nb_vs_poisson": {"stat": lr_alpha, "p": 0.5 * stats.chi2.sf(lr_alpha, 1)},
        "lr_weather": {"stat": lr_weather, "df": df_weather, "p": stats.chi2.sf(lr_weather, df_weather)},
    }


def fit_nb(f, df, offset=None, poisson_res=None):
    """NB2 by maximum likelihood, started from the Poisson estimates (Newton from the default start
    values can fail on ~100k rows with many dummies)."""
    if poisson_res is None:
        poisson_res = smf.glm(f, df, family=sm.families.Poisson(), offset=offset).fit()
    try:
        res = smf.negativebinomial(f, df, offset=offset).fit(
            start_params=np.append(poisson_res.params.to_numpy(), 0.1), method="newton", maxiter=100, disp=0)
        if not res.mle_retvals.get("converged", False) or not np.isfinite(res.params).all():
            start = res.params.to_numpy() if np.isfinite(res.params).all() else np.append(poisson_res.params.to_numpy(), 0.1)
            res = smf.negativebinomial(f, df, offset=offset).fit(start_params=start, method="bfgs", maxiter=1000, disp=0)
    except np.linalg.LinAlgError:
        res = None
    if res is None or not np.isfinite(res.params).all() or not np.isfinite(res.bse).all():
        # Very sparse outcomes: fall back to Poisson with robust (sandwich) standard errors.
        import warnings
        warnings.warn("negative binomial did not converge; using Poisson with robust standard errors", RuntimeWarning)
        res = smf.glm(f, df, family=sm.families.Poisson(), offset=offset).fit(cov_type="HC0")
    return res


def ratio_table(res, prefix_filter=None, exclude_prefixes=("C(hour", "C(dow", "C(month", "C(year", "Intercept", "alpha")):
    """exp(coefficient) with 95% CI: incidence rate ratios (count models) or odds ratios (logit)."""
    ci = res.conf_int()
    t = pd.DataFrame({"estimate": np.exp(res.params), "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]),
                      "p_value": res.pvalues})
    keep = [i for i in t.index if not i.startswith(exclude_prefixes)]
    if prefix_filter:
        keep = [i for i in keep if i.startswith(prefix_filter)]
    return t.loc[keep]


def hour_of_week_baseline(train: pd.DataFrame, test: pd.DataFrame, y: str) -> np.ndarray:
    means = train.groupby("hour_of_week")[y].mean()
    return test["hour_of_week"].map(means).to_numpy()


def mae_rmse(y, yhat) -> dict:
    e = np.asarray(y) - np.asarray(yhat)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2)))}


def poisson_rate_ci(events, exposure, per=100_000, alpha=0.05):
    """Exact (Garwood) confidence interval for a rate events/exposure."""
    events, exposure = np.asarray(events, float), np.asarray(exposure, float)
    lo = np.where(events > 0, stats.chi2.ppf(alpha / 2, 2 * events) / 2, 0.0)
    hi = stats.chi2.ppf(1 - alpha / 2, 2 * (events + 1)) / 2
    return per * events / exposure, per * lo / exposure, per * hi / exposure


# ----------------------------------------------------------------------------- severity
SEVERITY_NUMERIC = ["rain", "snow", "vis_lt1", "vis_1to5", "snow_on_ground", "holiday", "school_day",
                    "daylight", "covid", "automobile", "motorcycle", "passenger", "bicycle", "pedestrian"]
SEVERITY_TIME_ONLY = ["holiday", "school_day", "daylight", "covid"]
SEVERITY_CATEG = ["hour", "dow", "month", "temp_bin"]
SEVERITY_TIME_CATEG = ["hour", "dow", "month"]


def severity_design(df: pd.DataFrame, numeric=SEVERITY_NUMERIC, categ=SEVERITY_CATEG, columns=None):
    """One-hot design matrix for the severity models (same columns for train/val/test)."""
    X = pd.concat([df[numeric].astype("float32"),
                   pd.get_dummies(df[categ].astype(str), drop_first=True, dtype="float32")], axis=1)
    X["log_hours_since_precip"] = np.log1p(df["hours_since_precip"]).astype("float32")
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0.0)
    return X


def split_by_year(df: pd.DataFrame, cfg) -> dict:
    s = cfg["split"]
    return {k: df[df["year"].between(*s[k])] for k in ("train", "val", "test")}


def classification_metrics(y, p, threshold) -> dict:
    from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                                 roc_auc_score)
    yhat = (p >= threshold).astype(int)
    return {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat),
            "F1": f1_score(y, yhat), "threshold": threshold, "positive_rate": float(np.mean(y))}


def best_f1_threshold(y, p) -> float:
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    return float(th[np.nanargmax(f1[:-1])])


def train_torch_logistic(X_tr, y_tr, X_va, y_va, cfg, seed=42):
    """Logistic regression in PyTorch: one linear layer + BCEWithLogitsLoss with pos_weight for the
    class imbalance, Adam optimiser, early stopping on validation PR-AUC."""
    import torch
    from sklearn.metrics import average_precision_score

    t = cfg["severity"]["torch"]
    torch.manual_seed(seed)
    mu, sd = X_tr.mean(0), X_tr.std(0).replace(0, 1)
    def prep(X):
        return torch.tensor(((X - mu) / sd).to_numpy(np.float32))
    xt, xv = prep(X_tr), prep(X_va)
    yt = torch.tensor(np.asarray(y_tr, np.float32))
    model = torch.nn.Linear(xt.shape[1], 1)
    pos_weight = torch.tensor([(1 - yt.mean()) / yt.mean()])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    best, best_state, history = -1, None, []
    for epoch in range(t["epochs"]):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), t["batch_size"]):
            idx = perm[i:i + t["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv).squeeze(1)).numpy()
        ap = average_precision_score(y_va, pv)
        history.append({"epoch": epoch + 1, "train_loss": total / len(xt), "val_pr_auc": ap})
        if ap > best:
            best, best_state = ap, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(model(prep(X)).squeeze(1)).numpy()
    return model, predict, pd.DataFrame(history)


# ----------------------------------------------------------------------------- fixed-effects speed model
def twoway_demean(df: pd.DataFrame, cols: list[str], g1: str, g2: str, tol=1e-8, max_iter=100) -> pd.DataFrame:
    """Remove sign and hour-of-week effects by alternating projections (handles unbalanced panels)."""
    out = df[cols].astype("float64").copy()
    for _ in range(max_iter):
        before = out.to_numpy().copy()
        out -= out.groupby(df[g1]).transform("mean")
        out -= out.groupby(df[g2]).transform("mean")
        if np.max(np.abs(out.to_numpy() - before)) < tol:
            break
    return out


def fe_regression(df: pd.DataFrame, y: str, x: list[str], sign="sign_id", time="hour_of_week"):
    """Linear model with separate intercepts per sign and per hour of week; SEs clustered by sign."""
    d = df[[y] + x + [sign, time]].dropna()
    dm = twoway_demean(d, [y] + x, sign, time)
    res = sm.OLS(dm[y], dm[x]).fit(cov_type="cluster", cov_kwds={"groups": d[sign].to_numpy()})
    n_fe = d[sign].nunique() + d[time].nunique() - 1
    # small-sample correction for the absorbed fixed effects
    k = len(x)
    scale = (len(d) - k) / (len(d) - k - n_fe)
    se = res.bse * np.sqrt(scale)
    ci_l, ci_h = res.params - 1.96 * se, res.params + 1.96 * se
    return pd.DataFrame({"estimate": res.params, "se": se, "ci_low": ci_l, "ci_high": ci_h,
                         "p_value": 2 * stats.norm.sf(np.abs(res.params / se))}), {"n_obs": len(d), "n_signs": d[sign].nunique()}


def hypothesis_verdict(est, lo, hi, null=1.0, expected="greater") -> str:
    """Supported if the 95% CI excludes the null value in the expected direction (Section VII-D)."""
    if expected == "greater":
        return "supported" if lo > null else ("opposite direction" if hi < null else "not supported")
    return "supported" if hi < null else ("opposite direction" if lo > null else "not supported")
