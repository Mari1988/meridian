"""Defines model specification parameters for Meridian."""

from collections.abc import Sequence
import dataclasses

import numpy as np

from google3.ads.lift.mmm.modeling.research.contr_priors import constants
from google3.ads.lift.mmm.modeling.research.contr_priors.model import prior_distribution


__all__ = [
    "ModelSpec",
]


def _validate_roi_calibration_period(
    array: np.ndarray | None,
    array_name: str,
    media_channel_dim_name: str,
    paid_media_prior_type: str,
):
  """Validates the ROI calibration period array."""
  if array is None:
    return
  if paid_media_prior_type != constants.TREATMENT_PRIOR_TYPE_ROI:
    raise ValueError(
        f"The `{array_name}` should be `None` unless `paid_media_prior_type` is"
        f" '{constants.TREATMENT_PRIOR_TYPE_ROI}'."
    )
  if len(array.shape) != 2:
    raise ValueError(
        f"The shape of the `{array_name}` array {array.shape} should be"
        f" 2-dimensional (`n_media_times` x `{media_channel_dim_name}`)."
    )


@dataclasses.dataclass(frozen=True)
class ModelSpec:
  """Model specification parameters for Meridian.

  This class contains all model parameters that do not change between the runs
  of Meridian.

  Attributes:
    prior: A `PriorDistribution` object specifying the prior distribution of
      each set of model parameters. The distribution for a vector of parameters
      (for example, `alpha_m`) can be passed as either a scalar distribution or
      a vector distribution. If a scalar distribution is passed, it is broadcast
      to the actual shape of the parameter vector. See `paid_media_prior_type`
      for related details.
    media_effects_dist: A string to specify the distribution of media random
      effects across geos. This attribute is not used with a national-level
      model. Allowed values: `'normal'` or `'log_normal'`. Default:
      `'log_normal'`.
    hill_before_adstock: A boolean indicating whether to apply the Hill function
      before the Adstock function, instead of the default order of Adstock
      before Hill. This argument does not apply to RF channels. Default:
      `False`.
    max_lag: An integer indicating the maximum number of lag periods (≥ `0`) to
      include in the Adstock calculation. Can also be set to `None`, which is
      equivalent to infinite max lag. Default: `8`.
    unique_sigma_for_each_geo: A boolean indicating whether to use a unique
      residual variance for each geo. If `False`, then a single residual
      variance is used for all geos. Default: `False`.
    media_prior_type: A string or a list of strings to specify the prior type
      for the media coefficients. Allowed values of the string: `'roi'`,
      `'mroi'`, `'contribution`', `'coefficient'`. The `PriorDistribution`
      contains `roi_m`, `mroi_m`, `contribution_m`, and`beta_m`, but only one of
      these is used depending on the `paid_media_prior_type`. When
      `media_prior_type` is `'roi'`, the `PriorDistribution.roi_m` parameter is
      used to specify a prior on the ROI. When `media_prior_type` is `'mroi'`,
      the `PriorDistribution.mroi_m` parameter is used to specify a prior on the
      mROI. When `media_prior_type` is `'contribution'`, the
      `PriorDistribution.contribution_m` parameter is used to specify a prior on
      the contribution. When `media_prior_type` is `'coefficient'`, the
      `PriorDistribution.beta_m` parameter is used to specify a prior on the
      coefficient mean parameters. If a list of strings is passed as
      `media_prior_type`, then the list must have the same length as the number
      of paid media channels and each element specifies the prior type for the
      corresponding channel. Default: `'roi'`.
    rf_prior_type: A string or a list of strings to specify the prior type for
      the RF coefficients. Allowed values of the string: `'roi'`, `'mroi'`,
      `'contribution'`, `'coefficient'`. The `PriorDistribution` contains
      `roi_rf`, `mroi_rf`, `contribution_rf`, and`beta_rf`, but only one of
      these is used depending on the `rf_prior_type`. When `rf_prior_type` is
      `'roi'`, the `PriorDistribution.roi_rf` parameter is used to specify a
      prior on the ROI. When `rf_prior_type` is `'mroi'`, the
      `PriorDistribution.mroi_rf` parameter is used to specify a prior on the
      mROI. When `rf_prior_type` is `'contribution'`, the
      `PriorDistribution.contribution_rf` parameter is used to specify a prior
      on the contribution. When `rf_prior_type` is `'coefficient'`, the
      `PriorDistribution.beta_rf` parameter is used to specify a prior on the
      coefficient mean parameters. If a list of strings is passed as
      `rf_prior_type`, then the list must have the same length as the number of
      RF channels and each element specifies the prior type for the
      corresponding channel. Default: `'roi'`.
    roi_calibration_period: An optional boolean array of shape `(n_media_times,
      n_media_channels)` indicating the subset of `time` that the ROI value of
      the `roi_m` prior applies to. The ROI numerator is the incremental outcome
      generated during this time period, and the denominator is the spend during
      this time period. (Spend data by time period is required). If `None`, all
      times are used. Only used if `paid_media_prior_type` is `'roi'`.
      Default: `None`.
    rf_roi_calibration_period: An optional boolean array of shape
      `(n_media_times, n_rf_channels)` indicating the subset of `time` that the
      ROI value of the `roi_rf` prior applies to. The ROI numerator is the
      incremental outcome generated during this time period, and the denominator
      is the spend during this time period. (Spend data by time period is
      required). If `None`, all times are used. Only used if
      `paid_media_prior_type` is `'roi'`.
    organic_media_prior_type: A string or a list of strings to specify the prior
      type for the organic media coefficients. Allowed values: `'contribution'`,
      `'coefficient'`. The `PriorDistribution` contains `contribution_om`
      and`beta_om`, but only one of these is used depending on the
      `organic_media_prior_type`. When `organic_media_prior_type` is
      `'contribution'`, the `PriorDistribution.contribution_om` parameter is
      used to specify a prior on the contribution. When
      `organic_media_prior_type` is `'coefficient'`, the
      `PriorDistribution.beta_om` parameter is used to specify a prior on the
      coefficient mean parameters. If a list of strings is passed as
      `organic_media_prior_type`, then the list must have the same length as the
      number of organic media channels and each element specifies the prior type
      for the corresponding channel. Default: `'contribution'`.
    organic_rf_prior_type: A string or a list of strings to specify the prior
      type for the organic RF coefficients. Allowed values: `'contribution'`,
      `'coefficient'`. The `PriorDistribution` contains `contribution_orf`,
      and`beta_orf`, but only one of these is used depending on the
      `organic_rf_prior_type`. When `organic_rf_prior_type` is `'contribution'`,
      the `PriorDistribution.contribution_orf` parameter is used to specify a
      prior on the contribution. When `organic_rf_prior_type` is
      `'coefficient'`, the `PriorDistribution.beta_orf` parameter is used to
      specify a prior on the coefficient mean parameters. If a list of strings
      is passed as `organic_rf_prior_type`, then the list must have the same
      length as the number of organic RF channels and each element specifies the
      prior type for the corresponding channel. Default: `'contribution'`.
    non_media_treatments_prior_type: A string to specify the prior type for the
      non-media treatment coefficients. Allowed values: `'contribution'`,
      `'coefficient'`. The `PriorDistribution` contains `contribution_n`
      and`gamma_n`, but only one of these is used depending on the
      `non_media_prior_type`.When `non_media_prior_type` is `'contribution'`,
      the `PriorDistribution.contribution_n` parameter is used to specify a
      prior on the contribution. When `non_media_prior_type` is `'coefficient'`,
      the `PriorDistribution.gamma_n` parameter is used to specify a prior on
      the coefficient mean parameters. Default: `'contribution'`.
    non_media_treatments_baseline_values: Optional list of shape
      (n_non_media_channels,). Each element is either a float (which means that
      the fixed value will be used as baseline for the given channel) or one of
      the strings "min" or "max" (which mean that the global minimum or maximum
      value will be used as baseline for the scaled values of the given
      non_media treatments channel). If None, the minimum value is used as
      baseline for each non_media treatments channel.
    knots: An optional integer or list of integers indicating the knots used to
      estimate time effects. When `knots` is a list of integers, the knot
      locations are provided by that list. Zero corresponds to a knot at the
      first time period, one corresponds to a knot at the second time period,
      ..., and `(n_times - 1)` corresponds to a knot at the last time period).
      Typically, we recommend including knots at `0` and `(n_times - 1)`, but
      this is not required. When `knots` is an integer, then there are knots
      with locations equally spaced across the time periods, (including knots at
      zero and `(n_times - 1)`. When `knots` is` 1`, there is a single common
      regression coefficient used for all time periods. If `knots` is set to
      `None`, then the numbers of knots used is equal to the number of time
      periods in the case of a geo model. This is equivalent to each time period
      having its own regression coefficient. If `knots` is set to `None` in the
      case of a national model, then the number of knots used is `1`. Default:
      `None`.
    baseline_geo: An optional integer or a string for the baseline geo. The
      baseline geo is treated as the reference geo in the dummy encoding of
      geos. Non-baseline geos have a corresponding `tau_g` indicator variable,
      meaning that they have a higher prior variance than the baseline geo. When
      set to `None`, the geo with the biggest population is used as the
      baseline. Default: `None`.
    holdout_id: Optional boolean tensor of dimensions `(n_geos, n_times)` for a
      geo-level model or `(n_times,)` for a national model, indicating which
      observations are part of the holdout sample, which are excluded from the
      training sample. Only KPI data is excluded from the training sample. Media
      data is still included as it can affect Adstock for subsequent weeks. If
      "ROI priors" are used, then the `roi_m` parameters correspond to the ROI
      of all geos and times, even those in the holdout sample.
    control_population_scaling_id: An optional boolean tensor of dimension
      `(n_controls,)` indicating the control variables for which the control
      value will be scaled by population. If `None`, no control variables are
      scaled by population. Default: `None`.
    non_media_population_scaling_id: An optional boolean tensor of dimension
      `(n_non_media_channels,)` indicating the non-media variables for which the
      non-media value will be scaled by population. If `None`, then no non-media
      variables are scaled by population. Default: `None`.
  """

  prior: prior_distribution.PriorDistribution = dataclasses.field(
      default_factory=prior_distribution.PriorDistribution,
  )
  media_effects_dist: str = constants.MEDIA_EFFECTS_LOG_NORMAL
  hill_before_adstock: bool = False
  max_lag: int | None = 8
  unique_sigma_for_each_geo: bool = False
  media_prior_type: str | Sequence[str] = constants.TREATMENT_PRIOR_TYPE_ROI
  rf_prior_type: str | Sequence[str] = constants.TREATMENT_PRIOR_TYPE_ROI
  roi_calibration_period: np.ndarray | None = None
  rf_roi_calibration_period: np.ndarray | None = None
  organic_media_prior_type: str | Sequence[str] = (
      constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION
  )
  organic_rf_prior_type: str | Sequence[str] = (
      constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION
  )
  non_media_treatments_prior_type: str | Sequence[str] = (
      constants.TREATMENT_PRIOR_TYPE_CONTRIBUTION
  )
  non_media_treatments_baseline_values: Sequence[float | str] | None = None
  knots: int | list[int] | None = None
  baseline_geo: int | str | None = None
  holdout_id: np.ndarray | None = None
  control_population_scaling_id: np.ndarray | None = None
  non_media_population_scaling_id: np.ndarray | None = None

  def __post_init__(self):
    if self.media_effects_dist not in constants.MEDIA_EFFECTS_DISTRIBUTIONS:
      raise ValueError(
          f"The `media_effects_dist` parameter '{self.media_effects_dist}' must"
          f" be one of {sorted(list(constants.MEDIA_EFFECTS_DISTRIBUTIONS))}."
      )
    if self.media_prior_type not in constants.TREATMENT_PRIOR_TYPES:
      raise ValueError(
          "The `media_prior_type` parameter"
          f" '{self.media_prior_type}' must be one of"
          f" {sorted(list(constants.TREATMENT_PRIOR_TYPES))}."
      )
    if self.rf_prior_type not in constants.TREATMENT_PRIOR_TYPES:
      raise ValueError(
          "The `rf_prior_type` parameter"
          f" '{self.rf_prior_type}' must be one of"
          f" {sorted(list(constants.TREATMENT_PRIOR_TYPES))}."
      )
    if (
        self.organic_media_prior_type
        not in constants.NON_PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `organic_media_prior_type` parameter"
          f" '{self.organic_media_prior_type}' must be one of"
          f" {sorted(list(constants.NON_PAID_TREATMENT_PRIOR_TYPES))}."
      )
    if (
        self.organic_rf_prior_type
        not in constants.NON_PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `organic_rf_prior_type` parameter"
          f" '{self.organic_rf_prior_type}' must be one of"
          f" {sorted(list(constants.NON_PAID_TREATMENT_PRIOR_TYPES))}."
      )
    if self.non_media_treatments_prior_type not in (
        constants.NON_PAID_TREATMENT_PRIOR_TYPES
    ):
      raise ValueError(
          "The `non_media_treatments_prior_type` parameter"
          f" '{self.non_media_treatments_prior_type}' must be one of"
          f" {sorted(list(constants.NON_PAID_TREATMENT_PRIOR_TYPES))}."
      )
    _validate_roi_calibration_period(
        self.roi_calibration_period,
        "roi_calibration_period",
        "n_media_channels",
        self.media_prior_type,
    )
    _validate_roi_calibration_period(
        self.rf_roi_calibration_period,
        "rf_roi_calibration_period",
        "n_rf_channels",
        self.media_prior_type,
    )

    # Validate knots.
    if isinstance(self.knots, list) and not self.knots:
      raise ValueError("The `knots` parameter cannot be an empty list.")
    if isinstance(self.knots, int) and self.knots == 0:
      raise ValueError("The `knots` parameter cannot be zero.")
