# Wind-direction reference convention

The SBGR runway/wind calculations in this repository use a **true-north reference frame**.

METAR and TAF wind directions are meteorological directions referenced to true north. In the historical preprocessing, runway-relative wind components and the wind-availability state are computed against the SBGR runway axis expressed in the same reference frame:

- runway 10 direction: **73° true**;
- reciprocal runway 28 direction: **253° true**.

The 73°/253° values are rounded true runway-axis bearings used by the study preprocessing. They are **not** the runway designator values interpreted as 100°/280°, and they are not magnetic headings. Therefore the wind/runway trigonometric calculations compare angles expressed in the same true-north reference frame.

This convention is inherited from the runway-capacity preprocessing used by the preceding SBGR study and applies to files such as `R/generate_sbgr_data.R` and `R/forecast_data_decoder_mod.R`. The tactical reinforcement-learning environments consume the resulting processed wind state and do not apply a second magnetic/true conversion.

A later SBGR runway-configuration forecasting study uses more precise true bearings of approximately 73.69°/253.67°. This repository intentionally preserves the rounded 73°/253° values used by its historical preprocessing; this documentation does not modify code, processed data, retained policies, or published results.
