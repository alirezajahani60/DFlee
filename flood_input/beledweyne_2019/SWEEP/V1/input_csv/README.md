# Hydraulic inputs for V1

Put here: flood_level.csv (location flood levels from HEC-RAS), link_states.csv (link states per travel mode), produced with `python -m flee.preprocessing.hecras2dflee` from the 2019 HEC-RAS run (and `hecras2dflee.gauge_warnings` for warnings.csv).

They are not in the repository yet: they derive from the hydraulic model of Ibrahim et al. (2025) and are added once its authors agree. FabSim copies this folder over the case's input_csv, so these files replace the V0 flood levels.
