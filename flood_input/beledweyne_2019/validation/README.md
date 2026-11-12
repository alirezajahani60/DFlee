# Validation data

- `belet_weyne_2019_bulletins.csv`: river level readings at Belet Weyne quoted in the FAO SWALIM flood bulletins of October 2019 (date, level in m, source).
- PSMN weekly flood displacements from Belet Weyne district (not included): download the UNHCR Protection and Return Monitoring Network (PRMN/PSMN) displacement data for Somalia from the UNHCR Operational Data Portal, select Belet Weyne as the district of origin and "Flood" as the reason, and aggregate by week (Sunday start) from 29 September to 30 November 2019. Save as `psmn_belet_weyne_weekly.csv` with columns `week_start,displaced`.

These are compared with the model by the analysis pipeline of the paper, not by `run.py`: PSMN gives weekly district totals, not counts per location.
