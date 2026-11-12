# beledweyne_2019: the 2019 Shabelle River flood in Beledweyne, Somalia (DFlee with hydrodynamic forcing)

DFlee case of Jahani et al. (in preparation), "Coupling hydrodynamics and agent-based flood displacement: road accessibility, staged warnings and anticipatory routing in the 2019 Beledweyne flood". It needs a Flee version with the hydro rules, capacity rules and evacuation outcomes (`flee/hydro.py`, `flee/capacity.py`, `log_levels.evacuation_outcomes`).

## Running

```
fabsim localhost sflee:beledweyne_2019,simulation_period=304                        # V0 only, serial
fabsim localhost flee_ensemble:beledweyne_2019,simulation_period=304,replicas=20   # every variant in SWEEP, serial
fabsim <machine> pflee_ensemble:beledweyne_2019,simulation_period=304,replicas=20,cores=4   # the same with pflee
```

Then, for each run:

```
python -m flee.postprocessing.evacuation_outcomes <results_dir> --steps-per-day 4 --entrapment <results_dir>/entrapment.csv
```

One step is 6 hours: 304 steps cover 1 October to 15 December 2019. The `Date` column of `out.csv` counts steps as days, so read it as a step number.

## Variants (`SWEEP/`)

| Variant | Flood levels | Roads | Warnings | Anticipation |
|---|---|---|---|---|
| V0 | Prescribed: EMS extent of 3 Nov and reported dates (`input_csv/flood_level.csv`) | All open | | |
| V0h | HEC-RAS | All open | | |
| V1 | HEC-RAS | Closed or slowed per travel mode (`link_states.csv`) | | |
| V1A | HEC-RAS | As V1 | | On (aware share 0.8) |
| V2 | HEC-RAS | As V1 | Staged, response level s = 0.1 | |
| V3 | HEC-RAS | As V1 | As V2 | On |

FabSim copies each `SWEEP/<variant>/` folder over the case, so a variant's `input_csv/` files replace those of the base case. **The HEC-RAS files for V0h to V3 are not included yet** (see the README in each variant's `input_csv/`). They derive from the hydraulic model of Ibrahim et al. (2025) and will be added once its authors agree. Until then only V0 runs. To produce them, run `python -m flee.preprocessing.hecras2dflee` on the HEC-RAS plan.

`flood_movechances` is 0.22 times the DFlee defaults (k = 0.22, calibrated on V0 against the PSMN weekly departures). For V0h to V3, k is refitted once on V0h, and the value goes into every variant's settings.

## Files

| File | Content | Source |
|---|---|---|
| `input_csv/locations.csv` | 98 locations: 4 town zones, 50 other flood zones (villages, district exits), 41 IDP sites, and the shelter area east and north of the town as three points (Ceel_Jaale_S, _M, _N). `host_capacity`: displaced agents a non-camp location can host | OpenStreetMap, OCHA boundaries, REACH 2017 IDP site assessment |
| `input_csv/routes.csv` | 213 links, road lengths in km | OpenStreetMap |
| `input_csv/flood_level.csv` | V0 flood levels per 6 h step | Copernicus EMS EMSR404 and SWALIM reports |
| `input_csv/agents.csv` | 8,479 agents placed at the start, each standing for 20 residents of flooded buildings (EMS extent, 6 persons per building) | OpenStreetMap buildings, EMS extent |
| `input_csv/closures.csv`, `sim_period.csv` | No closures; start date and length | |
| `source_data/` | No camp counts exist for this flood: every location maps to an empty series, so the `data` and `error` columns of `out.csv` are placeholders | |
| `validation/` | River readings from the 2019 bulletins; instructions for the PSMN data | FAO SWALIM; UNHCR |
| `simsetting.yml`, `SWEEP/*/simsetting.yml` | Settings per variant | |

## How it differs from the other DFlee cases

- **6-hour steps** instead of days. Move speeds (`max_move_speed` 90, `max_walk_speed` 10) and move chances are per step.
- **Agents are placed at the start** from `agents.csv` (the exposed residents); `displaced_per_flood_day` is zero, and leaving is decided by `flood_movechances`.
- **Roads can close** (`hydro_rules`), per travel mode: foot 9%, large vehicles 67%, small vehicles 24% (IDMC 2021).
- **Capacity rules** are on in every variant (`capacity_rules`).
- **Outcomes** are per agent (`log_levels.evacuation_outcomes: 1`), not camp counts.
- `stay_close_to_home: True` with `home_distance_power: 2`, as in the DFlee release of Jahani et al. (2026), written out because they are not Flee's defaults.

## Licences and attribution

OpenStreetMap data are available under the Open Database Licence (ODbL), © OpenStreetMap contributors. Flood extent from the Copernicus Emergency Management Service (EMSR404), © European Union. Administrative boundaries from OCHA. River readings from FAO SWALIM flood bulletins.
