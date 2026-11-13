# Coupling HEC-RAS to DFlee (`hecras2dflee`)

`flee/preprocessing/hecras2dflee` turns HEC-RAS 2D flood results into DFlee inputs. It reads a HEC-RAS plan HDF file, samples water depth and velocity along roads and over settlement footprints, and writes the files that the [hydrodynamic movement rules](Hydrodynamic_Movement_Rules.md) read. Other tools in the package build warning schedules from a gauge record, report network fragmentation, compare the river stage with gauge readings, and record the provenance of a HEC-RAS run.

The coupling is offline and one-way: the hydraulic outputs are precomputed, and agent movement does not affect the flood.

## Installation

The tool needs packages that Flee itself does not, so they are optional:

```
pip install flee[hydro]        # h5py, scipy, shapely 2, geopandas, rasterio, pyyaml
```

## Running

```
python -m flee.preprocessing.hecras2dflee config.yaml
```

A complete configuration for the HEC-RAS "BaldEagleCrkMulti2D" dam-break example is in `test_data/test_data_dflee/dflee_test_hecras_bald_eagle/hecras2dflee/config.yaml`, with the DFlee case built from its output in the same folder.

| Output | Content | Used by |
|---|---|---|
| `flood_level.csv` | Flood level (0 to 3) per location and time step, in DFlee's format | DFlee, as `input_csv/flood_level.csv` |
| `link_states.csv` | Per step, link and travel mode: accessible (0/1), speed factor, high-percentile depth and speed, share outside the hydraulic domain, bridge state | `hydro_rules.link_states_file` |
| `closures_<mode>.csv` | Link closures for one travel mode, in `closures.csv` format | DFlee without hydro rules (one mode per run) |
| `summary.json` | Output times, coverage of the network by the hydraulic domain, locations outside it | Checking |

## Inputs

- **Plan HDF** (`.pXX.hdf`). If it has no mesh geometry, also the geometry HDF (`.gXX.hdf`).
- **Depth.** From a `Depth` output if present, otherwise water surface minus `Cells Minimum Elevation`, otherwise water surface minus a terrain raster (`dem`) at cell centres.
- **Velocity.** From `Cell Velocity - Velocity X/Y` if present, otherwise reconstructed per cell from face-normal velocities by weighted least squares.
- **Units.** US customary results are converted to SI.
- **Missing values.** NaN or infinite depths and velocities are set to zero with a warning that gives their number, and counted in `summary.json` (`nonfinite_values`). With `hydraulics.nonfinite: error` the run stops instead.
- **Links.** Line features with `name1`, `name2` matching `routes.csv`. An optional `bridge` column marks crossings, and `deck_clearance` gives a bridge's clearance in metres.
- **Locations.** Polygon or point features with `name` matching `locations.csv`; points are buffered by `point_buffer` metres.
- Vector files are reprojected to the model's coordinate system.

## Method

**Links.** Depth and speed are sampled every `sample_spacing` metres along each road. At each hydraulic output, a link is passable for a mode if the share of impassable samples is at most `max_impassable_share` (default 0: one impassable sample closes the link). A link is closed for a DFlee step if it is closed at any output within that step, so closures shorter than a step are kept if the output interval is shorter than the step. Its speed factor is the lowest over the step (`links.step_rule: worst`, the default; with `step_rule: start` the state at the first output of each step is used instead, as a check). With the default rule, `summary.json` also gives `within_step_closures`: the output interval and the share of closed link-steps, overall and per mode, that are open at the start of the step, i.e. what the conservative rule adds (`within_step_check: false` skips it), from the `percentile` depth (default 95th). By default a sample takes the depth of its cell; with `links.road_depth: point` and a `dem`, it takes the water surface minus the terrain at the sample point.

**Locations.** A statistic of depth over the footprint (`inundated_fraction`, `median_depth` or `p90_depth`) is computed at each output; its maximum over the step is turned into a flood level with `thresholds` [theta1, theta2, theta3].

**Passability criteria** (`criteria.py`), set per travel mode:

- `depth`: closed above `max_depth`.
- `hazard_rating`: HR = d (v + 0.5) + DF, closed above `max_hr` (UK FD2320/FD2321 classes); add `max_depth` for a wading limit.
- `xia2014`: closed where velocity reaches the incipient velocity for sliding or toppling of a person (Xia et al. 2014), with body height `h_p` and mass `m_p`; calibration `real` (real subjects) or `model` (flume model, more conservative).
- `dv_limits`: closed above `max_depth`, `max_velocity` or `max_dv` (depth times velocity), e.g. the vehicle limits of the Australian Disaster Resilience hazard classes.

**Speed reduction:** `linear`; `dias2021` for walking (Dias et al. 2021); `pregnolato2017` for vehicles (Pregnolato et al. 2017, closed above `closure_mm`, default 300 mm). Where a speed function reaches zero, the link is closed for that mode, whatever the criterion says. A `min_factor` floor (default 0.1) keeps passable links from getting a near-zero speed.

All limits are inclusive: a depth equal to `max_depth`, `closure_mm` or `zero_speed_depth` is still passable.

## Bridges and areas outside the hydraulic domain

These are choices that can change results, so they are explicit options rather than built in.

```yaml
links:
  exclude_initially_wet: 1.0           # samples wetter than this at the first output are the river channel
  bridge_rule: {type: rise, rise: 2.0}  # or {type: channel_open} (default) or {type: closed}
  outside_domain: {type: extent, file: observed_extent.geojson, depth: 0.5,
                   window: ["2019-10-21T00:00", "2019-11-15T00:00"]}   # default {type: open}
locations:
  outside_domain: {type: nearest, max_km: 5, level_offset: 1}           # or dry (default) or extent
```

- **Bridge rules.** Channel samples are always left out of the depth test. The crossing then stays open (`channel_open`), closes once the water at the crossing has risen by the deck clearance (`rise`), or is closed throughout (`closed`).
- **Links outside the domain** are open by default; with `extent`, samples inside an observed flood extent take the given depth during the window.
- **Locations outside the domain** are dry by default; alternatively they take the levels of the nearest location inside the domain (`nearest`) or a fixed level where an observed extent covers them (`extent`).
- **Extent check.** With a `validation` section, the modelled extent is compared with an observed one (hit rate, false alarm ratio, critical success index, bias). The scores count cells, not areas, so on a mesh with local refinement small cells weigh as much as large ones.

## Other tools

| Command | What it does |
|---|---|
| `python -m flee.preprocessing.hecras2dflee.gauge_warnings gauge.csv --thresholds 6.5 7.3 8.3 --start 2019-10-01 --steps-per-day 4 -o warnings.csv` | Builds `warnings.csv` from gauge readings and flood-risk thresholds. `--lead` shifts stages earlier; `--override k=date` sets a stage the record does not show. |
| `python -m flee.preprocessing.hecras2dflee.fragmentation link_states.csv routes.csv flood_level.csv locations.csv -o frag` | Per step and mode: connected components, flooded locations and population cut off from every safe location; per origin and mode, the last step at which a safe location can still be reached. Needs no agent runs. |
| `python -m flee.preprocessing.hecras2dflee.stage plan.hdf --refpoint NAME` | Modelled water surface at a gauge (reference point or 1D cross section), compared with gauge readings. Readings outside the simulation period are listed and left out of the error. |
| `python -m flee.preprocessing.hecras2dflee.provenance plan.hdf -o provenance` | Records the HEC-RAS version, files and checksums, intervals, mesh and structures of a run, and lists what to confirm with the modellers. |

## Limitations

- Points are matched to the nearest cell centre inside the flow-area perimeter, which is close to, but not always, the containing cell near break lines.
- 1D river reaches are read only for the gauge stage; links crossing the channel rely on the bridge rule.
- Forecasts used for anticipation come from the same flood levels (a perfect forecast), with an optional timing error in the hydro rules.
