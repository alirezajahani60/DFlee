# Hydrodynamic movement rules (DFlee)

`flee/hydro.py` lets flood depth and velocity from a hydrodynamic model shape movement in DFlee: which roads each agent can use, how fast it travels, flood warnings and anticipation of road closures. It works in serial Flee and in pflee. Nothing changes unless `hydro_rules: enabled: True` is set in `simsetting.yml`.

## Input files

**`link_states.csv`** gives, per time step, link and travel mode, whether the link can be used and a speed factor between 0 and 1:

```
step,name1,name2,mode,accessible,speed_factor
0,A,B,foot,1,1.0
0,A,B,vehicle,1,1.0
1,A,B,vehicle,0,0.0
```

Links are symmetric. Steps after the last one listed keep the last state; links not listed are open with factor 1. The file can be produced from HEC-RAS 2D results with the `hecras2dflee` tool (https://github.com/alirezajahani60/dflee-hecras-coupling).

**`warnings.csv`** (optional) lists warning stages, the step at which each is issued and the warned locations (`;`-separated, or `all`):

```
#stage,issue_step,locations
1,24,A;B
2,40,all
```

`flood_level.csv` is read as for DFlee; non-integer levels are interpolated.

## Configuration

```yaml
hydro_rules:
  enabled: True
  link_states_file: input_csv/link_states.csv
  modes:                       # travel mode drawn per agent
    foot:    {share: 0.7}      # max_speed defaults to max_walk_speed
    vehicle: {share: 0.3}      # max_speed defaults to max_move_speed
  aware_share_without_warnings: 0.0
  warnings:
    file: input_csv/warnings.csv
    evac_prob:   [0.4, 0.6, 0.8]   # minimum move chance at a warned location, per stage
    aware_share: [0.2, 0.5, 0.8]   # cumulative share of agents aware of the forecast, per stage
  anticipation:
    enabled: True
    horizon: 3                 # steps
    strength: 0.8              # 0 to 1
    weighting: links           # links | length | time
    forecast_shift: 0          # forecast timing error in steps (> 0: closures forecast too early)
  entrapment_level: 2          # flood level counted as unsafe
  entrapment_log: hydro_entrapment.csv
```

## What the rules do

- **Accessibility and speed.** Each agent is given a travel mode. Routes containing a link closed for that mode are dropped; a planned route whose next link has closed is replanned; an agent on a link when it closes turns back; agents with no open route stay where they are. Travel speed is the mode's maximum speed times the link's speed factor.
- **Warnings.** From the issue step of stage k, agents at warned locations move with probability at least `evac_prob[k]`, and the share of agents there aware of the forecast rises to `aware_share[k]`.
- **Anticipation.** For aware agents, a route's weight is multiplied by `1 - strength * phi`, where `phi` is the share of the route (by links, length or travel time) forecast to close before the agent, at its current speed, could pass it within the horizon. Destination flood weights use the highest forecast flood level over the horizon.
- **Entrapment log.** Per step and mode, the number of agents at locations at or above `entrapment_level` that cannot reach a safer location over open links within their awareness range (`entrapped`), and those that cannot reach one anywhere on the graph (`physically_isolated`).

## Outcomes

Set `log_levels: evacuation_outcomes: 1` to write per-agent outcomes (departure, exposure, time to safety, travel mode; see [Output file descriptions](Output_file_descriptions.md)), and summarise them with `python -m flee.postprocessing.evacuation_outcomes`. This works with the hydro rules off too, so a run with all roads open can be compared with one where they close.

## Notes

- Under pflee each rank holds the full network and its own agents, so the rules run unchanged; entrapment counts are summed over ranks and rank 0 writes the log. Every rank reads `link_states.csv`.
- `fixed_routes` is supported. Route weights computed elsewhere (e.g. `crawling.py`) are not filtered.
- Agent state (mode, awareness) is held in `hydro.py`, keyed by agent identity, not in `Person.attributes`.
