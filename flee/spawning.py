from flee.datamanager import handle_refugee_data, read_period
from flee.datamanager import DataTable
from flee.SimulationSettings import SimulationSettings

__refugees_raw = 0
__refugee_debt = 0

def add_initial_refugees(e, d, loc):
  """ Add the initial refugees to a location, using the location name"""
  if SimulationSettings.spawn_rules["InsertDayZeroRefugeesInCamps"]:
    num_refugees = int(d.get_field(loc.name, 0, FullInterpolation=True))
    for i in range(0, num_refugees):
      e.addAgent(location=loc) # Parallelization is incorporated *inside* the addAgent function.

def spawn_daily_displaced(e, t, d):
    global __refugees_raw, __refugee_debt
    """
    t = time
    e = Ecosystem object
    d = DataTable object
    refugees_raw = raw refugee count
    """

    if SimulationSettings.spawn_rules["conflict_driven_spawning"]:

      for i in range(0, len(e.conflict_zones)):
 
        ## BASE RATES  
        if SimulationSettings.spawn_rules["conflict_spawn_mode"] == "constant":
          num_spawned = SimulationSettings.spawn_rules["displaced_per_conflict_day"]

        elif SimulationSettings.spawn_rules["conflict_spawn_mode"] == "pop_ratio":
          num_spawned = int(SimulationSettings.spawn_rules["displaced_per_conflict_day"] * e.conflict_zones[i].pop)

        elif SimulationSettings.spawn_rules["conflict_spawn_mode"].lower() == "Poisson":
          num_spawned = np.random.poisson(SimulationSettings.spawn_rules["displaced_per_conflict_day"])


        ## ADD MULTIPLIER: SPAWN_DECAY
        if SimulationSettings.spawn_rules["conflict_spawn_decay"]:
          num_spawned = int(num_spawned * SimulationSettings.get_conflict_decay(time_since_conflict))
        

        ## Doing the actual spawning here.
        for j in range(0, num_spawned):
          e.addAgent(e.conflict_zones[i])


    else:

      # Determine number of new refugees to insert into the system.
      new_refs = d.get_daily_difference(t, FullInterpolation=True, SumFromCamps=False) - __refugee_debt
      __refugees_raw += d.get_daily_difference(t, FullInterpolation=True, SumFromCamps=False)

      #Refugees are pre-placed in Mali, so set new_refs to 0 on Day 0.
      if SimulationSettings.spawn_rules["InsertDayZeroRefugeesInCamps"]:
        if t == 0:
          new_refs = 0
          #refugees_raw = 0

      if new_refs < 0:
        __refugee_debt = -new_refs
        new_refs = 0
      elif __refugee_debt > 0:
        __refugee_debt = 0

      #Insert refugee agents
      for i in range(0, new_refs):
        e.addAgent(e.pick_conflict_location())

    return new_refs, __refugees_raw, __refugee_debt
