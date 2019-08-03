#pflee.py
# A parallelized implementation of FLEE (with original rules)
#example to run: mpiexec -n 4 python3 pflee.py 100
import numpy as np
import sys
import random
from flee import SimulationSettings
from flee import flee
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE


class MPIManager:
  def __init__(self):
    if not MPI.Is_initialized():
      print("Manual MPI_Init performed.")
      MPI.Init()
    self.comm = MPI.COMM_WORLD
    self.rank = self.comm.Get_rank()
    self.size = self.comm.Get_size()

  def CalcCommWorldTotalSingle(self, i):

    total = np.array([-1])
    # If you want this number on rank 0, just use Reduce.
    self.comm.Allreduce(np.array([i]), total, op=MPI.SUM)
    return total[0]

  def CalcCommWorldTotal(self, np_array):
    assert np_array.size > 0

    total = np.zeros(np_array.size, dtype='i')

    #print(self.rank, type(total), type(np_array), total, np_array, np_array.size)
    # If you want this number on rank 0, just use Reduce.
    self.comm.Allreduce([np_array, MPI.INT], [total, MPI.INT], op=MPI.SUM)

    return total

class Person(flee.Person):
  def __init__(self, e, location):
    super().__init__(location)
    self.location.numAgentsOnRank += 1
    self.e = e

  def evolve(self):

    if self.travelling == False:
      movechance = self.location.movechance

      outcome = random.random()
      if outcome < movechance:
        # determine here which route to take?
        chosenRoute = self.selectRoute()

        # if there is a viable route to a different location.
        if chosenRoute >= 0:
          # update location to link endpoint
          self.location.numAgentsOnRank -= 1
          self.location = self.location.links[chosenRoute]
          self.location.numAgentsOnRank += 1
          self.travelling = True
          self.distance_travelled_on_link = 0

    self.timesteps_since_departure += 1


  def finish_travel(self, distance_moved_this_timestep=0):
    if self.travelling:

      # update last location of agent.
      #if not SimulationSettings.SimulationSettings.TurnBackAllowed:
      #  self.last_location = self.location

      self.distance_travelled_on_link += SimulationSettings.SimulationSettings.MaxMoveSpeed

      # If destination has been reached.
      if self.distance_travelled_on_link - distance_moved_this_timestep > self.location.distance:

        # update agent logs
        if SimulationSettings.SimulationSettings.AgentLogLevel > 0:
          self.places_travelled += 1
          self.distance_travelled += self.location.distance

        # if link is closed, bring agent to start point and return.
        if self.location.closed == True:
          #print("agent moved out of closed link.", file=sys.stderr)
          self.location.numAgentsOnRank -= 1
          self.location = self.location.startpoint
          self.location.numAgentsOnRank += 1
          self.travelling = False
          self.distance_travelled_on_link = 0

        else:

          # if the person has moved less than the minMoveSpeed, it should go through another evolve() step in the new location.
          evolveMore = False
          if self.location.distance + distance_moved_this_timestep < SimulationSettings.SimulationSettings.MinMoveSpeed:
            distance_moved_this_timestep += self.location.distance
            evolveMore = True

          # update location (which is on a link) to link endpoint
          self.location.numAgentsOnRank -= 1
          self.location = self.location.endpoint
          self.location.numAgentsOnRank += 1

          self.travelling = False
          self.distance_travelled_on_link = 0

          if SimulationSettings.SimulationSettings.CampLogLevel > 0:
            if self.location.Camp == True:
              self.location.incoming_journey_lengths += [self.timesteps_since_departure]


          # Perform another evolve step if needed. And if it results in travel, then the current
          # travelled distance needs to be taken into account.
          if evolveMore == True:
            self.evolve()
            self.finish_travel(distance_moved_this_timestep)

  def getLinkWeight(self, link, awareness_level):
    """
    Calculate the weight of an adjacent link. Weight = probability that it will be chosen.
    """

    # If turning back is NOT allowed, remove weight from the last location.
    #if not SimulationSettings.SimulationSettings.TurnBackAllowed:
    #  if link.endpoint == self.last_location:
    #    return 0.0 #float(0.1 / float(SimulationSettings.SimulationSettings.Softening + link.distance))

    if awareness_level < 0:
      return 1.0


    return float(self.e.scores[(link.endpoint.id * self.e.NumAwarenessLevels) + awareness_level] / float(SimulationSettings.SimulationSettings.Softening + link.distance))


class Location(flee.Location):

  def __init__(self, e, cur_id, name, x=0.0, y=0.0, movechance=0.001, capacity=-1, pop=0, foreign=False, country="unknown"):
    super().__init__(name, x, y, movechance, capacity, pop, foreign, country)

    self.id = cur_id

    self.scores = [] # Emptying this array, as it is not used in the parallel version.
    # If it is referred to in Flee in any way, the code should crash.
    self.e = e


  def updateRegionScore(self):
    """ Attractiveness of the local point, based on neighbourhood information from local and adjacent points,
        weighted by link length. """

    # No links connected or a Camp? Use LocationScore.
    if len(self.links) == 0 or self.camp:
      self.RegionScore = self.LocationScore
      return

    self.RegionScore = 0.0
    total_link_weight = 0.0

    for i in self.links:
      self.RegionScore += i.endpoint.NeighbourhoodScore / float(i.distance)
      total_link_weight += 1.0 / float(i.distance)

    self.RegionScore /= total_link_weight
    self.e.scores[self.id * self.e.NumAwarenessLevels + 3] = self.RegionScore


  def updateNeighbourhoodScore(self):

    """ Attractiveness of the local point, based on information from local and adjacent points, weighted by link length. """
    # No links connected or a Camp? Use LocationScore.
    if len(self.links) == 0 or self.camp:
      self.NeighbourhoodScore = self.LocationScore
      return

    self.NeighbourhoodScore = 0.0
    total_link_weight = 0.0

    for i in self.links:
      self.NeighbourhoodScore += i.endpoint.LocationScore / float(i.distance)
      total_link_weight += 1.0 / float(i.distance)

    self.NeighbourhoodScore /= total_link_weight
    self.e.scores[self.id * self.e.NumAwarenessLevels + 2] = self.NeighbourhoodScore


  def updateLocationScore(self, time):
    """ 
    Attractiveness of the local point, based on local point information only. 
    Time arg is redundant here, but I keep it here for now to ensure the serial version
    cannot be accessed in pflee. 
    """


    if self.foreign:
      self.LocationScore = SimulationSettings.SimulationSettings.CampWeight
    elif self.conflict:
      self.LocationScore = SimulationSettings.SimulationSettings.ConflictWeight
    else:
      self.LocationScore = 1.0

    self.e.scores[self.id * self.e.NumAwarenessLevels] = 1.0
    self.e.scores[self.id * self.e.NumAwarenessLevels + 1] = self.LocationScore


  def updateAllScores(self, time):
    """ Updates all scores of a particular location. Not available in Serial Flee, due to the reversed order there. """
    self.time = time
    self.updateRegionScore()
    self.updateNeighbourhoodScore()
    self.updateLocationScore(self.time)


class Link(flee.Link):
  pass
  #def __init__(self, endpoint, distance, forced_redirection=False):
  #  self.numAgentsOnRank = 0
  #  super().__init__(endpoint, distance, forced_redirection)


class Ecosystem(flee.Ecosystem):
  def __init__(self):


    self.locations = []
    self.locationNames = []
    self.agents = []
    self.total_agents = 0
    self.closures = [] #format [type, source, dest, start, end]
    self.time = 0
    self.mpi = MPIManager()

    if self.getRankN(0):
      print("Creating Flee Ecosystem.", file=sys.stderr)

    self.cur_loc_id = 0
    self.NumAwarenessLevels = 4
    self.scores = np.array([1.0,1.0,1.0,1.0]) # single array holding all the location-related scores.


    # Bring conflict zone management into FLEE.
    self.conflict_zones = []
    self.conflict_zone_names = []
    self.conflict_weights = np.array([])
    self.conflict_pop = 0

    self.parallel_mode = "loc-par" # classic for replicated locations or loc-par for distributed locations.
    self.latency_mode = "high_latency" # high_latency for fewer MPI calls with more prep, or low_latency for more MPI calls with less prep.

    if SimulationSettings.SimulationSettings.CampLogLevel > 0:
      self.num_arrivals = [] # one element per time step.
      self.travel_durations = [] # one element per time step.

  def getRankN(self, t):
    """
    Returns the <rank N> value, which is the rank meant to perform diagnostics at a given time step.
    Argument t contains the current number of time steps taken by the simulation.
    """
    N = t % self.mpi.size
    if self.mpi.rank == N:
      return True
    return False

  def updateNumAgents(self, CountClosed=False, mode="high_latency"):
    total = 0

    if mode=="low_latency":
      for loc in self.locations:
        loc.numAgents = self.mpi.CalcCommWorldTotalSingle(loc.numAgentsOnRank)
        total += loc.numAgents
        #print("location:", self.time, loc.name, loc.numAgents, file=sys.stderr)
        for link in loc.links:
          link.numAgents = self.mpi.CalcCommWorldTotalSingle(link.numAgentsOnRank)
          #print(self.time, "link:", loc.name, link.numAgents, file=sys.stderr)
          total += link.numAgents
        if CountClosed:
          for link in loc.closed_links:
            link.numAgents = self.mpi.CalcCommWorldTotalSingle(link.numAgentsOnRank)
            #print(self.time, "link [closed]:", loc.name, link.numAgents, file=sys.stderr)
            total += link.numAgents
      self.total_agents = total
      print("Total agents in simulation:", total, file=sys.stderr)
    elif mode == "high_latency":
      buf_len = 0
        
      buf_len += len(self.locations)
      for loc in self.locations:
          buf_len += len(loc.links)
          if CountClosed:
            buf_len += len(loc.closed_links)
       
      numAgent_buffer = np.empty(buf_len, dtype='i')
      new_buffer = np.empty(buf_len, dtype='i')

      index = 0
      for loc in self.locations:
        numAgent_buffer[index] = loc.numAgentsOnRank
        index += 1
        for link in loc.links:
          numAgent_buffer[index] = link.numAgentsOnRank
          index += 1
        if CountClosed:
          for link in loc.closed_links:
            numAgent_buffer[index] = link.numAgentsOnRank
            index += 1
            
      new_buffer = self.mpi.CalcCommWorldTotal(numAgent_buffer)

      index = 0
      for loc in self.locations:
        loc.numAgents = new_buffer[index]
        index += 1
        for link in loc.links:
          link.numAgents = new_buffer[index]
          index += 1
        if CountClosed:
          for link in loc.closed_links:
            link.numAgents = new_buffer[index]
            index += 1
            
      self.total_agents = np.sum(new_buffer)
      print("Total agents in simulation:", self.total_agents, file=sys.stderr)


  """
  Add & insert agent functions.
  TODO: make addAgent function smarter, to prevent large load imbalances over time
  due to removals by clearLocationFromAgents?
  """

  def addAgent(self, location):
    if SimulationSettings.SimulationSettings.TakeRefugeesFromPopulation:
      if location.pop > 1:
        location.pop -= 1
      else:
        print("ERROR: Number of agents in the simulation is larger than the combined population of the conflict zones. Please amend locations.csv.")
        assert location.pop > 1
    self.total_agents += 1
    if self.total_agents % self.mpi.size == self.mpi.rank:
      self.agents.append(Person(self, location))

  def insertAgent(self, location):
    """
    Note: insert Agent does NOT take from Population.
    """
    self.total_agents += 1
    if self.total_agents % self.mpi.size == self.mpi.rank:
      self.agents.append(Person(self, location))

  def insertAgents(self, location, number):
    for i in range(0,number):
      self.insertAgent(location)


  def clearLocationsFromAgents(self, location_names): #TODO:REWRITE!!
    """
    Remove all agents from a list of locations by name.
    Useful for couplings to other simulation codes.
    """

    new_agents = []
    for i in range(0, len(self.agents)):
      if self.agents[i].location.name not in location_names:
        new_agents += [self.agents[i]]
      else:
        #print("Agent removed: ", self.agents[i].location.name)
        self.agents[i].location.numAgentsOnRank -= 1 #agent is removed from ecosystem and number of agents in location drops by one.

    self.agents = new_agents
    print("clearLocationsFromAgents()", file=sys.stderr)
    self.updateNumAgents() # when numAgentsOnRank has changed, we need to updateNumAgents (1x MPI_Allreduce)


  def numAgents(self):
    return self.total_agents

  def numAgentsOnRank(self):
    return len(self.agents)

  def synchronize_locations(self, start_loc_local, end_loc_local):
      """
      Gathers the scores from all the updated locations, and propagates them across the processes.
      """

      base = int((len(self.scores)/4) / self.mpi.size)
      leftover = int((len(self.scores)/4) % self.mpi.size)

      #print(self.mpi.rank, base, leftover, len(self.scores))

      sizes = np.ones(self.mpi.size, dtype='i')*base
      sizes[:leftover] += 1
      sizes *= 4
      offsets = np.zeros(self.mpi.size, dtype='i')
      offsets[1:] = np.cumsum(sizes)[:-1]

      # Populate scores array
      scores_start = int(offsets[self.mpi.rank])
      local_scores_size = int(sizes[self.mpi.rank])
      local_scores = self.scores[scores_start:scores_start+local_scores_size].copy()

      #print(self.mpi.rank, local_scores, self.scores, sizes, offsets)
      self.mpi.comm.Allgatherv(local_scores, [self.scores, sizes, offsets, MPI.DOUBLE])


  def evolve(self):
    if self.time == 0:
        #print("rank, num_agents:", self.mpi.rank, len(self.agents))

        # Update all scores three times to ensure code starts with updated scores.
        for times in range(0, 3):
          for i in range(0, len(self.locations)):
            if i % self.mpi.size == self.mpi.rank:
              self.locations[i].updateAllScores(self.time)

    if self.parallel_mode == "classic":
      # update level 1, 2 and 3 location scores.
      # Scores remain perfectly updated in classic mode.
      for l in self.locations:
        l.time = self.time
        l.updateLocationScore(self.time)

      for l in self.locations:
        l.updateNeighbourhoodScore()

      for l in self.locations:
        l.updateRegionScore()

    elif self.parallel_mode == "loc-par":
      # update scores in reverse order for efficiency.
      # Neighbourhood and Region score will be outdated by 1 and 2 time steps resp.
      
      loc_per_rank = len(self.locations) / self.mpi.size
      lpr_remainder = len(self.locations) % self.mpi.size

      offset = int(self.mpi.rank) * int(loc_per_rank) + int(min(self.mpi.rank, lpr_remainder))
      num_locs_on_this_rank = int(loc_per_rank)
      if self.mpi.rank < lpr_remainder:
        num_locs_on_this_rank += 1

      for i in range(offset, offset + num_locs_on_this_rank):
        self.locations[i].updateAllScores(self.time)

      self.synchronize_locations(offset, offset + num_locs_on_this_rank)

    #update agent locations
    for a in self.agents:
      a.evolve()

    #print("NumAgents after evolve:", file=sys.stderr)
    self.updateNumAgents(CountClosed = True)

    for a in self.agents:
      a.finish_travel()

    #print("NumAgents after finish_travel:", file=sys.stderr)
    self.updateNumAgents(mode=self.latency_mode)

    #update link properties
    if SimulationSettings.SimulationSettings.CampLogLevel > 0:
      self._aggregate_arrivals()

    self.time += 1

  def addLocation(self, name, x="0.0", y="0.0", movechance=SimulationSettings.SimulationSettings.DefaultMoveChance, capacity=-1, pop=0, foreign=False, country="unknown"):
    """ Add a location to the ABM network graph """

    l = Location(self, self.cur_loc_id, name, x, y, movechance, capacity, pop, foreign, country)

    # Enlarge the scores array in Ecosystem to reflect the new location. Pflee only.
    if self.cur_loc_id > 0:
      self.scores = np.append(self.scores, np.array([1.0,1.0,1.0,1.0]))
    #print(len(self.scores))

    self.cur_loc_id += 1

    if SimulationSettings.SimulationSettings.InitLogLevel > 0:
      print("Location:", name, x, y, l.movechance, capacity, ", pop. ", pop, foreign, file=sys.stderr)
    self.locations.append(l)
    self.locationNames.append(l.name)
    return l


if __name__ == "__main__":
  print("No testing functionality here yet.")
