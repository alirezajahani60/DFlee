# covid_flee.py
# Covid-19 model, based on the general Flee paradigm.
import numpy as np
import sys
import random
from flee import SimulationSettings
from flee import flee
import array

lids = {"park":0,"hospital":1,"supermarket":2,"office":3,"school":4,"leisure":5,"shopping":6} # location ids and labels

class Needs():
  def __init__(self):
    self.add_needs()

  def i(self, name):
    for k,e in enumerate(self.labels):
      if e == name:
        return k

  def add_needs(self):
    self.needs = np.zeros((len(lids),120))

    self.needs[lids["park"]][:] = 120
    
    self.needs[lids["hospital"]][:] = 10
    
    self.needs[lids["supermarket"]][:] = 60
    
    self.needs[lids["office"]][19:] = 1200
    self.needs[lids["office"]][:20] = 0

    self.needs[lids["school"]][19:] = 0
    self.needs[lids["school"]][:20] = 1200
    
    self.needs[lids["leisure"]][:] = 120
    
    self.needs[lids["shopping"]][:] = 60

  def get_need(self, age, need):
    return self.needs[need,age]

  def get_needs(self, age):
    print(self.needs[:,age])
    return self.needs[:,age]

# Global storage for needs now, to keep it simple.
needs = Needs()

class Person():
  def __init__(self, location, age):
    self.location = location # current location
    self.location.IncrementNumAgents()
    self.home_location = location

    self.status = "susceptible" # states: susceptible, exposed, infectious, recovered, dead.
    self.symptomatic = False # may be symptomatic if infectious

    self.age = age # age in years


  def plan_visits(self):
    personal_needs = needs.get_needs(self.age)
    for k,element in enumerate(personal_needs):
      nearest_locs = self.home_location.nearest_locations
      if nearest_locs[k]:
        location_to_visit = nearest_locs[k]
        location_to_visit.register_visit(self, element)

  def get_needs(self):
    print(needs.get_needs(self.age))

  def evolve(self):
    pass

class Household():
  def __init__(self, house, size=-1):
    self.house = house
    if size>-1:
      self.size = size
    else:
      self.size = random.choice([1,2,3,4])

    self.agents = []
    for i in range(0,self.size):
      self.agents.append(Person(self.house, random.randint(0,100)))


def calc_dist(x1, y1, x2, y2):
    return (abs(x1-x2)**2 + abs(y1+y2)**2)**0.5

class House:
  def __init__(self, e, x, y, num_households=1):
    self.x = x
    self.y = y
    self.households = []
    self.numAgents = 0
    self.nearest_locations = self.find_nearest_locations(e)
    for i in range(0, num_households):
        self.households.append(Household(self))

  def IncrementNumAgents(self):
    self.numAgents += 1

  def DecrementNumAgents(self):
    self.numAgents -= 1

  def find_nearest_locations(self, e):
    """
    identify preferred locations for each particular purpose,
    and store in an array.
    """
    n = []
    for l in lids.keys():
      print(e.locations)
      if l not in e.locations.keys():
        n.append(None)
      else:
        min_dist = 99999.0
        nearest_loc_index = 0
        for k,element in enumerate(e.locations[l]): # using 'element' to avoid clash with Ecosystem e.
          d = calc_dist(self.x, self.y, element.x, element.y)
          if d < min_dist:
            min_dist = d
            nearest_loc_index = k
        n.append(e.locations[l][nearest_loc_index])
    return n



class Location:
  def __init__(self, name, loc_type="home", x=0.0, y=0.0, sqm=10000):
    self.name = name
    self.x = x
    self.y = y
    self.links = [] # paths connecting to other locations
    self.closed_links = [] #paths connecting to other locations that are closed.
    self.type = loc_type # home, supermarket, park, hospital, shopping, school, office, leisure?
    self.sqm = sqm # size in square meters.

    self.print()

  def DecrementNumAgents(self):
    self.numAgents -= 1

  def IncrementNumAgents(self):
    self.numAgents += 1

  def print(self):
    for l in self.links:
      print("Link from %s to %s, dist: %s" % (self.name, l.endpoint.name, l.distance), file=sys.stderr)

  def register_visit(self, person, need):
    pass


class Ecosystem:
  def __init__(self):
    self.locations = {}
    self.houses = []
    self.house_names = []

  def evolve(self):
    for h in self.houses:
      for hh in h.households:
        for a in hh.agents:
          a.plan_visits()
          a.evolve()

  def addHouse(self, name, x, y, num_households=1):
    self.houses.append(House(self, x, y, num_households))
    self.house_names.append(name)

  def addLocation(self, name, loc_type, x, y, sqm=10000):
    l = Location(name, loc_type, x, y, sqm)
    if loc_type in self.locations.keys():
      self.locations[loc_type].append(l)
    else:
      self.locations[loc_type] = [l]

  def print_needs(self):
    for k,e in enumerate(self.houses):
      for hh in e.households:
        for a in hh.agents:
          print(k, a.get_needs())


if __name__ == "__main__":
  print("No testing functionality here yet.")
