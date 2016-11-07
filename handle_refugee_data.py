import numpy as np
import csv
from DataTable import *
from datetime import datetime

class RefugeeTable(DataTable):

  def retrofit_time_to_refugee_count(self, refugee_count, camp_names):
    """
    Takes the total refugee count in camps (from simulation), and determines at which time this number of refugees
    are found in the data. It then returns this time, which can be fractional.
    refugee_count: number of refugees in camps in the simulation, and the value we seek to match time against.
    camp_names: list of names (strings) of the refugee camp locations.
    TODO: make camp_names list auto-detectable in the simulation. 

    LIMITATION: This approach assumes a continual increase in refugee populations. Long-term decreasing trends 
    in the data will cause the function to return garbage.
    """

    last_data_count = 0
    initial_data_count = 0
    last_t = 0
    last_time_in_data = int(self.data_table[0][-1][self.days_column])
    #print("LAST TIME IN DATA = ", int(self.data_table[0][-1][self.days_column]))

    for name in camp_names:
      # aggregate refugee counts from all camps in the simulation
      initial_data_count += self.get_field(name , 0)
    last_data_count = initial_data_count

    for t in range(1, last_time_in_data-1):
      data_count = 0
      for name in camp_names:
        # aggregate refugee counts from all camps in the simulation
        data_count += self.get_field(name , t)

      #print(last_data_count, refugee_count, data_count)
      if int(refugee_count) >= last_data_count:
        if data_count > refugee_count:
          # the current entry in the table has a number that exceeds the refugee count we're looking for.
          # action: interpolate between current and last entry to get the accurate fractional time.
          t_frac = float(refugee_count - last_data_count) / float(data_count - last_data_count)
          #print("RETURN t_corr = ", last_t + t_frac * (t - last_t), ", t = ", t, ", last_t = ", last_t, ", refugees in data = ", data_count, "refugees in sim = ", refugee_count)
          return last_t + t_frac * (t - last_t)

      if data_count > last_data_count:
        # Only when the current refugee count in the data exceeds the highest one found previously, 
        # will we make a new interpolation checkpoint.
        last_data_count = data_count
        last_t = t

  def get_new_refugees(self, day, format="mali-portal", Debug=False, FullInterpolation=False):
     return self.get_daily_difference(day, Debug, FullInterpolation)
