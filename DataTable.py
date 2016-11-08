import sys
import numpy as np
import csv
from datetime import datetime

def subtract_dates(date1, date2):
  """
  Takes two dates %Y-%m-%d format. Returns date1 - date2, measured in days.
  """
  date_format = "%Y-%m-%d"
  a = datetime.strptime(date1, date_format)
  b = datetime.strptime(date2, date_format)
  delta = a - b
  #print(date1,"-",date2,"=",delta.days)
  return delta.days 

def _processEntry(row, table, data_type, date_column, count_column, start_date):
  """
  Code to process a population count from a CSV file.
  column <date_column> contains the corresponding date in %Y-%m-%d format.
  column <count_column> contains the population size on that date.
  """
  if(row[0][0] == "#"):
    return table

  # Make sure the date column becomes an integer, which contains the offset in days relative to the start date.
  row[date_column] = subtract_dates(row[date_column], start_date)

  if data_type == "int":
    table = np.vstack([table,[int(row[date_column]), int(row[count_column])]])
  else:
    table = np.vstack([table,[float(row[date_column]), float(row[count_column])]])

  return table


def ConvertCsvFileToNumPyTable(csv_name, data_type="int", date_column=0, count_column=1, start_date="2012-02-29"):
  """
  Converts a CSV file to a table with date offsets from 29 feb 2012.
  CSV format for each line is:
  yyyy-mm-dd,number

  Default settings:
  - subtract_dates is used on column 0.
  - Use # sign to comment out lines. (first line is NOT ignored by default)
  """
  table = np.zeros([0,2])

  with open(csv_name, newline='') as csvfile:
    values = csv.reader(csvfile)

    row = next(values)  
    
    if(len(row)>1): 
      if len(row[0])>0 and "DateTime" not in row[0]:
        table = _processEntry(row, table, data_type, date_column, count_column, start_date) 

    for row in values:
      table = _processEntry(row, table, data_type, date_column, count_column, start_date)

  return table


class DataTable:
  def __init__(self, data_directory="mali2012", data_layout="data_layout_refugee.csv", start_date="2012-02-29", csvformat="generic"):
    """
    read in CSV data files containing refugee data.
    """
    self.csvformat = csvformat
    self.total_refugee_column = 1
    self.days_column = 0
    self.header = []
    self.data_table = []
    self.start_date = start_date

    if self.csvformat=="generic":
      with open("%s/%s" % (data_directory, data_layout), newline='') as csvfile:
        values = csv.reader(csvfile)
        for row in values:
          if(len(row)==2):
            if(row[0][0] == "#"):
              continue
            self.header.append(row[0])

            self.data_table.append(ConvertCsvFileToNumPyTable("%s/%s" % (data_directory, row[1]), start_date=start_date))

    #print(self.header, self.data_table)

  def get_daily_difference(self, day, day_column=0, count_column=1, Debug=False, FullInterpolation=True, ZeroOnDayZero=False):
    """
    Extrapolate count of new refugees at a given time point, based on input data.
    count_column = column which contains the relevant difference.
    FullInterpolation: when disabled, the function ignores any decreases in refugee count.
    when enabled, the function can return negative numbers when the new total is higher than the older one.
    """
    
    self.total_refugee_column = count_column
    self.days_column = day_column
    ref_table = self.data_table[0]

    # Refugees only come in *after* day 0.
    if int(day) == 0:
      if ZeroOnDayZero:
        return 0
      else:
        ref_table = self.data_table[0]

        new_refugees = 0
        for i in self.header[1:]:
          new_refugees += self.get_field(i, 0, FullInterpolation)
          #print("Day 0 data:",i,self.get_field(i, 0, FullInterpolation))

        return int(new_refugees)

    else:

      new_refugees = 0
      for i in self.header[1:]:
        new_refugees += self.get_field(i, day, FullInterpolation) - self.get_field(i, day-1, FullInterpolation)

        #print self.get_field("Mbera", day), self.get_field("Mbera", day-1)
      return int(new_refugees)

    # If the day exceeds the validation data table, then we return 0
    return 0



  def get_interpolated_data(self, column, day):
    """
    Gets in a given column for a given day. Interpolates between days as needed.
    """

    ref_table = self.data_table[column]

    old_val = ref_table[0,self.total_refugee_column]
    #print(ref_table[0][self.days_column])
    old_day = ref_table[0,self.days_column]
    if day <= old_day:
      return old_val

    for i in range(1, len(ref_table)):
      #print(day, ref_table[i][self.days_column])
      if day < ref_table[i,self.days_column]:

        old_val = ref_table[i-1,self.total_refugee_column]
        old_day = ref_table[i-1,self.days_column]

        fraction = float(day - old_day) / float(ref_table[i,self.days_column] - old_day)

        if fraction > 1.0:
          print("Error with days_column: ", ref_table[i,self.days_column])
          return -1

        #print(day, old_day, ref_table[i][self.total_refugee_column], old_val)
        return int(old_val + fraction * float(ref_table[i,self.total_refugee_column] - old_val))

    #print("# warning: ref_table length exceeded for column: ",day, self.header[column], ", last ref_table values: ", ref_table[i-1][self.total_refugee_column], ref_table[i][self.days_column])
    return ref_table[-1,self.total_refugee_column]

  def get_raw_data(self, column, day):
    """
    Gets in a given column for a given day. Does not Interpolate.
    """
    ref_table = self.data_table[column]

    old_val = ref_table[0][self.total_refugee_column]
    old_day = 0

    for i in range (0,len(ref_table)):
      if day >= ref_table[i][self.days_column]:
        old_val = ref_table[i][self.total_refugee_column]
        old_day = ref_table[i][self.days_column]
      else:
        break
    return int(old_val)

  def _find_headerindex(self, name):
    """
    Finds matching index number for a particular name in the list of headers.
    """
    for i in range(0,len(self.header)):
      if self.header[i] == name:
        return i

    print(self.header)
    sys.exit("Error: can't find the header %s in the header list" % (name))


  def get_field(self, name, day, FullInterpolation=True):
    """
    Gets in a given named column for a given day. Interpolates between days if needed.
    """

    i = self._find_headerindex(name)
    if FullInterpolation:
      #print(name, i, day, self.get_interpolated_data(i, day))
      return self.get_interpolated_data(i, day)
    else:
      return self.get_raw_data(i, day)

  def is_interpolated(self, name, day):
    """
    Checks if data for a given day is inter/extrapolated or not.
    """
    for i in range(0,len(self.header)):
      if self.header[i] == name:
        ref_table = self.data_table[i]
        for j in range(0, len(ref_table)):
          if int(day) == int(ref_table[j][self.days_column]):
            return False
          if int(day) < int(ref_table[j][self.days_column]):
            return True

    return True

  #def d.correctLevel1Registrations(name, date):
  # correct for start date.
