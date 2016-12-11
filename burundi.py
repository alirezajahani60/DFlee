import flee
import handle_refugee_data
import numpy as np
import analysis as a
import sys

"""
Generation 1 code. Incorporates only distance, travel always takes one day.
"""

#Burundi Simulation

if __name__ == "__main__":


  if len(sys.argv)>1:
    end_time = int(sys.argv[1])
    last_physical_day = int(sys.argv[1])
  else:
    end_time = 396
    last_physical_day = 396

  RetroFitting = False
  if len(sys.argv)>2:
    if "-r" in sys.argv[2]:
      RetroFitting = True
      end_time *= 10

  e = flee.Ecosystem()

  locations = []

  #Burundi
  locations.append(e.addLocation("Bujumbura", movechance=1.0, pop=497166))
  locations.append(e.addLocation("Bubanza", movechance=0.3))
  locations.append(e.addLocation("Bukinanyana", movechance=0.3, pop=75750))
  locations.append(e.addLocation("Cibitoke", movechance=0.3, pop=460435))
  locations.append(e.addLocation("Isale", movechance=0.3))

  locations.append(e.addLocation("Muramvya", movechance=0.3))
  locations.append(e.addLocation("Kayanza", movechance=0.3))
  locations.append(e.addLocation("Kabarore", movechance=0.3, pop=62303)) #This resides in Kayanza province in Burundi. Not to be confused with Kabarore, Rwanda.
  locations.append(e.addLocation("Mwaro", movechance=0.3, pop=273143))
  locations.append(e.addLocation("Rumonge", movechance=0.3))

  locations.append(e.addLocation("Burambi", movechance=0.3, pop=57167))
  locations.append(e.addLocation("Bururi", movechance=0.3))
  locations.append(e.addLocation("Rutana", movechance=0.3))
  locations.append(e.addLocation("Makamba", movechance=0.3))
  locations.append(e.addLocation("Gitega", movechance=0.3))

  locations.append(e.addLocation("Karuzi", movechance=0.3))
  locations.append(e.addLocation("Ruyigi", movechance=0.3))
  locations.append(e.addLocation("Gisuru", movechance=0.3, pop=99461))
  locations.append(e.addLocation("Cankuzo", movechance=0.3))
  locations.append(e.addLocation("Muyinga", movechance=0.3))

  locations.append(e.addLocation("Kirundo", movechance=0.3))
  locations.append(e.addLocation("Ngozi", movechance=0.3))
  locations.append(e.addLocation("Gashoho", movechance=0.3))
  locations.append(e.addLocation("Gitega-Ruyigi", movechance=0.3))
  locations.append(e.addLocation("Makebuko", movechance=0.3))

  locations.append(e.addLocation("Commune of Mabanda", movechance=0.3))

  camp_movechance = 0.001

  #Rwanda, Tanzania, Uganda and DRCongo camps
  locations.append(e.addLocation("Mahama", movechance=camp_movechance, capacity=49451, foreign=True))
  locations.append(e.addLocation("Nduta", movechance=camp_movechance, capacity=55320, foreign=True))
  locations.append(e.addLocation("Kagunga", movechance=1/21.0, foreign=True))
  locations.append(e.addLocation("Nyarugusu", movechance=camp_movechance, capacity=100925, foreign=True))
  locations.append(e.addLocation("Nakivale", movechance=camp_movechance, capacity=18734, foreign=True))
  locations.append(e.addLocation("Lusenda", movechance=camp_movechance, capacity=17210, foreign=True))

  #Within Burundi
  e.linkUp("Bujumbura","Bubanza","48.0")
  e.linkUp("Bubanza","Bukinanyana","74.0")
  e.linkUp("Bujumbura","Cibitoke","63.0")
  e.linkUp("Cibitoke","Bukinanyana","49.0")
  #e.linkUp("Bujumbura","Isale","11.0") # ???
  #e.linkUp("Isale","Muramvya","47.0") # ???
  e.linkUp("Bujumbura","Muramvya","58.0") 
  e.linkUp("Muramvya","Gitega","44.0")
  e.linkUp("Gitega","Karuzi","54.0")
  e.linkUp("Gitega","Ruyigi","55.0") 
  e.linkUp("Ruyigi","Karuzi","43.0") 
  e.linkUp("Karuzi","Muyinga","42.0")
  #e.linkUp("Isale","Kayanza","84.0") # ???
  e.linkUp("Bujumbura","Kayanza","95.0") 
  e.linkUp("Kayanza","Ngozi","31.0") ##
  e.linkUp("Ngozi","Gashoho","41.0") ##
  e.linkUp("Kayanza","Kabarore","18.0")
  e.linkUp("Gashoho","Kirundo","42.0")
  e.linkUp("Gashoho","Muyinga","34.0")
  e.linkUp("Bujumbura","Mwaro","67.0")
  e.linkUp("Mwaro","Gitega","46.0")
  e.linkUp("Bujumbura","Rumonge","75.0")
  e.linkUp("Rumonge","Bururi","31.0")
  e.linkUp("Rumonge","Burambi","22.0")
  e.linkUp("Rumonge","Commune of Mabanda","73.0")
  e.linkUp("Commune of Mabanda","Makamba","18.0") # ??
  e.linkUp("Bururi","Rutana","65.0")
  e.linkUp("Makamba","Rutana","50.0") # ??
  e.linkUp("Rutana","Makebuko","46.0") # ??
  e.linkUp("Makebuko","Gitega","24.0") # ??
  e.linkUp("Makebuko","Ruyigi","40.0")
  e.linkUp("Ruyigi","Cankuzo","51.0")
  e.linkUp("Ruyigi","Gisuru","31.0")
  e.linkUp("Cankuzo","Muyinga","63.0")

  #Camps, starting at index locations[26] (at time of writing).
  e.linkUp("Muyinga","Mahama","135.0")
  e.linkUp("Ruyigi","Nduta","90.0")
  e.linkUp("Gisuru","Nduta","60.0")
  e.linkUp("Commune of Mabanda","Kagunga","36.0")
  e.linkUp("Kagunga","Nyarugusu","91.0", forced_redirection=True) #From Kagunga to Kigoma by ship (Kagunga=Kigoma)
  e.linkUp("Kirundo","Nakivale","318.0")
  e.linkUp("Kayanza","Nakivale","413.0")
  e.linkUp("Bujumbura","Lusenda","53.0")


  d = handle_refugee_data.RefugeeTable(csvformat="generic", data_directory="burundi2015", start_date="2015-05-01")

  # Correcting for overestimations due to inaccurate level 1 registrations in five of the camps.
  # These errors led to a perceived large drop in refugee population in all of these camps.
  # We correct by linearly scaling the values down to make the last level 1 registration match the first level 2 registration value.
  # To our knowledge, all level 2 registration procedures were put in place by the end of 2016.

  d.correctLevel1Registrations("Mahama","2015-10-04")
  d.correctLevel1Registrations("Nduta","2016-04-06")
  d.correctLevel1Registrations("Nyarugusu","2015-11-10")
  d.correctLevel1Registrations("Nakivale","2015-08-18")
  d.correctLevel1Registrations("Lusenda","2015-09-30")
  
  locations[26].capacity = d.getMaxFromData("Mahama", last_physical_day)
  locations[27].capacity = d.getMaxFromData("Nduta", last_physical_day)
  locations[29].capacity = d.getMaxFromData("Nyarugusu", last_physical_day)
  locations[30].capacity = d.getMaxFromData("Nakivale", last_physical_day)
  locations[31].capacity = d.getMaxFromData("Lusenda", last_physical_day)

  list_of_cities = "Time"

  for l in locations:
    list_of_cities = "%s,%s" % (list_of_cities, l.name)

  #print(list_of_cities)
  #print("Time, campname")
  print("Day,Mahama sim,Mahama data,Mahama error,Nduta sim,Nduta data,Nduta error,Nyarugusu sim,Nyarugusu data,Nyarugusu error,Nakivale sim,Nakivale data,Nakivale error,Lusenda sim,Lusenda data,Lusenda error,Total error,refugees in camps (UNHCR),total refugees (simulation),raw UNHCR refugee count,retrofitted time,refugees in camps (simulation),refugee_debt,Total error (retrofitted)")


  e.add_conflict_zone("Bujumbura")

  #Set up a mechanism to incorporate temporary decreases in refugees
  refugee_debt = 0
  refugees_raw = 0 #raw (interpolated) data from TOTAL UNHCR refugee count only

  t_retrofitted = 0

  for t in range(0,end_time):

    if RetroFitting==False:
      t_data = t
    else:
      t_data = int(t_retrofitted)
      if t_data > end_time / 10:
        break

    #Append conflict_zone and weight to list.
    if t_data == 70: #Intense fighting between military & multineer military forces
      e.add_conflict_zone("Kabarore")

    elif t_data == 71: #Intense fighting between military & mulineer military forces
      e.add_conflict_zone("Bukinanyana")

    elif t_data == 75: #Battles unidentified armed groups coordinately attacked military barracks
      e.add_conflict_zone("Cibitoke")

    elif t_data == 178: #Clashes and battles police forces
      e.add_conflict_zone("Mwaro")

    elif t_data == 206: #Battles unidentified armed groups coordinate attacks
      e.add_conflict_zone("Gisuru")

    elif t_data == 221: #Military forces
      e.add_conflict_zone("Burambi")

    #new_refs = d.get_new_refugees(t)
    new_refs = d.get_new_refugees(t, FullInterpolation=True) - refugee_debt
    refugees_raw += d.get_new_refugees(t, FullInterpolation=True)
    if new_refs < 0:
      refugee_debt = -new_refs
      new_refs = 0
    elif refugee_debt > 0:
      refugee_debt = 0

    # Here we use the random choice to make a weighted choice between the source locations.
    for i in range(0, new_refs):
      e.addAgent(e.pick_conflict_location())

    #Propagate the model by one time step.
    e.evolve()

    #e.printInfo()

    #Validation/data comparison
    mahama_data = d.get_field("Mahama", t) #- d.get_field("Mahama", 0)
    nduta_data = d.get_field("Nduta", t) #-d.get_field("Nduta", 0)
    nyarugusu_data = d.get_field("Nyarugusu", t) #- d.get_field("Nyarugusu", 0)
    nakivale_data = d.get_field("Nakivale", t) #- d.get_field("Nakivale", 0)
    lusenda_data = d.get_field("Lusenda", t) #- d.get_field("Lusenda", 0)

    errors = []
    abs_errors = []
    loc_data = [mahama_data, nduta_data, nyarugusu_data, nakivale_data, lusenda_data]
    camp_locations = [26, 27, 29, 30, 31]

    camps = []
    for i in camp_locations:
      camps += [locations[i]]
    camp_names = ["Mahama", "Nduta", "Nyarugusu", "Nakivale", "Lusenda"]

    camp_pops_retrofitted = []
    errors_retrofitted = []
    abs_errors_retrofitted = []

    # calculate retrofitted time.
    refugees_in_camps_sim = 0
    for c in camps:
      refugees_in_camps_sim += c.numAgents
    t_retrofitted = d.retrofit_time_to_refugee_count(refugees_in_camps_sim, camp_names)

    # calculate errors
    for i in range(0,len(camp_locations)):
      camp_number = camp_locations[i]
      errors += [a.rel_error(locations[camp_number].numAgents, loc_data[i])]
      abs_errors += [a.abs_error(locations[camp_number].numAgents, loc_data[i])]

      # errors when using retrofitted time stepping.
      camp_pops_retrofitted += [d.get_field(camp_names[i], t_retrofitted, FullInterpolation=True)]
      errors_retrofitted += [a.rel_error(camps[i].numAgents, camp_pops_retrofitted[-1])]
      abs_errors_retrofitted += [a.abs_error(camps[i].numAgents, camp_pops_retrofitted[-1])]

    output = "%s" % t

    for i in range(0,len(errors)):
      camp_number = camp_locations[i]
      output += ",%s,%s,%s" % (locations[camp_number].numAgents, loc_data[i], errors[i])


    if refugees_raw>0:
      #output_string += ",%s,%s,%s,%s" % (float(np.sum(abs_errors))/float(refugees_raw), int(sum(loc_data)), e.numAgents(), refugees_raw)
      output += ",%s,%s,%s,%s,%s,%s,%s,%s" % (float(np.sum(abs_errors))/float(refugees_raw), int(sum(loc_data)), e.numAgents(), refugees_raw, t_retrofitted, refugees_in_camps_sim, refugee_debt, float(np.sum(abs_errors_retrofitted))/float(refugees_raw))
    else:
      output += ",0,0,0,0,0,0,0"
      #output_string += ",0"


    print(output)

