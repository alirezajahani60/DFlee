Simulation instance construction
================================


Overview
--------

This documentaion details how to construct a conflict scenario for forced displacement simulation. Each conflict situation requires:

- Three input CSV files:
   - input_csv/locations.csv
   - input_csv/routes.csv
   - input_csv/closures.csv
   
- Validation data files:
    - source_data/refugees.csv
    - source_data/data_layout.csv
    - source_data/<country_name-camp_name1>.csv
    - source_data/<country_name-camp_name2>.csv


Data extraction
---------------

1. The UNHCR situations provides an overview of active situations worldwide that are facing forced displacement distress. To construct a new conflict situation:
  - Select an active (conflict) situation of interest from an interactive map and click to access data and documentation      
    relevant to a chosen conflict situation from https://data2.unhcr.org/en/situations.
  - Select a simulation period for conflict situation from ``Refugees and asylum-seekers from `chosen situation name` -       
    Total`` timeline, which also presents forced displacement counts for a chosen period.
  - Obtain total counts of forcibly displaced people by clicking JSON button of ``Refugees and asylum-seekers from `chosen       
    situation name` - Total`` section. 
  - Identify camps for each neighbouring country through ``Breakdown by Country`` section of the conflict situation.
  - Collect and save data for each camp (e.g. <country_name-camp_name>.csv).
  
2. The ACLED database provides conflict location data for forced displacement simulations. To obtain data on chosen conflict situation, complete the ACLED data export tool fields (<ttps://www.acleddata.com/data>) as follows:
  - Provide dates of interest for conflict situation (i.e. From and To).
  - Select ``Event Type: Battle``.
  - Select ``Sub Event Type``: 
      - Armed clash, 
      - Attack, 
      - Government regains territory and 
      - Non-state actor overtakes territory.
  - Specify ``Region`` and ``Country`` of conflict situation choice.
  - Accept ``Terms of Use and Attribution Policy``.
  - <name>.csv file exports to Downloads automatically.
  - Revise the downloaded <name>.csv file:
     - Target the ``fatalities`` column and remove all rows in <name>.csv file with fatalities less than 1.
     - Choose the first conflict location occurrences of each location but exclude syndicated (repeated) locations.

     .. list-table:: An example of conflict locations (A, B and C). Conflict zone A occurs two times
                        with fatality numbers 3 and 38, while conflict zone C repeats twice with fatalities 14 and
                        7. Choose essence of locations (one of each location) at the first occurrence (e.g. A = 3, B = 23 and       
                        C = 14) for simulation purposes.
        :widths: 10 10 10
        :header-rows: 1
           
        * - ...
          - Location
          - Fatalities
        * - ...
          - A
          - 3
        * - ...
          - B
          - 23
        * - ...
          - A
          - 38
        * - ...
          - C
          - 14
        * - ...
          - C
          - 7
        * - ...
          - ...
          - ...
             


Construct input CSV files
-------------------------

1. Construct an input **locations.csv** file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ACLED conflict data provides conflict locations to construct **locations.csv** input file for simulation purposes. After identifying conflict locations and producing **locations.csv**, the last column is filled with population data for conflict locations. Population distributions can be obtained from <https://www.citypopulation.de> or other population databases.

**locations.csv** has the following format:

.. list-table:: Input camp names (i.e. destination locations) and their capacity into **locations.csv** file. Camp     
                capacity is the highest number of forced migrants for each camp and obtained from individual camp CSV    
                files that we set in **locations.csv.
   :widths: 5 10 10 5 5 15 15 30
   :header-rows: 1
           
   * - name
     - region
     - country 
     - lat
     - long 
     - location_type 
     - conflict_date 
     - population/capacity
   * - A
     - AA 
     - ABC 
     - xxx 
     - xxx 
     - conflict  
     - xxx      
     - xxx 
   * - B
     - BB 
     - ABC 
     - xxx 
     - xxx 
     - conflict  
     - xxx      
     - xxx 
   * - C
     - CC 
     - ABC 
     - xxx 
     - xxx 
     - conflict  
     - xxx      
     - xxx 
   * - ...
     - ...
     - ... 
     - ...
     - ... 
     - ...  
     - ...     
     - ...
             
             
2. Construct an input **routes.csv** file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Identified conflict zones and camps provide origin and destination locations. We connect these locations to represent how forcibly displaced people flee. We use http://www.bing.com/maps (or other mapping services) to connect conflict zones and camps, and add additional locations (if required) as a location type **town** to locations.csv as illustrated below:

.. list-table:: 
   :widths: 5 10 10 5 5 20 20 30
   :header-rows: 1
   :align: center
           
   * - name
     - region
     - country 
     - lat
     - long 
     - location_type 
     - conflict_date 
     - population/capacity
   * - A
     - AA 
     - ABC 
     - xxx 
     - xxx 
     - conflict  
     - xxx      
     - xxx 
   * - B
     - BB 
     - ABC 
     - xxx 
     - xxx 
     - conflict  
     - xxx      
     - xxx 
   * - C
     - CC 
     - ABC 
     - xxx 
     - xxx 
     - conflict  
     - xxx      
     - xxx 
   * - Z
     - ZZ 
     - ZZZ 
     - xxx 
     - xxx 
     - camp  
     -       
     - xxx 
   * - N
     - NN 
     - ABC 
     - xxx 
     - xxx 
     - town  
     -       
     - 
   * - ...
     - ...
     - ... 
     - ...
     - ... 
     - ...  
     - ...     
     - ...
          

Record distances between locations in **routes.csv** file for simulation using the following format:

.. list-table:: 
   :widths: 10 10 15 20
   :header-rows: 1
   :align: center
           
   * - name1
     - name2
     - distance [km]
     - forced_redirection
   * - A
     - B
     - x1
     -
   * - B
     - C
     - x2
     -
   * - A
     - C
     - x3
     - 
   * - B
     - N
     - x4
     - 
   * - C
     - N
     - x3
     -
   * - N
     - Z
     - x5
     - 
   * - ...
     - ...
     - ...
     -

   .. note: **forced_redirection** refers to redirection from source location (can be town or camp) to destination location     
            (mainly camp) and source location indicated as forwarding_hub. The value of 0 indicates no redirection, 1  
            indicates redirection (from name2) to name1and 2 corresponds to redirection (from name1) to name2.


3. Define location and border closures in **closures.csv** file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: We identify location or border closure events and document them in **closures.csv** file
   :widths: 20 10 10 30 30
   :headers-rows: 1
   :align: center
       
   * - closure_type
     - name1
     - name2 
     - closure_start = 0 
     - closure_end = -1
   * - location
     - A
     - B
     - xxx
     - xxx
   * - country
     - ABC
     - ZZZ
     - xxx
     - xxx 
   * - ...
     - ...
     - ...
     - ...
     - ...
      
      
**closure_type** has 2 possible values: 
- **location** corresponding to camp or town closure and 
- **country** referring to border closure. 

**closure_start** and **closure_end** are given as integers, counting the number of days after the simulation start. The value of 0 indicates the start, while -1 indicates the end of the simulation.


4. Construct a network map for a conflict situation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Construct an agent-based network map from **locations.csv** and **routes.csv** using <https://carto.com>.

.. image:: images/network.png
   :width: 300
   :align: center



Validation data
---------------

There are three CSV file formats required for validation of simulation outputs:

.. list-table:: CSV file containing total forced migrant counts **forced_migrants.csv** comprises total counts of forcibly   
                   displaced people from ``Refugees and asylum-seekers from `chosen situation name` - Total`` JSON file and     
                   has the format as demonstrated:
   :widths: 20 10
   :headers-rows: 1
   :align: center
       
   * - ...
     - ...
   * - YYYY-MM-DD
     - xxx
   * - YYYY-MM-DD
     - xxx
   * - ...
     - ...
      
        
.. list-table:: We obtain data for each camp using the format and label them as **country_name-camp_name.csv**.
   :widths: 20 10
   :headers-rows: 1
   :align: center
       
   * - ...
     - ...
   * - YYYY-MM-DD
     - xxx
   * - YYYY-MM-DD
     - xxx
   * - ...
     - ...
        
        
.. list-table:: **data_layout.csv** contains camp names for each camp/destination locations.
   :widths: 20 10
   :headers-rows: 1
   :align: center
       
   * - Total
     - refugees.csv
   * - camp_name1
     - <country_name-camp_name1>.csv
   * - camp_name2
     - <country_name-camp_name2>.csv
   * - ...
     - ...
