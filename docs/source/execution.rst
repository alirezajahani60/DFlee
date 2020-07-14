.. _execution:

Simulation instance execution
=============================

Execute test instance
---------------------

To run simulation instance using Flee with test, simply type:

.. code:: console

        python3 run_csv_vanilla.py test_data/test_input_csv test_data/test_input_csv/refugee_data 5 2010-01-01 2>/dev/null
  
.. note:: The **2>/dev/null** ensures that any diagnostics are not displayed on the screen. Instead, pure CSV output for the toy model should appear on the screen if this works correctly.
  

Execute a conflict scenario
---------------------------

1. Create an output directory **out<country_name>**.

2. Run the following command to execute <country_name>.py and obtain the simulation output, which will be written to out<country_name> as out.csv:
   
   .. code:: console

           python3 <country_name>.py <simulation_period> > out<country_name>/out.csv

3. Plot the simulation output using:

   .. code:: console

           python3 plot-flee-output.py out<country_name>

4. To analyse and interpret simulation output, open out<country_name>, which will contain simulation output and UNHCR data comparison graphs for each camp, as well as average relative difference graph for the simulated conflict situation.

   

Execute simplified simulation of Central African Republic (CAR) situation
-------------------------------------------------------------------------

To run the (simplified) CAR simulation:

1. Create an output directory **outcar**.

2. Execute car-csv.py in conflicts directory and obtain the simulation output:

   .. code:: console

           python3 conflicts/car-csv.py 50 > outcar/out.csv

3. Plot the simulation output using:

   .. code:: console

           python3 plot-flee-output.py outcar
    
4. Analyse and interpret simulation output graphs in the **outcar** directory.


Parallel Performance Testing
----------------------------

Parallel tests can be performed using test_par.py. The interface is as follows:

.. code:: console

        mpirun -np [number of cores] python3 tests/test_par.py [options]

Options can be as follows::

    "-p", "--parallelmode" - Parallelization mode ([advanced], classic, cl-hilat OR adv-lowlat).
    "-N", "--initialagents" - Number of agents at the start of the simulation [100000].
    "-d", "--newagentsperstep", Number of agents added per time step [1000].
    "-t", "--simulationperiod", Duration of the simulation in days [10].

Here are a few settings good for benchmarking::

    mpirun -np <cores> python3 test_par.py -N 500000 -p advanced -d 10000 -t 10
    mpirun -np <cores> python3 test_par.py -N 500000 -p classic -d 10000 -t 10
    mpirun -np <cores> python3 test_par.py -N 500000 -p cl-hilat -d 10000 -t 10
    mpirun -np <cores> python3 test_par.py -N 500000 -p adv-lowlat -d 10000 -t 10
