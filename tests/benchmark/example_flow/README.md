# Example of a simple flow of operations

This example is to show that DDF is capable of join narrow tasks in only one stage. In this example, we perform a projection on a selected column, after that, we filter values greather than zero. Finally, we create a new column, by inverting the signal of the previous column.


# Use Case:

 - 8 workers (32 cores)


## Execution time by Input size

To the next test, we executed this application using five different numbers of rows (100kk, 200kk, 500kk, 800kk, 1000kk). Furthermore, each configuration was executed five times. In this experiment, we excluded the time to data generation. 

![time_per_size](./time_per_size.png)


## DAG

![dag](./dag.png)


## Trace

![trace](./trace.png)



