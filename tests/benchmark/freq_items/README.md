# Frequent Items

Finding frequent items for columns, possibly with false positives. Using the frequent element count algorithm described in "https://doi.org/10.1145/762471.762473, proposed by Karp, Schenker, and Papadimitriou".


# Use Case:

 - Number of workers/partitions: 8 workers / 32 fragments
 - Parameters: support=0.2


## Execution time by Input size

To the next test, we executed this application using five different numbers of rows (200kk, 400kk, 1kkk, 1.6kkk, 2kkk). Furthermore, each configuration was executed five times. In this experiment, we excluded the time to data generation. 

![time_per_size](./time_per_size.png)


## DAG

![dag](./dag.png)


## Trace

![trace](./trace.png)




