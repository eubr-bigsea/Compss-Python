# Frequent Items

Finding frequent items for columns, possibly with false positives. Using the frequent element count algorithm described in "https://doi.org/10.1145/762471.762473, proposed by Karp, Schenker, and Papadimitriou".


# Use Case:

 - Number of workers/partitions: 8 workers / 32 fragments
 - Data length: 10kk rows
 - Parameters: support=0.2
 - Time to run: 29 seconds


## DAG

![dag](./dag.png)


## Trace

![trace](./trace.png)


