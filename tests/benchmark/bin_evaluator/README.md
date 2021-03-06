# Binary Evaluator

Calculates: Accuracy, Precision, Recall, F-measure and Confusion matrix


## DAG

DAG using 4 cores/fragments

![dag](./dag.png)


## Trace

Trace using 32 cores/fragments

![trace](./trace.png)


## Execution time by Input size

To the next test, we executed this application using five different numbers of rows (100kk, 200kk, 500k, 800kk, 1000kk). Furthermore, each configuration was executed five times. In this experiment, we excluded the time to data generation. 

![time_per_size](./time_per_size.png)
