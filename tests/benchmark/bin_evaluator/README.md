# Binary Evaluator

Calculates: Accuracy, Precision, Recall, F-measure and Confusion matrix



# Use Case:

 - Number of workers/partitions: 8 workers / 32 fragments
 - Data length: 100kk rows of 2-dimension (col_label and col1)
 - Time to run: 132 seconds


## DAG

DAG using 4 cores/fragments

![dag](./dag.png)


## Trace

Trace using 32 cores/fragments

![trace](./trace.png)


