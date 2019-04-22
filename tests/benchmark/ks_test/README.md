# Kolmogorov-Smirnov test

Perform the Kolmogorov-Smirnov test for goodness of fit. This implementation of Kolmogorov–Smirnov test is a two-sided test for the null hypothesis that the sample is drawn from a continuous distribution.

The algorithm start first by creating a new data with only the column to be tested. Then, this column is sorted by sort algorithm (available in DDF library). After that, the minimal distance between the empirical data and the CDFs is calculated in each partition. Finally, these distances is merged to generate the p-value.


## Internal algorithms

This algorithm executes internaly the following operations/algorithms: 

1. Select: To project only the right column, reducing the data;
2. Range Partition: Used to be possible the sorting stage;
3. Sort Operation: To compare each row with the theorical CDFs;


In this sense, any improvement on these operations will benefit also this current algorithm. 


## DAG

DAG using 4 cores/fragments

![dag](./dag.png)


## Trace

![trace](./trace.png)



