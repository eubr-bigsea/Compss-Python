# Standard Scaler

Normalize a list of features using Standard scaler. This process has two stages: Fit and transform.  The first, 'fit', is responsible to generate all informations about the data, while in 'transform' stage is used this information to transform the input columns into new ones. In this example, 50% of dataset is used as training set.


# Use Case:

 - Number of workers: 8




## Performance

We executed this application using five different numbers of rows (100kk, 200kk, 500kk, 800kk, 1000kk). Furthermore, each configuration was executed five times. 

### Fit time using 50% of data as training set



![time_per_size](./time_per_size_fit.png)


### Transform time


![time_per_size](./time_per_size_transform.png)


## DAG

![dag](./dag.png)


## Trace

![trace](./trace.png)

