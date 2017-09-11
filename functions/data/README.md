
# Data Operations:





## Change attributes:

Rename or change the data's type of some columns.




```sh

  AttributesChangerOperation():

  :param data:       A list with numFrag pandas's dataframe;
  :param settings:   A dictionary that contains:
      - attributes:  The column(s) to be changed (or renamed).
      - new_name:    The new name of the column. If used with multiple
                     attributes, a numerical suffix will be used to
                     distinguish them.
      - new_data_type: The new type of the selected columns:
          * 'keep';
          * 'integer';
          * 'string';
          * 'double';
          * 'Date';
          * 'Date/time';
  :param numFrag:    The number of fragments;
  :return:           Returns a list with numFrag pandas's dataframe.
```

## WorkloadBalancer:

Rebalance all the data in equal parts.

```sh
  WorkloadBalancerOperation():

  :param data:       A list with numFrag pandas's dataframe;
  :param numFrag:    The number of fragments;
  :return:           Returns a balanced list with numFrag pandas's dataframe.
```
