
# Data Operations:

## Data Reader:

### Read CSV:

Method used to load a unique pandas DataFrame from N csv files.

```sh
  ReadCSVOperation():
  :param filename:        The absolute path where the dataset is stored.
                          Each dataset is expected to be in a specific folder.
                          The folder will have the name of the dataset with the suffix "_folder".
                          The dataset is  already divided into numFrags files, sorted lexicographically.
  :param separator:       The string used to separate values;
  :param header:          True if the first line is a header, otherwise is False;
  :param infer:
    - "NO":               Do not infer the data type of each column (will be string);
    - "FROM_VALUES":      Try to infer the data type of each column;
    - "FROM_LIMONERO":    !! NOT IMPLEMENTED YET!!
  :param na_values:       A list with the all nan caracteres to be considerated.
  :return                 A DataFrame splitted in a list with length N.
```

#### Example:

```sh
$ ls /var/workspace/dataset_folder
    dataset_00     dataset_02
    dataset_01     dataset_03
```

### Read Json:




Method used to load a pandas DataFrame from N json files.

```sh
  ReadCSVOperation():
    :param filename:        The absolute path where the dataset is stored.
                            Each dataset is expected to be in a specific folder.
                            The folder will have the name of the dataset with the suffix "_folder".
                            The dataset is already divided into numFrags files, sorted lexicographically.
    :param settings:        A dictionary with the following parameters:
      - 'infer':
        * "NO":              Do not infer the data type of each column (will be string);
        * "FROM_VALUES":     Try to infer the data type of each column (default);
        * "FROM_LIMONERO":   !! NOT IMPLEMENTED YET!!
    :return                  A DataFrame splitted in a list with length N.
```

#### Example:

```sh    
    $ ls /var/workspace/dataset_folder
        dataset_00     dataset_02
        dataset_01     dataset_03
```

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
