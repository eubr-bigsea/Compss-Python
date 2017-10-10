# ETL Operations:


## Add Columns:

Merge two dataframes, column-wise, similar to the command paste in Linux.

```
     AddColumnsOperation():

     :param df1:         A list with numFrag pandas's dataframe;
     :param df2:         A list with numFrag pandas's dataframe;
     :param balanced:    True only if len(df1[i]) == len(df2[i]) to each i;
     :param numFrag:     The number of fragments;
     :param suffixes:    Suffixes for attributes (a list with 2 values);
     :return:            A list with numFrag pandas's dataframe.
```

## Aggregation:

Computes aggregates and returns the result as a DataFrame.

```
     AggregationOperation():

     :param data:     A list with numFrag pandas's dataframe;
     :param params:   A dictionary that contains:
         - columns:   A list with the columns names to aggregates;
         - alias:     A dictionary with the aliases of all aggregated columns;
         - operation: A dictionary with the functionst to be applied in the aggregation:
             'mean':  Computes the average of each group;
             'count': Counts the total of records of each group;
             'first': Returns the first element of group;
             'last':  Returns the last element of group;
             'max':   Returns the max value of each group for one attribute;
             'min':   Returns the min value of each group for one attribute;
             'sum':   Returns the sum of values of each group for one attribute;
             'list':  Returns a list of objects with duplicates;
             'set':  Returns a set of objects with duplicate elements eliminated.
     :param numFrag:  The number of fragments;
     :return:         Returns a list with numFrag pandas's dataframe.

     example:
         settings['columns']   = ["col1"]
         settings['operation'] = {'col2':['sum'],     'col3':['first','last']}
         settings['aliases']   = {'col2':["Sum_col2"],'col3':['col_First','col_Last']}
```

## Clean Missing:

Clean missing fields from data set.

```
CleanMissingOperation():

:param data:          A list with numFrag pandas's dataframe;
:param params:        A dictionary that contains:
    - attributes:     A list of attributes to evaluate;
    - cleaning_mode:  What to do with missing values;
      * "VALUE":         replace by parameter "value";
      * "REMOVE_ROW":    remove entire row;
      * "MEDIAN":        replace by median value;
      * "MODE":          replace by mode value;
      * "MEAN":          replace by mean value;
      * "REMOVE_COLUMN": remove entire column;
    - value:         Used to replace missing values (if mode is "VALUE");
:param numFrag:      The number of fragments;
:return:             Returns a list with numFrag pandas's dataframe.
```

## Difference:

Function which returns a new set with containing rows in the first frame but not in the second one.

```
     DifferenceOperation():

     :param data1: A list with numFrag pandas's dataframe;
     :param data2: The second list with numFrag pandas's dataframe.
     :return:      A list with numFrag pandas's dataframe.
```

## Distinct (Remove Duplicated Rows):

Function which remove duplicates elements in a pandas dataframe based in some columns.

```
     DistinctOperation():

     :param data:      A list with numFrag pandas's dataframe;
     :param cols:      A list with the columns names to take in count (if no field is choosen, all fields are used).
     :param numFrag:   The number of fragments;
     :return:          Returns a list with numFrag pandas's dataframe.
```

## Drop:

Returns a new DataFrame that drops the specified column. Nothing is done if schema doesn't contain the given column name(s).

```
     DropOperation():

     :param data:    A list with numFrag pandas's dataframe;
     :param columns: A list with the columns names to be removed;
     :param numFrag: The number of fragments;
     :return:        A list with numFrag pandas's dataframe.
```

## Filter:

Returns a subset rows of dataFrame according to the specified query.

```
    FilterOperation():

    :param data:       A list with numFrag pandas's dataframe;
    :param settings:   A dictionary that contains:
        - 'query':     A valid query.
    :param numFrag:    The number of fragments;
    :return:           Returns a list with numFrag pandas's dataframe.


    Note: Visit the link bellow to more information about the query.
    https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html

    example:
        settings['query'] = "(VEIC == 'CARX')" to rows where VEIC is CARX
        settings['query'] = "(VEIC == VEIC) and (YEAR > 2000)" to rows where VEIC is not NaN and YEAR is greater than 2000
```


## Intersection:

Returns a new DataFrame containing rows only in both this frame and another frame.

```
     IntersectionOperation():

     :param data1:  A list with numFrag pandas's dataframe;
     :param data2:  Other list with numFrag pandas's dataframe;
     :return:       Returns a new pandas dataframe.
```

## Join:

Joins with another DataFrame, using the given join expression.

```
    JoinOperation():

    :param data1:      A list with numFrag pandas's dataframe;
    :param data2:      Other list with numFrag pandas's dataframe;
    :param settings:   A dictionary that contains:
        - 'option':    'inner' to inner join,'left' to left join and 'right' to right join.
        - 'key1':      A list of keys of the first dataframe;
        - 'key2':      A list of keys of the second dataframe;
        - 'case':      True to be case-sensitive, otherwise is False (default is True);
        - 'keep_keys': True to keep the keys of the second dataset (default is False).
    :param numFrag:    The number of fragments;
    :return:           Returns a list with numFrag pandas's dataframe.

    example:
    settings['key1'] = ["ID_CAR",'VEICBH']
    settings['key2'] = ["ID_CAR",'VEICRJ']
    settings['option'] = 'inner'
    settings['keep_keys'] = False
    settings['case'] = True

```

## Normalize:

Perform a Feature scaling (Range Normalization) or a Standard Score Normalization on the selected columns.

```
     NormalizeOperation():

     :param data:        A list with numFrag pandas's dataframe to perform the Normalization.
     :param settings:    A dictionary that contains:
       - mode:
         * 'range', to perform the Range Normalization, also called Feature scaling. (default option)
         * 'standard', to perform the Standard Score Normalization.
       - attributes: 	 Columns names to nrmalize;
       - alias:          A list of aliases of the new columns (if empty, overwrite the old fields);
     :param numFrag:     The number of fragments;
     :return:            A list with numFrag pandas's dataframe
```

## Replace Values:

Replace one or more values to new ones in a pandas's dataframe.

```
     ReplaceValuesOperation():

     :param data:      A list with numFrag pandas's dataframe;
     :param settings:  A dictionary that contains:
		- replaces:    A dictionary where each key is a column to perform an operation.
                       Each key is linked to a matrix of 2xN.
                       The first row is respect to the old values (or a regex) and the last is the new values.
        - regex:       True, to use a regex expression, otherwise is False (default is False);
     :param numFrag:   The number of fragments;
     :return:          Returns a list with numFrag pandas's dataframe

     example:
		* settings['replaces'] = {
		'Col1':[[<old_value1>,<old_value2>],[<new_value1>,<new_value2>]],
		'Col2':[[<old_value3>],[<new_value3>]]
		}
```

## Sample:

Returns a sampled subset of the input panda's dataFrame.

```
     SampleOperation():

     :param data:           A list with numFrag pandas's dataframe;
     :param params:         A dictionary that contains:
         - type:
             * 'percent':   Sample a random amount of records (default)
             * 'value':     Sample a N random records
             * 'head':      Sample the N firsts records of the dataframe
         - seed :           Optional, seed for the random operation.
         - int_value:       Value N to be sampled (in 'value' or 'head' type)
         - per_value:       Percentage to be sampled (in 'value' or 'head' type)
     :param numFrag:        The number of fragments;
     :return:               A list with numFrag pandas's dataframe.
```

## Select:

Function which do a Projection with the columns choosed.

```
     SelectOperation():

     :param data:    A list with numFrag pandas's dataframe;
     :param columns: A list with the columns names which will be selected;
     :param numFrag: The number of fragments;
     :return:        A list with numFrag pandas's dataframe with only the columns choosed.
```

## Sort:

Returns a new DataFrame sorted by the specified column(s).

```
     SortOperation():

     :param data:        A list with numFrag pandas's dataframe;
     :param settings:    A dictionary that contains:
         - algorithm:
             * 'odd-even', to sort using Odd-Even Sort (default);
             * 'bitonic',  to sort using Bitonic Sort (only if numFrag is power of 2);
         - columns:      The list of columns to be sorted;
         - ascending:    A list indicating whether the sort order is ascending (True) for the columns;
     :param numFrag:     The number of fragments;
     :return:            A list with numFrag pandas's dataframe.

     Condition:  the list of columns should have the same size of the list
                 of boolean to indicating if it is ascending sorting.
```

## Split:

Randomly splits a Data Frame into two data frames.

```
     SplitOperation():

     :param data:      A list with numFrag pandas's dataframe;
     :settings:        A dictionary that contains:
       - 'percentage': Percentage to split the data (default, 0.5);
       - 'seed':       Optional, seed in case of deterministic random operation.
     :return:          Returns two lists with numFrag pandas's dataframe with
                       distincts subsets of the input.

     Note:  if percentage = 0.25, the final dataframes
            will have respectively, 25% and 75%.  
```


## Transform:

Returns a new DataFrame applying the expression to the specified column.

```
     TransformOperation():

     :param data:      A list with numFrag pandas's dataframe;
     :param settings:  A dictionary that contains:
        - functions:   A list with an array with 3-dimensions.
          * 1ª position:  The lambda function to be applied as a string;
          * 2ª position:  The alias to new column to be applied the function;
          * 3ª position:  The string to import some needed module (if needed);
     :param numFrag: The number of fragments;
     :return:   Returns a list with numFrag pandas's dataframe with the news columns.

     example:
     	* settings['functions'] = [['newCol', "lambda row: np.sqrt(row['col1'])", 'import numpy as np']]
```




## Union:

Function which do a union between two pandas dataframes.

```
     UnionOperation():

     :param data1:   A list with numFrag pandas's dataframe;
     :param data2:   Other list with numFrag pandas's dataframe;
     :param numFrag: The number of fragments;
     :return:        Returns a list with numFrag pandas's dataframe.
```
