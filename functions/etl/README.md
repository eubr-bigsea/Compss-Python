# ETL Operations:


## Add Columns:

Merge two dataframes, column-wise, similar to the command paste in Linux.

```
     AddColumnsOperation():
     
     :param df1:         A list with numFrag pandas's dataframe;
     :param df2:         A list with numFrag pandas's dataframe;
     :param balanced:    True only if len(df1[i]) == len(df2[i]) to each i;
     :param numFrag:     The number of fragments;
     :return:            A list with numFrag pandas's dataframe.
```

- Aggregation
- Clean Missing
- Difference
- Distinct (Remove Duplicated Rows)

## Drop:

Returns a new DataFrame that drops the specified column. Nothing is done if schema doesn't contain the given column name(s).

```
     DropOperation():
        
     :param data:    A list with numFrag pandas's dataframe;
     :param columns: A list with the columns names to be removed;
     :param numFrag: The number of fragments;
     :return:        A list with numFrag pandas's dataframe.
```

- Intersection
- Join

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
       - alias:          Aliases of the new columns;
     :param numFrag:     The number of fragments;
     :return:            A list with numFrag pandas's dataframe
```

- Replace Values
- Sample

## Select:

Function which do a Projection with the columns choosed.

```
     SelectOperation():
         
     :param data:    A list with numFrag pandas's dataframe;
     :param columns: A list with the columns names which will be selected;
     :param numFrag: The number of fragments;
     :return:        A list with numFrag pandas's dataframe with only the columns choosed.
```

- Sort
- Split
- Transform

## Union:

Function which do a union between two pandas dataframes.

```
     UnionOperation():
        
     :param data1: 		A list with numFrag pandas's dataframe;
     :param data2: 		Other list with numFrag pandas's dataframe;
     :param numFrag: 	The number of fragments;
     :return:      		Returns a list with numFrag pandas's dataframe.
```