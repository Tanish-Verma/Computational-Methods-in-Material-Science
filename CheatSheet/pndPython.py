import pandas as pd

# Methods of pandas.DataFrame along with descriptions

dictionary={
    'name':['Dev','Rajat','Tanish'],
    'roll':[6,20,23],
    'dumb':[True,False,False],
    'ProblemCreation':[101,0,50]
}
df=pd.DataFrame(dictionary)
# Accessing data
df.head()        # Returns the first n rows (default 5).
df.tail()        # Returns the last n rows (default 5).
df.at[2]          # Access a single value for a row/column label pair.
df.iat[3]         # Access a single value for a row/column position pair.
df.loc[2]         # Access a group of rows and columns by labels or a boolean array.
df.iloc[1]        # Access a group of rows and columns by integer position(s).
df.get()         # Get item from object for a given key.

# Data cleaning and processing
df.drop()        # Remove rows or columns by specifying label names and axis.
df.dropna()      # Remove missing values.
df.fillna()      # Fill NA/NaN values using a specified method or value.
df.replace()     # Replace values with others using a mapping or condition.
df.apply()       # Apply a function along an axis of the DataFrame.
df.applymap()    # Apply a function to a DataFrame elementwise.
df.map()         # Map values of Series according to input correspondence (used for series).
df.mask()        # Replace values where the condition is True.
df.where()       # Replace values where the condition is False.

# Data aggregation and transformation
df.agg()         # Aggregate using one or more operations across one or more columns.
df.aggregate()   # Alias of agg().
df.groupby()     # Group data using a mapper or by a series of columns.
df.transform()   # Apply a function elementwise, transforming the data.
df.cumsum()      # Return cumulative sum over a DataFrame or Series axis.
df.cumprod()     # Return cumulative product over a DataFrame or Series axis.
df.cummax()      # Return cumulative maximum over a DataFrame or Series axis.
df.cummin()      # Return cumulative minimum over a DataFrame or Series axis.

# Data description and statistics
df.describe()    # Generate descriptive statistics that summarize the central tendency and dispersion.
df.corr()        # Compute pairwise correlation of columns, excluding NA/null values.
df.cov()         # Compute pairwise covariance of columns, excluding NA/null values.
df.mean()        # Return the mean of the values for the requested axis.
df.median()      # Return the median of the values for the requested axis.
df.mode()        # Return the mode(s) of the DataFrame.
df.min()         # Return the minimum of the values for the requested axis.
df.max()         # Return the maximum of the values for the requested axis.
df.sum()         # Return the sum of the values for the requested axis.
df.std()         # Return sample standard deviation over the requested axis.
df.var()         # Return unbiased variance over requested axis.
df.mad()         # Return the mean absolute deviation of the values.
df.prod()        # Return the product of the values over the requested axis.
df.skew()        # Return unbiased skew over the requested axis.
df.kurt()        # Return unbiased kurtosis over the requested axis.

# Reshaping, sorting, and pivoting
df.pivot()       # Reshape data based on column values.
df.pivot_table() # Create a pivot table as a DataFrame.
df.melt()        # Unpivot a DataFrame from wide format to long format.
df.stack()       # Stack the DataFrame columns into a single column.
df.unstack()     # Unstack the DataFrame index to columns.
df.transpose()   # Transpose the DataFrame (rows become columns and vice versa).
df.T             # Transpose the DataFrame (short form of transpose()).
df.sort_values() # Sort DataFrame by the values of a specific column.
df.sort_index()  # Sort DataFrame by index labels.

# Handling duplicates
df.duplicated()  # Return boolean Series denoting duplicate rows.
df.drop_duplicates() # Remove duplicate rows from the DataFrame.

# Index and columns manipulation
df.set_index()   # Set the DataFrame index using an existing column.
df.reset_index() # Reset the index, converting it into a column.
df.rename()      # Alter labels of rows and columns.
df.rename_axis() # Set or rename the axis labels.

# Input and output operations
df.to_csv()      # Write DataFrame to a CSV file.
df.to_excel()    # Write DataFrame to an Excel file.
df.to_json()     # Write DataFrame to a JSON file.
df.to_sql()      # Write records stored in DataFrame to a SQL database.
df.to_pickle()   # Serialize the DataFrame object to a file using pickle.
df.to_dict()     # Convert the DataFrame to a dictionary.
df.to_html()     # Render a DataFrame as an HTML table.
df.to_records()  # Convert DataFrame to a record array.
df.to_string()   # Render DataFrame as a string (for pretty-printing).
df.to_latex()    # Render DataFrame to a LaTeX tabular environment table.

# Merging, joining, and concatenating
df.merge()       # Merge DataFrame or named Series objects with a database-style join.
df.join()        # Join columns of another DataFrame.
df.concat()      # Concatenate two or more DataFrames along a particular axis.
df.append()      # Append rows of another DataFrame to the current one.
df.combine()     # Combine two DataFrames using a function.
df.combine_first() # Update null elements with corresponding elements from another DataFrame.

# Boolean indexing
df.any()         # Return whether any element is True over requested axis.
df.all()         # Return whether all elements are True over requested axis.

# Miscellaneous
df.memory_usage() # Return memory usage of each column.
df.info()        # Print concise summary of the DataFrame.
df.isna()        # Detect missing values (NaN).
df.notna()       # Detect existing (non-missing) values.
df.empty         # Indicator whether DataFrame is empty.
df.ndim          # Return the number of dimensions (axes) of the DataFrame.
df.size          # Return the number of elements in the DataFrame.
df.shape         # Return a tuple representing the dimensionality of the DataFrame.
df.values        # Return a Numpy representation of the DataFrame.
df.columns       # The column labels of the DataFrame.
df.index         # The row (axis) labels of the DataFrame.
df.dtypes        # Return the data types of each column.
df.memory_usage()# Memory usage of each column.

# Plotting (when using matplotlib)
df.plot()        # Make plots of DataFrame columns.
