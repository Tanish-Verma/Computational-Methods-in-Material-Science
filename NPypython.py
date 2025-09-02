import numpy as np

'''To use PI'''
p=np.pi

'''Syntax to declare a numpy array'''
arr=np.array([1,2,3]) #1d array
#or
l=[1,2,3]
arr=np.array(l)
arr=np.array([[1,2,3],[4,5,6]])

'''To access element at ith row jth column'''
element=arr[1,2]

'''To get dimensions of array (shape of array)'''
s=arr.shape

'''To change shape of array'''
newarr=arr.reshape(3,2)

'''Iterating is same as general list. Apply that i am lazy to write again'''

'''Syntax to declare a array of range 0 to n. Example n=5
arange function excluses the number provided to it (6 in this case)'''
arr=np.arange(6)

'''To declare array of range m to n. Example m=2 n=5
arange function includes first number but excludes second number'''
arr=np.arange(2,6)

'''To declare array from m to n with difference k. Example 3 to 30 with difference 3
Again second number provided will be exluded in arange function
useful to make array of multiplication table'''
arr=np.arange(3,31,3)

'''To create a array of 1000 numbers between 1 and 10'''
arr=np.linspace(0,10,100)

'''To create an array with all elements as zero
Pass the dimensions to function
elements will be of type float
Dont't remove the inner bracket because the elements inside together define dimension'''
arr=np.zeros((3,4))
#to make elements int
arr=np.zeros((3,4),dtype=int)

'''To create an array with all elements as one
Pass the dimensions to function'''
arr=np.ones((3,4))
#to make all elements int
arr=np.ones((3,4),dtype=int)

'''To make an array with all elements as any value x
pass dimension and value to the function
here 10 is the value and (3,4 )are dimension
Again here type is float'''
arr=np.full((3,4),10)

'''Create identity matrix of NxN'''
arr=np.eye(5)

''''Create square array with diagonal elements as [1,2,3]'''
arr=np.diag([1,2,3])

'''To make a copy of array so that modifying one doesn't affect other'''
arr2=arr.copy()

'''To repeat the array
in below exampls if arr is [1,2,3]  arr2 will be [1,2,3,1,2,3]'''
arr2=np.tile(arr,2)

'''To repeat eacch element n times
here if arr is [1,2,3] then arr2 is [1,1,2,2,3,3]'''
arr2=np.repeat(arr,2)

'''To make a copy of array so that modifying one affects other'''
arr2=arr.view()
#or
arr2=arr[:]
#but dooing so in a list will make a copy of the list

'''Joining 2 arrays'''
#1d
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
arr=np.concatenate((arr1,arr2)) # will be [1,2,3,4,5,6]
#2d
arr1=np.array([[1,2],[3,4]])
arr2=np.array([[5,6],[7,8]])
arr=np.concatenate((arr1,arr2)) #will be [1,2],[3,4],[5,6],[7,8]
arr=np.concatenate((arr1,arr2),axis=1) # will be [1,2,5,6],[3,4,7,8]

'''To stack array over other specially 1d arrays'''
#to stack horizontally
arr1=np.array([1,2,3])
arr2=np.array([4,5,6])
arr=np.hstack((arr1,arr2)) #will give [1,2,3,4,5,6]
#to stack vertically
arr=np.vstack((arr1,arr2)) #will be [1,2,3],[4,5,6]

'''To contruct blocks using sub_arrays or scalars'''
arr=np.block([np.array([1,2,3]),np.array([4,5,6])]) #1d array [1,2,3,4,5,6]
result = np.block([[np.array([1, 2]), np.array([3, 4])],[np.array([5, 6]), np.array([7, 8])]])#2d array [[1 2 3 4] [5 6 7 8]] 
result = np.block([[1, np.array([2, 3])], [np.array([4, 5]), 6]])#2d array [[1 2 3] [4 5 6]]
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6]])
result = np.block([[A], [B]]) #[[1 2][3 4][5 6]]

'''To make nD array into 1D array'''
arr2=arr.flatten()

'''To split the array. Have to pass array and n. It will split array into n parts
If there are not enough elements to split it equally properly it will adjust on its own'''
arr=np.array([1,2,3,4,5,6])
newarr=np.array_split(arr,3) # will give a 2d array containing [1,2],[3,4],[5,6]
newarr=np.array_split(arr,4) # will give a 2d array containing [1,2],[3,4],[5],[6]

'''To search index an element in a array'''
index=np.where(arr==4) #if more than 1 index it return an array
index=np.where(arr%2==0) #return indexes of even numbers
# Use np.where() for multiple conditions
arr = np.array([1, 2, 3, 4, 5])
index = np.where((arr > 2) & (arr < 5))  # Find indices where elements are greater than 2 and less than 5
# index will contain the indices [2, 3]

'''To sort array'''
arr=np.sort(arr) # can sort character array alphabetically

'''To find sum, mean, median, standard deviation, cumulative sum,cumulative product'''
s=np.sum(arr)
mn=np.mean(arr)
mdn=np.median(arr)
st=np.std(arr)
cs=np.cumsum(arr)
cp=np.cumprod(arr)

'''to get max min'''
max_value=np.max(arr)
max_index=np.argmax(arr)
min_value=np.min(arr)
min_index=np.argmin(arr)

'''To do dot product of array'''
result=np.dot(arr,arr1)

'''to do matrix multiplication'''
arr2=np.matmul(arr,arr1)

'''Broadcasting examples'''
A = np.array([1, 2, 3])
B = np.array([[1], [2], [3]])
C = A + B  # A is a row matrix and B is column matrix. to add these both extend itself so that both become 3x3 matrix

'''Inverse of a matrix'''
arr2=np.linalg.inv(arr)

'''determinant of matrix'''
result=np.linalg.det(arr)

'''to solve system of linear equations'''
ans=np.linalg.solve(arr,arr1)

'''To evaluate a value of polynomial at a value of x'''
value_of_x=2
coefficients=np.array([1,2,3,4])
value=np.polyval(coefficients,value_of_x)
# ----------------------- MESHGRID  -----------------------
arr=np.array([1,2,3])
arr2=np.array([4,5,6])
x,y=np.meshgrid(arr,arr2) #Here x is [1,2,3] in 3 rows. Y is [4 4 4][5 5 5][6 6 6]
x,y=np.meshgrid(arr2,arr) #here x is [4 5 6] in 3 rows. y is [1 1 1][2 2 2][3 3 3]
# Create meshgrid for generating coordinate matrices in higher dimensions
x = np.array([1, 2])
y = np.array([3, 4])
z = np.array([5, 6])
xx, yy, zz = np.meshgrid(x, y, z)  # Generates 3D grid coordinates
# xx, yy, zz will be used for 3D plotting or calculations
# This can be extended to more than 3 dimensions if needed.
'''To make a co-ordinate system'''


'''To generate random integers from m to n
m is included and n is excluded'''
num=np.random.randint(10,100)

'''To generate an array of shape aXb with random integers between m to n'''
arr=np.random.randint(10,100,size=(3,4))

'''To generate a random float between 0 and 1 (1 is exluded)'''
num=np.random.rand()

'''To generate an array of random float between 0 to 1 (1 is excluded) of shape aXb'''
num=np.random.rand(3,4)

'''To generate a random number in an array'''
arr=[1,2,3,4,5]
num=np.random.choice(arr)

'''To generate an array with random numbers in an array'''
arr=[1,2,3,4,5]
arr2=np.random.choice(arr,size=(3,4))

# ----------------------- ARRAY INITIALIZATION -----------------------

# Create an empty array (no initialization, faster than zeros/ones)
arr = np.empty((3, 4))  # empty array

# Create an identity matrix
identity_matrix = np.identity(3)  # identity matrix

# Create an array of random numbers from the standard normal distribution (mean=0, stddev=1)
randn_array = np.random.randn(3, 4)  # random standard normal distribution

# ----------------------- BOOLEAN INDEXING -----------------------

# Boolean indexing to filter even numbers from an array
arr = np.array([1, 2, 3, 4, 5])
even = arr[arr % 2 == 0]  # boolean indexing for even numbers

# ----------------------- FANCY INDEXING -----------------------

# Indexing using arrays of indices (fancy indexing)
arr = np.array([10, 20, 30, 40])
indices = [0, 3]
result = arr[indices]  # fancy indexing

# ----------------------- RANDOM SAMPLING -----------------------

# Generate a random permutation of a range of numbers (0 to 9)
permutation = np.random.permutation(10)  # random permutation

# Generate random numbers from a normal (Gaussian) distribution
normal_dist = np.random.normal(loc=0, scale=1, size=(3, 4))  # normal distribution random numbers

# ----------------------- ARRAY MANIPULATION -----------------------

# Transpose an array (swap rows and columns)
arr = np.array([[1, 2, 3], [4, 5, 6]])
transposed_arr = arr.T  # transpose an array

# Flatten an n-dimensional array to a 1D array
flattened_arr = arr.flatten()  # flatten array to 1D

# Convert 1D arrays into columns using column_stack()
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
stacked_arr = np.column_stack((arr1, arr2))  # stack arrays as columns

# ----------------------- BROADCASTING -----------------------

# Broadcasting example: adding a scalar to an array
arr = np.array([1, 2, 3])
arr2 = arr + 1  # broadcasting to add scalar 1 to each element

# ----------------------- ADVANCED MATHEMATICS -----------------------

# Compute the matrix rank
matrix = np.array([[1, 2], [3, 4]])
rank = np.linalg.matrix_rank(matrix)  # matrix rank

# Compute the trace of a matrix (sum of diagonal elements)
trace_value = np.trace(matrix)  # trace of matrix

# ----------------------- CLIPPING VALUES -----------------------

# Clip (limit) values in an array to a specified range
arr = np.array([1, 2, 3, 4, 5])
clipped_arr = np.clip(arr, 2, 4)  # clip values to between 2 and 4

# ----------------------- LOGICAL OPERATIONS -----------------------

# Use np.all() to check if all elements satisfy a condition
print(np.all(arr > 0))  # check if all elements are greater than 0

# Use np.any() to check if any element satisfies a condition
print(np.any(arr > 4))  # check if any element is greater than 4

# ----------------------- PEAK-TO-PEAK RANGE -----------------------

# Compute the peak-to-peak range (max - min)
arr = np.array([1, 2, 3, 4, 5])
range_val = np.ptp(arr)  # peak-to-peak range

# ----------------------- SORTING AND ARGUMENT SORTING -----------------------

# Sort array elements and return their sorted indices using argsort
arr = np.array([3, 1, 2])
sorted_indices = np.argsort(arr)  # argsort for indices of sorted array

# ----------------------- UNIQUE ELEMENTS -----------------------

# Find unique elements in an array
arr = np.array([1, 2, 2, 3, 3, 3, 4])
unique_elements = np.unique(arr)  # unique elements in array
# Find unique elements in an array along with their counts
arr = np.array([1, 2, 2, 3, 3, 3, 4])
unique_elements, counts = np.unique(arr, return_counts=True)  # unique elements with their counts
# unique_elements will be [1, 2, 3, 4] and counts will be [1, 2, 3, 1]

# ----------------------- CREATING MESHGRID -----------------------

# Create meshgrid for generating 2D coordinate matrices
x = np.array([1, 2, 3])
y = np.array([4, 5])
xx, yy = np.meshgrid(x, y)  # meshgrid for coordinate arrays
# xx = [[1, 2, 3], [1, 2, 3]], yy = [[4, 4, 4], [5, 5, 5]]

# ----------------------- ADVANCED DATA TYPE HANDLING -----------------------

# Define a NumPy array with a specific data type
arr = np.array([1, 2, 3], dtype=np.int32)  # specify int32 data type

# Convert the data type of an array to float
float_arr = arr.astype(np.float64)  # convert array to float64

# Check the data type of an array
print(arr.dtype)  # Output: int32


# ----------------------- HANDLING SPECIAL VALUES (np.nan, np.inf) -----------------------

# Creating an array with np.nan (Not a Number) and np.inf (infinity)
special_values_arr = np.array([1, np.nan, 3, np.inf])

# Check for NaN values in the array
nan_mask = np.isnan(special_values_arr)  # boolean mask for NaN values

# Check for infinity values in the array
inf_mask = np.isinf(special_values_arr)  # boolean mask for infinity values


# ----------------------- MEMORY LAYOUT -----------------------

# Copy an array (deep copy)
arr = np.array([1, 2, 3])
arr_copy = arr.copy()  # deep copy of the array

# Create a view of the array (shallow copy)
arr_view = arr.view()  # view, changes in arr_view affect arr

# Get the memory strides of an array (the number of bytes to step in each dimension)
print(arr.strides)  # Output: strides of the array (memory layout)


# ----------------------- MATRIX OPERATIONS -----------------------

# Raise a matrix to a specific power
matrix = np.array([[1, 2], [3, 4]])
powered_matrix = np.linalg.matrix_power(matrix, 2)  # matrix raised to power 2

# Compute the trace of a matrix (sum of diagonal elements)
trace_value = np.trace(matrix)  # trace of the matrix


# ----------------------- PADDING AND ROLLING -----------------------

# Pad an array with constant values (e.g., 0s)
arr = np.array([[1, 2], [3, 4]])
padded_arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)  # add padding

# Roll elements of an array (circular shift)
rolled_arr = np.roll(arr, 1)  # roll elements by 1 position


# ----------------------- LINEAR ALGEBRA -----------------------

# Compute the eigenvalues and eigenvectors of a matrix
eigenvalues, eigenvectors = np.linalg.eig(matrix)  # eigenvalues and eigenvectors

# Compute the cross product of two 3D vectors
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
cross_prod = np.cross(a, b)  # cross product of vectors a and b


# ----------------------- ARRAY MANIPULATION -----------------------

# Add a new axis to an array using np.newaxis
arr = np.array([1, 2, 3])
expanded_arr = arr[:, np.newaxis]  # adds a new axis

# Repeat elements of an array using np.tile
arr = np.array([1, 2, 3])
tiled_arr = np.tile(arr, (2, 3))  # repeat the array elements in rows and columns

# Stack arrays along a new axis using np.stack
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
stacked_arr = np.stack((arr1, arr2), axis=0)  # stack arrays along a new axis


# ----------------------- USEFUL MATHEMATICAL OPERATIONS -----------------------

# Compute the cumulative product of array elements
arr = np.array([1, 2, 3, 4])
cumprod_arr = np.cumprod(arr)  # cumulative product of array elements

# Clip (limit) values in an array to a specified range
arr = np.array([1, 2, 3, 4, 5])
clipped_arr = np.clip(arr, 2, 4)  # clip values to between 2 and 4

# Advanced slicing using np.s_[] for better readability
arr = np.arange(10)
sliced_arr = arr[np.s_[2:8:2]]  # slice every 2nd element from index 2 to 7
print(sliced_arr)  # Output: [2, 4, 6]

import numpy as np

# ----------------------- LOGARITHM AND EXPONENTIAL -----------------------

# Natural logarithm (element-wise log)
arr = np.array([1, np.e, np.e**2])
log_arr = np.log(arr)  # natural log of each element
print(log_arr)  # Output: array([0., 1., 2.])

# Exponential (element-wise exp)
arr = np.array([0, 1, 2])
exp_arr = np.exp(arr)  # exponential of each element
print(exp_arr)  # Output: array([ 1.        ,  2.71828183,  7.3890561 ])

# ----------------------- DISCRETE DIFFERENCE -----------------------

# Compute the n-th discrete difference (difference between consecutive elements)
arr = np.array([1, 2, 4, 7, 0])
diff_arr = np.diff(arr)  # difference between consecutive elements
print(diff_arr)  # Output: array([ 1,  2,  3, -7])

# ----------------------- CROSS PRODUCT -----------------------

# Compute the cross product of two vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
cross_product = np.cross(a, b)  # cross product of vectors a and b
print(cross_product)  # Output: array([-3,  6, -3])

# ----------------------- OUTER PRODUCT -----------------------

# Compute the outer product of two arrays
outer_product = np.outer(a, b)  # outer product of vectors a and b
print(outer_product)
# Output:
# array([[ 4,  5,  6],
#        [ 8, 10, 12],
#        [12, 15, 18]])

# ----------------------- BINCOUNT (COUNT OCCURRENCES OF INTEGERS) -----------------------

# Count occurrences of integers in an array
arr = np.array([1, 1, 2, 3, 4, 4, 4, 5])
bin_counts = np.bincount(arr)  # counts the occurrences of each integer
print(bin_counts)  # Output: array([0, 2, 1, 1, 3, 1])  # counts from 0 to max value

# ----------------------- HISTOGRAM -----------------------

# Compute the histogram of a dataset (frequency count in bins)
data = np.array([1, 2, 1, 4, 5, 6, 2, 3, 5, 6, 7, 8, 9])
hist, bin_edges = np.histogram(data, bins=5)  # divide into 5 bins
print(hist)  # Output: array([3, 2, 2, 2, 4])  # counts in each bin
print(bin_edges)  # Output: array([1. , 2.6, 4.2, 5.8, 7.4, 9. ])
# ----------------------- 2D HISTOGRAM -----------------------
# Compute the 2D histogram of two datasets
x = np.random.randn(1000)
y = np.random.randn(1000)
hist, xedges, yedges = np.histogram2d(x, y, bins=30)  # 2D histogram
# hist contains counts in each bin, xedges and yedges contain bin edges

# ----------------------- CORRELATION MATRIX -----------------------

# Compute the correlation coefficient matrix of two datasets
x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 4, 3, 2, 1])
corr_matrix = np.corrcoef(x, y)  # correlation matrix of x and y
print(corr_matrix)
# Output:
# array([[ 1., -1.],
#        [-1.,  1.]])

# ----------------------- TRIANGULAR MATRICES -----------------------

# Extract the upper triangular part of a matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
upper_tri = np.triu(matrix)  # upper triangular part
print(upper_tri)
# Output:
# array([[1, 2, 3],
#        [0, 5, 6],
#        [0, 0, 9]])

# Extract the lower triangular part of a matrix
lower_tri = np.tril(matrix)  # lower triangular part
print(lower_tri)
# Output:
# array([[1, 0, 0],
#        [4, 5, 0],
#        [7, 8, 9]])

import numpy as np

# 1. **Rounding Functions** - np.round(), np.floor(), np.ceil()
# These functions round, floor, or ceil the values in an array element-wise.
arr = np.array([1.2, 2.5, 3.7])
rounded_arr = np.round(arr)  # Rounds to nearest integer
floored_arr = np.floor(arr)  # Floors values to the nearest lower integer
ceiled_arr = np.ceil(arr)    # Ceils values to the nearest upper integer

# 2. **Modulus Function** - np.mod()
# Computes element-wise modulus operation.
arr1 = np.array([10, 20, 30])
arr2 = np.array([3, 7, 5])
mod_arr = np.mod(arr1, arr2)  # Computes modulus of arr1 by arr2

# 3. **Element-wise Power** - np.power()
# Raises each element of an array to the specified power.
arr = np.array([1, 2, 3])
powered_arr = np.power(arr, 3)  # Raises each element to the power of 3

# 4. **Covariance Matrix** - np.cov()
# Computes the covariance matrix of two datasets.
x = np.array([1, 2, 3, 4])
y = np.array([4, 5, 6, 7])
covariance_matrix = np.cov(x, y)  # Computes covariance between x and y

# 5. **variance of array** - np.var() ::
#computes the variance of the data
# Example data
data = [1, 2, 3, 4, 5]
pop_variance = np.var(data)

# 6. **Kronecker Product** - np.kron()
# Computes the Kronecker product of two arrays.
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 5], [6, 7]])
kron_product = np.kron(A, B)  # Computes the Kronecker product of A and B

# 7. **Evenly Spaced Values** - np.linspace()
# Generates an array of evenly spaced values between two numbers.
linspace_arr = np.linspace(0, 10, 5)  # 5 values between 0 and 10

# 8. **Logarithmically Spaced Values** - np.logspace()
# Generates an array of numbers spaced evenly on a log scale.
logspace_arr = np.logspace(1, 3, 4)  # 4 values between 10^1 and 10^3

# 9. **Expanded Meshgrid for 3D Coordinates** - np.meshgrid()
# Creates coordinate matrices from coordinate vectors for 3D.
x = np.array([1, 2])
y = np.array([3, 4])
z = np.array([5, 6])
xx, yy, zz = np.meshgrid(x, y, z)  # Generates 3D grid coordinates

# 10. **Padding an Array** - np.pad()
# Pads an array with constant values or specific modes.
arr = np.array([[1, 2], [3, 4]])
padded_arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)  # Pads with zeros

# 11. **Insert and Delete Elements** - np.insert(), np.delete()
# Insert values into an array or delete values from an array.
arr = np.array([1, 2, 3, 4, 5])
inserted_arr = np.insert(arr, 2, [10, 20])  # Inserts 10 and 20 at index 2
deleted_arr = np.delete(arr, [1, 2])  # Deletes elements at index 1 and 2

# 12. **Numerical Integration** - np.trapz()
# Compute the approximate integral of an array using the trapezoidal rule.
y = np.array([1, 2, 3])
integral = np.trapz(y, x=[0, 1, 2])  # Approximate integral of y

# 13. **Eigenvalues and Eigenvectors** - np.linalg.eig()
# Computes the eigenvalues and eigenvectors of a matrix.
matrix = np.array([[1, 2], [2, 1]])
eigenvalues, eigenvectors = np.linalg.eig(matrix)  # Computes eigenvalues and eigenvectors

# 14. **QR Decomposition** - np.linalg.qr()
# Decomposes a matrix into an orthogonal matrix Q and an upper triangular matrix R.
matrix = np.array([[1, 2], [3, 4]])
Q, R = np.linalg.qr(matrix)  # Performs QR decomposition

# 15. **Cumulative Product** - np.cumprod()
# Compute the cumulative product along an array.
arr = np.array([1, 2, 3, 4])
cumprod_arr = np.cumprod(arr)  # Cumulative product

# 16. **Roll Array Elements** - np.roll()
# Rolls (shifts) elements of an array along an axis.
arr = np.array([1, 2, 3, 4, 5])
rolled_arr = np.roll(arr, 2)  # Rolls elements by 2 positions

# 17. **Vectorizing Functions** - np.vectorize()
# Apply a Python function element-wise over arrays.
def my_function(x):
    return x ** 2  # Example function to square the input

vectorized_function = np.vectorize(my_function)  # Vectorizes the function
arr = np.array([1, 2, 3, 4])
result = vectorized_function(arr)  # Applies the function element-wise
#hello world

### 1. Handling Floating-Point Precision Issues

# Compare two floating-point arrays with a tolerance
# np.isclose(arr1, arr2) returns a boolean array where True indicates that
# the values are close to each other within a specified tolerance.
arr1 = np.array([1.0000001, 2.0000002])
arr2 = np.array([1.0000002, 2.0000003])
print(np.isclose(arr1, arr2))  # Output: [ True  True ]

# Check if all elements in two arrays are close
# np.allclose(arr1, arr2) returns True if all corresponding elements
# are close within the specified tolerance.
print(np.allclose(arr1, arr2))  # Output: True

### 2. Controlling Precision with np.set_printoptions()

# Set the precision for printing floating-point arrays.
np.set_printoptions(precision=3)  # Set to 3 decimal places
arr = np.array([1.123456789, 2.987654321])
print(arr)  # Output: [1.123 2.988]

# Reset to default precision
np.set_printoptions(precision=8)

### 3. Handling Special Floating-Point Values

# Check for NaN values in an array.
arr = np.array([np.nan, np.inf, 1.0, -np.inf])
nan_mask = np.isnan(arr)  # Output: [ True False False False ]
print(nan_mask)

# Check if elements are finite (not NaN or Inf).
finite_mask = np.isfinite(arr)  # Output: [False False  True False]
print(finite_mask)

### 4. Avoiding Floating-Point Overflows and Underflows

# Use np.log1p for accurate logarithm computation for small values.
small_values = np.array([1e-20, 1e-10, 1])
log_result = np.log1p(small_values)  # More accurate than np.log(1 + small_values)
print(log_result)

# Use np.expm1 for accurate exponential computation for small values.
exp_result = np.expm1(small_values)  # More accurate than np.exp(small_values) - 1
print(exp_result)

### 5. Precision Control with np.float32 and np.float64

# Create arrays with specified precision.
arr_single_precision = np.array([1.123456789], dtype=np.float32)  # Single precision (32-bit)
arr_double_precision = np.array([1.123456789], dtype=np.float64)  # Double precision (64-bit)
print(arr_single_precision)  # Output may lose precision
print(arr_double_precision)  # More precise output

### 6. Floating-Point Sum with Reduced Error

# To reduce precision errors when summing many floating-point numbers,
# specify dtype in np.sum().
arr = np.random.rand(1000).astype(np.float32)
sum_result = np.sum(arr, dtype=np.float64)  # Use double precision for the sum
print(sum_result)

### 7. Higher Precision Polynomial Evaluation

# Evaluate polynomials with high precision using np.polyval().
coefficients = np.array([1.0, -2.0, 1.0])  # Coefficients for x^2 - 2x + 1
x_value = np.float64(1.0000001)  # High precision input
polynomial_value = np.polyval(coefficients, x_value)
print(polynomial_value)

### 8. Handling Small Floating-Point Differences in Array Comparisons

# When comparing floating-point arrays, use np.isclose() to handle precision errors.
arr1 = np.array([0.1 + 0.2, 0.3])
arr2 = np.array([0.3, 0.3])
print(arr1 == arr2)  # Direct comparison may fail: Output: [False  True]
print(np.isclose(arr1, arr2))  # Use isclose() for comparison: Output: [ True  True ]

'''To create structured array'''
data_type=[('name','U10'),('age','i4'),('wieght','f4')] #i4 is int f4 is float
arr=np.array([('Rajat',18,55),('Devdath',1,100)],dtype=data_type)


# 1. Using as_strided: Create a view of an array with custom strides.
# -------------------------------------------------------------------
# Let's create an array and use as_strided to generate a sliding window without copying the data.
x = np.array([1, 2, 3, 4])

# as_strided allows us to specify a custom shape and stride. Here we create overlapping 2-element windows.
# The stride (in bytes) between successive elements is set to 8 (size of a float64 in bytes).
windowed_x = np.lib.stride_tricks.as_strided(x, shape=(3, 2), strides=(8, 8))
print("1. as_strided result:")
print(windowed_x)
# Output: 
# [[1 2]
#  [2 3]
#  [3 4]]
# Explanation: This creates overlapping windows without copying the data. The stride tells how far to move in memory.


# 2. Using sliding_window_view: A convenient function to get sliding windows.
# ---------------------------------------------------------------------------
# sliding_window_view abstracts away the need to manually specify strides and is safer to use.
window_view = np.lib.stride_tricks.sliding_window_view(x, window_shape=2)
print("\n2. sliding_window_view result:")
print(window_view)
# Output:
# [[1 2]
#  [2 3]
#  [3 4]]
# Explanation: sliding_window_view generates the same result as as_strided, but it is safer and easier to use.


# 3. Using numpy.broadcast: Manually creating a broadcast object.
# ----------------------------------------------------------------
# Broadcasting allows us to work with arrays of different shapes as if they had the same shape.
a = np.array([1, 2, 3])
b = np.array([[4], [5], [6]])

# The broadcast object can iterate over the arrays as if they had the same shape.
broadcasted = np.broadcast(a, b)
print("\n3. broadcast object shape:")
print(broadcasted.shape)
# Output: (3, 3)
# Explanation: Broadcasting expands the arrays a and b to act as if they have shape (3, 3).


# 4. Using broadcast_to: Broadcasting an array to a new shape.
# -------------------------------------------------------------
# broadcast_to returns a view of the input array broadcasted to a new shape without copying the data.
broadcasted_a = np.broadcast_to(a, (3, 3))
print("\n4. broadcast_to result:")
print(broadcasted_a)
# Output:
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]]
# Explanation: broadcast_to expands the shape of array `a` to (3, 3), repeating its values in a memory-efficient way.


# 5. Using broadcast_arrays: Broadcasting multiple arrays to a common shape.
# ---------------------------------------------------------------------------
# broadcast_arrays takes multiple arrays and returns views of the arrays broadcasted to the same shape.
broadcasted_a, broadcasted_b = np.lib.stride_tricks.broadcast_arrays(a, b)
print("\n5. broadcast_arrays result:")
print("Broadcasted array a:")
print(broadcasted_a)
# Output:
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]]
print("Broadcasted array b:")
print(broadcasted_b)
# Output:
# [[4 4 4]
#  [5 5 5]
#  [6 6 6]]
# Explanation: broadcast_arrays broadcasts both arrays to the common shape (3, 3), repeating elements where necessary.


# Summary of Results:
# 1. as_strided allows us to create custom views with overlapping data.
# 2. sliding_window_view is a safer and more user-friendly function to create sliding windows.
# 3. broadcast and broadcast_to expand arrays' dimensions in a memory-efficient way.
# 4. broadcast_arrays synchronizes multiple arrays to a common shape using broadcasting.


# 1. Creating a datetime64 object
# --------------------------------
date = np.datetime64('2024-10-15')
print("1. np.datetime64:", date)
# Output: 2024-10-15

# 2. Creating datetime64 with time precision (e.g., hours, minutes)
# -----------------------------------------------------------------
datetime = np.datetime64('2024-10-15T12:30')
print("2. np.datetime64 with time:", datetime)
# Output: 2024-10-15T12:30

# 3. Creating a timedelta64 object (time difference)
# --------------------------------------------------
delta_days = np.timedelta64(5, 'D')  # 5 days
print("3. np.timedelta64 (days):", delta_days)
# Output: 5 days

delta_hours = np.timedelta64(12, 'h')  # 12 hours
print("3. np.timedelta64 (hours):", delta_hours)
# Output: 12 hours

# 4. Adding a timedelta64 to a datetime64
# ----------------------------------------
new_date = date + delta_days
print("4. Adding timedelta to datetime64:", new_date)
# Output: 2024-10-20

# 5. Subtracting a timedelta64 from a datetime64
# ----------------------------------------------
subtracted_date = date - delta_days
print("5. Subtracting timedelta from datetime64:", subtracted_date)
# Output: 2024-10-10

# 6. Difference between two datetime64 objects (returns timedelta64)
# ------------------------------------------------------------------
date1 = np.datetime64('2024-10-15')
date2 = np.datetime64('2024-10-10')
date_diff = date1 - date2
print("6. Difference between two dates:", date_diff)
# Output: 5 days

# 7. Creating datetime64 array with automatic intervals (date range)
# -------------------------------------------------------------------
date_range = np.arange('2024-10-01', '2024-10-10', dtype='datetime64[D]')
print("7. Date range (arange):", date_range)
# Output: ['2024-10-01' '2024-10-02' ... '2024-10-09']

# 8. Creating datetime64 array with a specific frequency (linspace)
# ------------------------------------------------------------------
date_linspace = np.linspace(np.datetime64('2024-10-01'), np.datetime64('2024-10-10'), 5, dtype='datetime64[D]')
print("8. Date range (linspace):", date_linspace)
# Output: ['2024-10-01' '2024-10-03' '2024-10-05' '2024-10-07' '2024-10-10']

# 9. Converting a datetime64 to a different unit (e.g., to minutes)
# -----------------------------------------------------------------
datetime_minute = np.datetime64('2024-10-15T12:30', 'm')
print("9. Convert datetime to minutes:", datetime_minute)
# Output: 2024-10-15T12:30

# 10. Getting the current date and time using `datetime64('now')`
# ---------------------------------------------------------------
current_datetime = np.datetime64('now')
print("10. Current datetime:", current_datetime)
# Output: e.g., 2024-10-15T12:45

# 11. Extracting the year, month, day, etc. from a datetime64
# ------------------------------------------------------------
date = np.datetime64('2024-10-15T12:30')
year = date.astype('datetime64[Y]')
month = date.astype('datetime64[M]')
day = date.astype('datetime64[D]')
hour = date.astype('datetime64[h]')
minute = date.astype('datetime64[m]')

print("11. Year:", year)
print("11. Month:", month)
print("11. Day:", day)
print("11. Hour:", hour)
print("11. Minute:", minute)
# Output:
# Year: 2024
# Month: 2024-10
# Day: 2024-10-15
# Hour: 2024-10-15T12
# Minute: 2024-10-15T12:30

# 12. Creating timedelta64 in units like years, months, hours, minutes
# ---------------------------------------------------------------------
delta_years = np.timedelta64(1, 'Y')  # 1 year
delta_months = np.timedelta64(2, 'M')  # 2 months
delta_minutes = np.timedelta64(30, 'm')  # 30 minutes
delta_seconds = np.timedelta64(120, 's')  # 120 seconds (2 minutes)

print("12. Timedelta in years:", delta_years)
print("12. Timedelta in months:", delta_months)
print("12. Timedelta in minutes:", delta_minutes)
print("12. Timedelta in seconds:", delta_seconds)
# Output:
# Timedelta in years: 1 years
# Timedelta in months: 2 months
# Timedelta in minutes: 30 minutes
# Timedelta in seconds: 120 seconds

# 13. Calculating number of days between two datetime64 dates
# ------------------------------------------------------------
date1 = np.datetime64('2024-10-01')
date2 = np.datetime64('2024-10-15')
days_diff = (date2 - date1) / np.timedelta64(1, 'D')
print("13. Number of days between two dates:", days_diff)
# Output: 14.0 days

# 14. Convert string of dates to datetime64
# ------------------------------------------
dates = np.array(['2024-10-01', '2024-10-15', '2024-10-20'], dtype='datetime64')
print("14. Convert string array to datetime64:", dates)
# Output: ['2024-10-01' '2024-10-15' '2024-10-20']

# 15. Convert datetime64 back to string format
# --------------------------------------------
string_dates = dates.astype(str)
print("15. Convert datetime64 back to string:", string_dates)
# Output: ['2024-10-01' '2024-10-15' '2024-10-20']


# ----------------------- ADVANCED INDEXING -----------------------

# Advanced indexing using boolean arrays
arr = np.array([1, 2, 3, 4, 5])
bool_index = np.array([True, False, True, False, True])
advanced_indexed_arr = arr[bool_index]  # Select elements where bool_index is True
print(advanced_indexed_arr)  # Output: [1 3 5]

# Advanced indexing using integer arrays
int_index = np.array([0, 2, 4])
advanced_indexed_arr = arr[int_index]  # Select elements at specified indices
print(advanced_indexed_arr)  # Output: [1 3 5]

# ----------------------- MEMORY MANAGEMENT -----------------------

# Create a memory-mapped file
memmap_arr = np.memmap('memmap_file.dat', dtype=np.float64, mode='w+', shape=(100, 100))
print(memmap_arr)  # Output: memmap((100, 100), dtype=float64)

# Write data to the memory-mapped file
memmap_arr[:] = np.random.rand(100, 100)
print(memmap_arr)  # Output: memmap((100, 100), dtype=float64)

# Close the memory-mapped file
del memmap_arr

# ----------------------- SAVING AND LOADING ARRAYS -----------------------

# Save an array to a file
arr = np.random.rand(100, 100)
np.save('array_file.npy', arr)

# Load an array from a file
loaded_arr = np.load('array_file.npy')
print(loaded_arr)  # Output: loaded array

# Save an array to a text file
np.savetxt('array_file.txt', arr)

# Load an array from a text file
loaded_arr = np.loadtxt('array_file.txt')
print(loaded_arr)  # Output: loaded array

# ----------------------- NAN MEAN AND NAN STD -----------------------
# Compute mean and standard deviation while ignoring NaN values
arr = np.array([1, 2, np.nan, 4])
mean_value = np.nanmean(arr)  # Mean ignoring NaN values
std_value = np.nanstd(arr)  # Standard deviation ignoring NaN values
# mean_value will be 2.3333 and std_value will be calculated accordingly

# ----------------------- CONCATENATE MORE THAN TWO ARRAYS -----------------------
# Joining multiple arrays together
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr3 = np.array([7, 8, 9])
arr = np.concatenate((arr1, arr2, arr3))  # Concatenate multiple arrays
# arr will be [1, 2, 3, 4, 5, 6, 7, 8, 9]

# ----------------------- EXPAND DIMS -----------------------
# Add a new axis to an array using np.expand_dims
arr = np.array([1, 2, 3])
expanded_arr = np.expand_dims(arr, axis=1)  # Adds a new axis, converting it to a column vector
# expanded_arr will be [[1], [2], [3]]

# ----------------------- CLIP WITH OUT PARAMETER -----------------------
# Clip values in an array to a specified range and store in a preallocated array
arr = np.array([1, 2, 3, 4, 5])
clipped_arr = np.empty_like(arr)  # Preallocate an array for the result
np.clip(arr, 2, 4, out=clipped_arr)  # Clip values between 2 and 4
# clipped_arr will contain the clipped values

# ----------------------- ADVANCED INDEXING WITH MULTI-DIMENSIONAL ARRAYS -----------------------
# Advanced indexing using integer arrays with multi-dimensional arrays
arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
row_indices = np.array([0, 1, 2])
col_indices = np.array([2, 1, 0])
advanced_indexed_values = arr[row_indices, col_indices]  # Select elements at specified indices
# advanced_indexed_values will be [30, 50, 70]