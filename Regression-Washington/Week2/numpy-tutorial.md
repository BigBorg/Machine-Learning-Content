
#Numpy Tutorial

Numpy is a computational library for Python that is optimized for operations on multi-dimensional arrays. In this notebook we will use numpy to work with 1-d arrays (often called vectors) and 2-d arrays (often called matrices).

For a the full user guide and reference for numpy see: http://docs.scipy.org/doc/numpy/


```python
import numpy as np # importing this way allows us to refer to numpy as np
```

# Creating Numpy Arrays

New arrays can be made in several ways. We can take an existing list and convert it to a numpy array:


```python
mylist = [1., 2., 3., 4.]
mynparray = np.array(mylist)
mynparray
```




    array([ 1.,  2.,  3.,  4.])



You can initialize an array (of any dimension) of all ones or all zeroes with the ones() and zeros() functions:


```python
one_vector = np.ones(4)
print one_vector # using print removes the array() portion
```

    [ 1.  1.  1.  1.]



```python
one2Darray = np.ones((2, 4)) # an 2D array with 2 "rows" and 4 "columns"
print one2Darray
```

    [[ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]]



```python
zero_vector = np.zeros(4)
print zero_vector
```

    [ 0.  0.  0.  0.]


You can also initialize an empty array which will be filled with values. This is the fastest way to initialize a fixed-size numpy array however you must ensure that you replace all of the values.


```python
empty_vector = np.empty(5)
print empty_vector
```

    [  8.74993421e-317   7.90436259e-317   6.91660455e-310   3.97643794e-317
       2.37151510e-322]


#Accessing array elements

Accessing an array is straight forward. For vectors you access the index by referring to it inside square brackets. Recall that indices in Python start with 0.


```python
mynparray[2]
```




    3.0



2D arrays are accessed similarly by referring to the row and column index separated by a comma:


```python
my_matrix = np.array([[1, 2, 3], [4, 5, 6]])
print my_matrix
```

    [[1 2 3]
     [4 5 6]]



```python
print my_matrix[1, 2]
```

    6


Sequences of indices can be accessed using ':' for example


```python
print my_matrix[0:2, 2] # recall 0:2 = [0, 1]
```

    [3 6]



```python
print my_matrix[0, 0:3]
```

    [1 2 3]


You can also pass a list of indices. 


```python
fib_indices = np.array([1, 1, 2, 3])
random_vector = np.random.random(10) # 10 random numbers between 0 and 1
print random_vector
```

    [ 0.51773286  0.91335982  0.74303847  0.91654891  0.45731365  0.72120271
      0.05532227  0.84926229  0.00282214  0.59791861]



```python
print random_vector[fib_indices]
```

    [ 0.91335982  0.91335982  0.74303847  0.91654891]


You can also use true/false values to select values


```python
my_vector = np.array([1, 2, 3, 4])
select_index = np.array([True, False, True, False])
print my_vector[select_index]
```

    [1 3]


For 2D arrays you can select specific columns and specific rows. Passing ':' selects all rows/columns


```python
select_cols = np.array([True, False, True]) # 1st and 3rd column
select_rows = np.array([False, True]) # 2nd row
```


```python
print my_matrix[select_rows, :] # just 2nd row but all columns
```

    [[4 5 6]]



```python
print my_matrix[:, select_cols] # all rows and just the 1st and 3rd column
```

    [[1 3]
     [4 6]]


#Operations on Arrays

You can use the operations '\*', '\*\*', '\\', '+' and '-' on numpy arrays and they operate elementwise.


```python
my_array = np.array([1., 2., 3., 4.])
print my_array*my_array
```

    [  1.   4.   9.  16.]



```python
print my_array**2
```

    [  1.   4.   9.  16.]



```python
print my_array - np.ones(4)
```

    [ 0.  1.  2.  3.]



```python
print my_array + np.ones(4)
```

    [ 2.  3.  4.  5.]



```python
print my_array / 3
```

    [ 0.33333333  0.66666667  1.          1.33333333]



```python
print my_array / np.array([2., 3., 4., 5.]) # = [1.0/2.0, 2.0/3.0, 3.0/4.0, 4.0/5.0]
```

    [ 0.5         0.66666667  0.75        0.8       ]


You can compute the sum with np.sum() and the average with np.average()


```python
print np.sum(my_array)
```

    10.0



```python
print np.average(my_array)
```

    2.5



```python
print np.sum(my_array)/len(my_array)
```

    2.5


#The dot product

An important mathematical operation in linear algebra is the dot product. 

When we compute the dot product between two vectors we are simply multiplying them elementwise and adding them up. In numpy you can do this with np.dot()


```python
array1 = np.array([1., 2., 3., 4.])
array2 = np.array([2., 3., 4., 5.])
print np.dot(array1, array2)
```

    40.0



```python
print np.sum(array1*array2)
```

    40.0


Recall that the Euclidean length (or magnitude) of a vector is the squareroot of the sum of the squares of the components. This is just the squareroot of the dot product of the vector with itself:


```python
array1_mag = np.sqrt(np.dot(array1, array1))
print array1_mag
```

    5.47722557505



```python
print np.sqrt(np.sum(array1*array1))
```

    5.47722557505


We can also use the dot product when we have a 2D array (or matrix). When you have an vector with the same number of elements as the matrix (2D array) has columns you can right-multiply the matrix by the vector to get another vector with the same number of elements as the matrix has rows. For example this is how you compute the predicted values given a matrix of features and an array of weights.


```python
my_features = np.array([[1., 2.], [3., 4.], [5., 6.], [7., 8.]])
print my_features
```

    [[ 1.  2.]
     [ 3.  4.]
     [ 5.  6.]
     [ 7.  8.]]



```python
my_weights = np.array([0.4, 0.5])
print my_weights
```

    [ 0.4  0.5]



```python
my_predictions = np.dot(my_features, my_weights) # note that the weights are on the right
print my_predictions # which has 4 elements since my_features has 4 rows
```

    [ 1.4  3.2  5.   6.8]


Similarly if you have a vector with the same number of elements as the matrix has *rows* you can left multiply them.


```python
my_matrix = my_features
my_array = np.array([0.3, 0.4, 0.5, 0.6])
```


```python
 which has 2 elements because my_matrix has 2 columnsprint np.dot(my_array, my_matrix) #
```


      File "<ipython-input-35-e848acda58c6>", line 1
        which has 2 elements because my_matrix has 2 columnsprint np.dot(my_array, my_matrix) #
                ^
    SyntaxError: invalid syntax



#Multiplying Matrices

If we have two 2D arrays (matrices) matrix_1 and matrix_2 where the number of columns of matrix_1 is the same as the number of rows of matrix_2 then we can use np.dot() to perform matrix multiplication.


```python
matrix_1 = np.array([[1., 2., 3.],[4., 5., 6.]])
print matrix_1
```


```python
matrix_2 = np.array([[1., 2.], [3., 4.], [5., 6.]])
print matrix_2
```


```python
print np.dot(matrix_1, matrix_2)
```
