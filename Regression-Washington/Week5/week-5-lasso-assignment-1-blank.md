
# Regression Week 5: Feature Selection and LASSO (Interpretation)

In this notebook, you will use LASSO to select features, building on a pre-implemented solver for LASSO (using GraphLab Create, though you can use other solvers). You will:
* Run LASSO with different L1 penalties.
* Choose best L1 penalty using a validation set.
* Choose best L1 penalty using a validation set, with additional constraint on the size of subset.

In the second notebook, you will implement your own LASSO solver, using coordinate descent. 

# Fire up graphlab create


```python
import graphlab
```

# Load in house sales data

Dataset is from house sales in King County, the region where the city of Seattle, WA is located.


```python
sales = graphlab.SFrame('kc_house_data.gl/')
```

    This non-commercial license of GraphLab Create is assigned to 770188954@qq.com and will expire on January 14, 2017. For commercial licensing options, visit https://dato.com/buy/.


    2016-05-01 15:17:30,416 [INFO] graphlab.cython.cy_server, 176: GraphLab Create v1.9 started. Logging: /tmp/graphlab_server_1462087045.log


# Create new features

As in Week 2, we consider features that are some transformations of inputs.


```python
from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']

# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to float, before creating a new feature.
sales['floors'] = sales['floors'].astype(float) 
sales['floors_square'] = sales['floors']*sales['floors']
```

* Squaring bedrooms will increase the separation between not many bedrooms (e.g. 1) and lots of bedrooms (e.g. 4) since 1^2 = 1 but 4^2 = 16. Consequently this variable will mostly affect houses with many bedrooms.
* On the other hand, taking square root of sqft_living will decrease the separation between big house and small house. The owner may not be exactly twice as happy for getting a house that is twice as big.

# Learn regression weights with L1 penalty

Let us fit a model with all the features available, plus the features we just created above.


```python
all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']
```

Applying L1 penalty requires adding an extra parameter (`l1_penalty`) to the linear regression call in GraphLab Create. (Other tools may have separate implementations of LASSO.)  Note that it's important to set `l2_penalty=0` to ensure we don't introduce an additional L2 penalty.


```python
model_all = graphlab.linear_regression.create(sales, target='price', features=all_features,
                                              validation_set=None, 
                                              l2_penalty=0., l1_penalty=1e10)
```


<pre>Linear regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 21613</pre>



<pre>Number of features          : 17</pre>



<pre>Number of unpacked features : 17</pre>



<pre>Number of coefficients    : 18</pre>



<pre>Starting Accelerated Gradient (FISTA)</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+-----------+--------------+--------------------+---------------+</pre>



<pre>| Iteration | Passes   | Step size | Elapsed Time | Training-max_error | Training-rmse |</pre>



<pre>+-----------+----------+-----------+--------------+--------------------+---------------+</pre>



<pre>Tuning step size. First iteration could take longer than subsequent iterations.</pre>



<pre>| 1         | 2        | 0.000002  | 1.223721     | 6962915.603493     | 426631.749026 |</pre>



<pre>| 2         | 3        | 0.000002  | 1.241054     | 6843144.200219     | 392488.929838 |</pre>



<pre>| 3         | 4        | 0.000002  | 1.262938     | 6831900.032123     | 385340.166783 |</pre>



<pre>| 4         | 5        | 0.000002  | 1.280870     | 6847166.848958     | 384842.383767 |</pre>



<pre>| 5         | 6        | 0.000002  | 1.303306     | 6869667.895833     | 385998.458623 |</pre>



<pre>| 6         | 7        | 0.000002  | 1.323637     | 6847177.773672     | 380824.455891 |</pre>



<pre>+-----------+----------+-----------+--------------+--------------------+---------------+</pre>



<pre>TERMINATED: Iteration limit reached.</pre>



<pre>This model may not be optimal. To improve it, consider increasing `max_iterations`.</pre>


Find what features had non-zero weight.


```python
model_all.get("coefficients").print_rows(18)
```

    +------------------+-------+---------------+--------+
    |       name       | index |     value     | stderr |
    +------------------+-------+---------------+--------+
    |   (intercept)    |  None |  274873.05595 |  None  |
    |     bedrooms     |  None |      0.0      |  None  |
    | bedrooms_square  |  None |      0.0      |  None  |
    |    bathrooms     |  None | 8468.53108691 |  None  |
    |   sqft_living    |  None | 24.4207209824 |  None  |
    | sqft_living_sqrt |  None | 350.060553386 |  None  |
    |     sqft_lot     |  None |      0.0      |  None  |
    |  sqft_lot_sqrt   |  None |      0.0      |  None  |
    |      floors      |  None |      0.0      |  None  |
    |  floors_square   |  None |      0.0      |  None  |
    |    waterfront    |  None |      0.0      |  None  |
    |       view       |  None |      0.0      |  None  |
    |    condition     |  None |      0.0      |  None  |
    |      grade       |  None | 842.068034898 |  None  |
    |    sqft_above    |  None | 20.0247224171 |  None  |
    |  sqft_basement   |  None |      0.0      |  None  |
    |     yr_built     |  None |      0.0      |  None  |
    |   yr_renovated   |  None |      0.0      |  None  |
    +------------------+-------+---------------+--------+
    [18 rows x 4 columns]
    


Note that a majority of the weights have been set to zero. So by setting an L1 penalty that's large enough, we are performing a subset selection. 

***QUIZ QUESTION***:
According to this list of weights, which of the features have been chosen? 

# Selecting an L1 penalty

To find a good L1 penalty, we will explore multiple values using a validation set. Let us do three way split into train, validation, and test sets:
* Split our sales data into 2 sets: training and test
* Further split our training data into two sets: train, validation

Be *very* careful that you use seed = 1 to ensure you get the same answer!


```python
(training_and_validation, testing) = sales.random_split(.9,seed=1) # initial train/test split
(training, validation) = training_and_validation.random_split(0.5, seed=1) # split training into train and validate
```

Next, we write a loop that does the following:
* For `l1_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, type `np.logspace(1, 7, num=13)`.)
    * Fit a regression model with a given `l1_penalty` on TRAIN data. Specify `l1_penalty=l1_penalty` and `l2_penalty=0.` in the parameter list.
    * Compute the RSS on VALIDATION data (here you will want to use `.predict()`) for that `l1_penalty`
* Report which `l1_penalty` produced the lowest RSS on validation data.

When you call `linear_regression.create()` make sure you set `validation_set = None`.

Note: you can turn off the print out of `linear_regression.create()` with `verbose = False`


```python
import numpy as np
RSS=[]
penaltys=[]
for l1_penalty in np.logspace(1,7,num=13):
    model=graphlab.linear_regression.create(training,
                                            features=all_features,
                                            target="price",
                                            validation_set=None,
                                            l2_penalty=0,
                                            l1_penalty=l1_penalty,
                                            verbose=False)
    est= model.predict(validation)
    rss=sum((est-validation['price'])**2)
    penaltys.append(l1_penalty)
    RSS.append(rss)
    print rss
```

    6.25766285142e+14
    6.25766285362e+14
    6.25766286058e+14
    6.25766288257e+14
    6.25766295212e+14
    6.25766317206e+14
    6.25766386761e+14
    6.25766606749e+14
    6.25767302792e+14
    6.25769507644e+14
    6.25776517727e+14
    6.25799062845e+14
    6.25883719085e+14



```python
mini=RSS[0]
min_penalty=penaltys[0]
i=-1
for ele in RSS:
    i=i+1
    if ele < mini:
        mini=ele
        mini_penalty=penaltys[i]
print mini,min_penalty
```

    6.25766285142e+14 10.0


*** QUIZ QUESTIONS ***
1. What was the best value for the `l1_penalty`?
2. What is the RSS on TEST data of the model with the best `l1_penalty`?


```python
model=graphlab.linear_regression.create(training,
                                            features=all_features,
                                            target="price",
                                            validation_set=None,
                                            l2_penalty=0,
                                            l1_penalty=10,
                                            verbose=False)
sum((model.predict(testing)-testing['price'])**2)
```




    156983602381664.4



***QUIZ QUESTION***
Also, using this value of L1 penalty, how many nonzero weights do you have?


```python
model.get("coefficients").print_rows(18)
```

    +------------------+-------+------------------+--------+
    |       name       | index |      value       | stderr |
    +------------------+-------+------------------+--------+
    |   (intercept)    |  None |  18993.4272128   |  None  |
    |     bedrooms     |  None |  7936.96767903   |  None  |
    | bedrooms_square  |  None |  936.993368193   |  None  |
    |    bathrooms     |  None |  25409.5889341   |  None  |
    |   sqft_living    |  None |  39.1151363797   |  None  |
    | sqft_living_sqrt |  None |  1124.65021281   |  None  |
    |     sqft_lot     |  None | 0.00348361822299 |  None  |
    |  sqft_lot_sqrt   |  None |  148.258391011   |  None  |
    |      floors      |  None |   21204.335467   |  None  |
    |  floors_square   |  None |  12915.5243361   |  None  |
    |    waterfront    |  None |  601905.594545   |  None  |
    |       view       |  None |  93312.8573119   |  None  |
    |    condition     |  None |  6609.03571245   |  None  |
    |      grade       |  None |  6206.93999188   |  None  |
    |    sqft_above    |  None |  43.2870534193   |  None  |
    |  sqft_basement   |  None |  122.367827534   |  None  |
    |     yr_built     |  None |  9.43363539372   |  None  |
    |   yr_renovated   |  None |  56.0720034488   |  None  |
    +------------------+-------+------------------+--------+
    [18 rows x 4 columns]
    


# Limit the number of nonzero weights

What if we absolutely wanted to limit ourselves to, say, 7 features? This may be important if we want to derive "a rule of thumb" --- an interpretable model that has only a few features in them.

In this section, you are going to implement a simple, two phase procedure to achive this goal:
1. Explore a large range of `l1_penalty` values to find a narrow region of `l1_penalty` values where models are likely to have the desired number of non-zero weights.
2. Further explore the narrow region you found to find a good value for `l1_penalty` that achieves the desired sparsity.  Here, we will again use a validation set to choose the best value for `l1_penalty`.


```python
max_nonzeros = 7
```

## Exploring the larger range of values to find a narrow range with the desired sparsity

Let's define a wide range of possible `l1_penalty_values`:


```python
l1_penalty_values = np.logspace(8, 10, num=20)
```

Now, implement a loop that search through this space of possible `l1_penalty` values:

* For `l1_penalty` in `np.logspace(8, 10, num=20)`:
    * Fit a regression model with a given `l1_penalty` on TRAIN data. Specify `l1_penalty=l1_penalty` and `l2_penalty=0.` in the parameter list. When you call `linear_regression.create()` make sure you set `validation_set = None`
    * Extract the weights of the model and count the number of nonzeros. Save the number of nonzeros to a list.
        * *Hint: `model['coefficients']['value']` gives you an SArray with the parameters you learned.  If you call the method `.nnz()` on it, you will find the number of non-zero parameters!* 


```python
nonzeros=[]
penaltys=[]
for l1_penalty in np.logspace(8,10,num=13):
    model=graphlab.linear_regression.create(training,
                                            features=all_features,
                                            target="price",
                                            validation_set=None,
                                            l2_penalty=0,
                                            l1_penalty=l1_penalty,
                                            verbose=False)
    coefficient=model.get("coefficients")
    nonzeros.append(sum(coefficient['value']!=0))
    penaltys.append(l1_penalty)
print nonzeros
print penaltys
```

    [18, 18, 18, 17, 17, 17, 16, 15, 12, 10, 6, 2, 1]
    [100000000.0, 146779926.76220676, 215443469.00318867, 316227766.01683795, 464158883.36127728, 681292069.05796218, 1000000000.0, 1467799267.6220675, 2154434690.0318866, 3162277660.1683793, 4641588833.6127729, 6812920690.5796223, 10000000000.0]


Out of this large range, we want to find the two ends of our desired narrow range of `l1_penalty`.  At one end, we will have `l1_penalty` values that have too few non-zeros, and at the other end, we will have an `l1_penalty` that has too many non-zeros.  

More formally, find:
* The largest `l1_penalty` that has more non-zeros than `max_nonzero` (if we pick a penalty smaller than this value, we will definitely have too many non-zero weights)
    * Store this value in the variable `l1_penalty_min` (we will use it later)
* The smallest `l1_penalty` that has fewer non-zeros than `max_nonzero` (if we pick a penalty larger than this value, we will definitely have too few non-zero weights)
    * Store this value in the variable `l1_penalty_max` (we will use it later)


*Hint: there are many ways to do this, e.g.:*
* Programmatically within the loop above
* Creating a list with the number of non-zeros for each value of `l1_penalty` and inspecting it to find the appropriate boundaries.


```python
l1_penalty_min = penaltys[9]
l1_penalty_max = penaltys[10]
print l1_penalty_min,l1_penalty_max
```

    3162277660.17 4641588833.61


***QUIZ QUESTIONS***

What values did you find for `l1_penalty_min` and`l1_penalty_max`? 

## Exploring the narrow range of values to find the solution with the right number of non-zeros that has lowest RSS on the validation set 

We will now explore the narrow region of `l1_penalty` values we found:


```python
l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)
```

* For `l1_penalty` in `np.linspace(l1_penalty_min,l1_penalty_max,20)`:
    * Fit a regression model with a given `l1_penalty` on TRAIN data. Specify `l1_penalty=l1_penalty` and `l2_penalty=0.` in the parameter list. When you call `linear_regression.create()` make sure you set `validation_set = None`
    * Measure the RSS of the learned model on the VALIDATION set

Find the model that the lowest RSS on the VALIDATION set and has sparsity *equal* to `max_nonzero`.


```python
RSS=[]
penaltys=[]
non_zeros=[]
for l1_penalty in np.linspace(l1_penalty_min,l1_penalty_max,20):
    model=graphlab.linear_regression.create(training,
                                            features=all_features,
                                            target="price",
                                            validation_set=None,
                                            l2_penalty=0,
                                            l1_penalty=l1_penalty,
                                            verbose=False)
    est= model.predict(validation)
    rss=sum((est-validation['price'])**2)
    penaltys.append(l1_penalty)
    RSS.append(rss)
    non_zeros.append(sum(model['coefficients']['value']!=0))
    
print RSS
print non_zeros
```

    [1001942638318607.4, 1019679000386924.9, 1034427058376774.0, 1042086071169676.6, 1049103004858990.1, 1057858745143422.9, 1066458162712115.1, 1073555095721075.4, 1081088683843512.8, 1089282174316465.1, 1097981325825341.5, 1107161130602021.4, 1118115604316346.8, 1129162828252089.2, 1140267600282791.2, 1153506410262662.5, 1167064941197950.5, 1180824492981377.5, 1194804565514781.8, 1208943977016249.2]
    [10, 10, 8, 8, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]


***QUIZ QUESTIONS***
1. What value of `l1_penalty` in our narrow range has the lowest RSS on the VALIDATION set and has sparsity *equal* to `max_nonzeros`?
2. What features in this model have non-zero coefficients?


```python
print penaltys[4]
model=graphlab.linear_regression.create(training,
                                            features=all_features,
                                            target="price",
                                            validation_set=None,
                                            l2_penalty=0,
                                            l1_penalty=penaltys[4],
                                            verbose=False)
model.get('coefficients').print_rows(18)
```

    3473711591.42
    +------------------+-------+---------------+--------+
    |       name       | index |     value     | stderr |
    +------------------+-------+---------------+--------+
    |   (intercept)    |  None | 223585.832873 |  None  |
    |     bedrooms     |  None | 571.154073813 |  None  |
    | bedrooms_square  |  None |      0.0      |  None  |
    |    bathrooms     |  None | 15745.9460397 |  None  |
    |   sqft_living    |  None | 32.2981629056 |  None  |
    | sqft_living_sqrt |  None | 683.974473503 |  None  |
    |     sqft_lot     |  None |      0.0      |  None  |
    |  sqft_lot_sqrt   |  None |      0.0      |  None  |
    |      floors      |  None |      0.0      |  None  |
    |  floors_square   |  None |      0.0      |  None  |
    |    waterfront    |  None |      0.0      |  None  |
    |       view       |  None |      0.0      |  None  |
    |    condition     |  None |      0.0      |  None  |
    |      grade       |  None | 2858.52537692 |  None  |
    |    sqft_above    |  None | 29.8468590203 |  None  |
    |  sqft_basement   |  None |      0.0      |  None  |
    |     yr_built     |  None |      0.0      |  None  |
    |   yr_renovated   |  None |      0.0      |  None  |
    +------------------+-------+---------------+--------+
    [18 rows x 4 columns]
    

