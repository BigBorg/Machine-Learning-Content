
# Regression Week 4: Ridge Regression (interpretation)

In this notebook, we will run ridge regression multiple times with different L2 penalties to see which one produces the best fit. We will revisit the example of polynomial regression as a means to see the effect of L2 regularization. In particular, we will:
* Use a pre-built implementation of regression (GraphLab Create) to run polynomial regression
* Use matplotlib to visualize polynomial regressions
* Use a pre-built implementation of regression (GraphLab Create) to run polynomial regression, this time with L2 penalty
* Use matplotlib to visualize polynomial regressions under L2 regularization
* Choose best L2 penalty using cross-validation.
* Assess the final fit using test data.

We will continue to use the House data from previous notebooks.  (In the next programming assignment for this module, you will implement your own ridge regression learning algorithm using gradient descent.)

# Fire up graphlab create


```python
import graphlab
```

# Polynomial regression, revisited

We build on the material from Week 3, where we wrote the function to produce an SFrame with columns containing the powers of a given input. Copy and paste the function `polynomial_sframe` from Week 3:


```python
def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1']=feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name]=feature**power
    return poly_sframe
```

Let's use matplotlib to visualize what a polynomial regression looks like on the house data.


```python
import matplotlib.pyplot as plt
%matplotlib inline
```

    This non-commercial license of GraphLab Create is assigned to 770188954@qq.com and will expire on January 14, 2017. For commercial licensing options, visit https://dato.com/buy/.


    2016-05-01 15:16:19,347 [INFO] graphlab.cython.cy_server, 176: GraphLab Create v1.9 started. Logging: /tmp/graphlab_server_1462086973.log



```python
sales = graphlab.SFrame('kc_house_data.gl/')
```

As in Week 3, we will use the sqft_living variable. For plotting purposes (connecting the dots), you'll need to sort by the values of sqft_living. For houses with identical square footage, we break the tie by their prices.


```python
sales = sales.sort(['sqft_living','price'])
```

Let us revisit the 15th-order polynomial model using the 'sqft_living' input. Generate polynomial features up to degree 15 using `polynomial_sframe()` and fit a model with these features. When fitting the model, use an L2 penalty of `1e-5`:


```python
l2_small_penalty = 1e-5
```

Note: When we have so many features and so few data points, the solution can become highly numerically unstable, which can sometimes lead to strange unpredictable results.  Thus, rather than using no regularization, we will introduce a tiny amount of regularization (`l2_penalty=1e-5`) to make the solution numerically stable.  (In lecture, we discussed the fact that regularization can also help with numerical stability, and here we are seeing a practical example.)

With the L2 penalty specified above, fit the model and print out the learned weights.

Hint: make sure to add 'price' column to the new SFrame before calling `graphlab.linear_regression.create()`. Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set=None` in this call.


```python
poly_data=polynomial_sframe(sales['sqft_living'],15)
features=poly_data.column_names()
poly_data['price']=sales['price']
model1=graphlab.linear_regression.create(poly_data,features=features,target='price',l2_penalty=l2_small_penalty,validation_set=None,verbose=False)
model1.get("coefficients").print_rows(16,4)
```

    +-------------+-------+--------------------+-------------------+
    |     name    | index |       value        |       stderr      |
    +-------------+-------+--------------------+-------------------+
    | (intercept) |  None |   167924.858154    |   932257.208736   |
    |   power_1   |  None |   103.090949754    |   4735.64047203   |
    |   power_2   |  None |   0.134604553044   |   9.85916611863   |
    |   power_3   |  None | -0.000129071365146 |  0.0111681953814  |
    |   power_4   |  None | 5.18928960684e-08  | 7.69612934514e-06 |
    |   power_5   |  None | -7.77169308381e-12 | 3.40375283346e-09 |
    |   power_6   |  None | 1.71144848253e-16  | 9.86487827549e-13 |
    |   power_7   |  None | 4.51177961859e-20  | 1.85595597809e-16 |
    |   power_8   |  None | -4.78839845626e-25 | 2.13680017115e-20 |
    |   power_9   |  None | -2.33343504241e-28 | 1.22638027914e-24 |
    |   power_10  |  None | -7.29022430191e-33 | 3.82656972229e-29 |
    |   power_11  |  None | 7.22829191056e-37  | 4.29139903741e-33 |
    |   power_12  |  None | 6.90470812665e-41  |        nan        |
    |   power_13  |  None | -3.65844733582e-46 |  4.8034831595e-41 |
    |   power_14  |  None | -3.79575894541e-49 | 2.76882860034e-45 |
    |   power_15  |  None | 1.13723143004e-53  | 5.32281057886e-50 |
    +-------------+-------+--------------------+-------------------+
    [16 rows x 4 columns]
    


***QUIZ QUESTION:  What's the learned value for the coefficient of feature `power_1`?***

# Observe overfitting

Recall from Week 3 that the polynomial fit of degree 15 changed wildly whenever the data changed. In particular, when we split the sales data into four subsets and fit the model of degree 15, the result came out to be very different for each subset. The model had a *high variance*. We will see in a moment that ridge regression reduces such variance. But first, we must reproduce the experiment we did in Week 3.

First, split the data into split the sales data into four subsets of roughly equal size and call them `set_1`, `set_2`, `set_3`, and `set_4`. Use `.random_split` function and make sure you set `seed=0`. 


```python
(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)
```

Next, fit a 15th degree polynomial on `set_1`, `set_2`, `set_3`, and `set_4`, using 'sqft_living' to predict prices. Print the weights and make a plot of the resulting model.

Hint: When calling `graphlab.linear_regression.create()`, use the same L2 penalty as before (i.e. `l2_small_penalty`).  Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set = None` in this call.


```python
def poly_l2_regression(train,deg,l2_value):
    poly_data=polynomial_sframe(train['sqft_living'],deg)
    features=poly_data.column_names()
    poly_data['price']=train['price']
    model=graphlab.linear_regression.create(poly_data,features=features,target='price',l2_penalty=l2_value,validation_set=None,verbose=False)
    model.get('coefficients').print_rows(2,3)
    plt.plot(train['sqft_living'],train['price'],'.',
            train['sqft_living'],model.predict(poly_data),'-')
    return model
```


```python
poly_l2_regression(set_1,15,l2_small_penalty)
```

    +-------------+-------+---------------+-----+
    |     name    | index |     value     | ... |
    +-------------+-------+---------------+-----+
    | (intercept) |  None |  9306.4606221 | ... |
    |   power_1   |  None | 585.865823394 | ... |
    +-------------+-------+---------------+-----+
    [16 rows x 4 columns]
    





    Class                         : LinearRegression
    
    Schema
    ------
    Number of coefficients        : 16
    Number of examples            : 5404
    Number of feature columns     : 15
    Number of unpacked features   : 15
    
    Hyperparameters
    ---------------
    L1 penalty                    : 0.0
    L2 penalty                    : 0.0
    
    Training Summary
    ----------------
    Solver                        : newton
    Solver iterations             : 1
    Solver status                 : SUCCESS: Optimal solution found.
    Training time (sec)           : 0.016
    
    Settings
    --------
    Residual sum of squares       : 3.3424415999e+14
    Training RMSE                 : 248699.1173
    
    Highest Positive Coefficients
    -----------------------------
    (intercept)                   : 9306.4606
    power_1                       : 585.8658
    power_3                       : 0.0001
    power_6                       : 0.0
    power_7                       : 0.0
    
    Lowest Negative Coefficients
    ----------------------------
    power_2                       : -0.3973
    power_4                       : -0.0
    power_5                       : -0.0
    power_9                       : -0.0
    power_10                      : -0.0




![png](output_23_2.png)



```python
poly_l2_regression(set_2,15,l2_small_penalty)
```

    +-------------+-------+----------------+-----+
    |     name    | index |     value      | ... |
    +-------------+-------+----------------+-----+
    | (intercept) |  None | -25115.9044254 | ... |
    |   power_1   |  None |  783.49380028  | ... |
    +-------------+-------+----------------+-----+
    [16 rows x 4 columns]
    





    Class                         : LinearRegression
    
    Schema
    ------
    Number of coefficients        : 16
    Number of examples            : 5398
    Number of feature columns     : 15
    Number of unpacked features   : 15
    
    Hyperparameters
    ---------------
    L1 penalty                    : 0.0
    L2 penalty                    : 0.0
    
    Training Summary
    ----------------
    Solver                        : newton
    Solver iterations             : 1
    Solver status                 : SUCCESS: Optimal solution found.
    Training time (sec)           : 0.0127
    
    Settings
    --------
    Residual sum of squares       : 2.96922466393e+14
    Training RMSE                 : 234533.6106
    
    Highest Positive Coefficients
    -----------------------------
    power_1                       : 783.4938
    power_3                       : 0.0004
    power_5                       : 0.0
    power_6                       : 0.0
    power_10                      : 0.0
    
    Lowest Negative Coefficients
    ----------------------------
    (intercept)                   : -25115.9044
    power_2                       : -0.7678
    power_4                       : -0.0
    power_7                       : -0.0
    power_8                       : -0.0




![png](output_24_2.png)



```python
poly_l2_regression(set_3,15,l2_small_penalty)
```

    +-------------+-------+----------------+-----+
    |     name    | index |     value      | ... |
    +-------------+-------+----------------+-----+
    | (intercept) |  None | 462426.565731  | ... |
    |   power_1   |  None | -759.251842854 | ... |
    +-------------+-------+----------------+-----+
    [16 rows x 4 columns]
    





    Class                         : LinearRegression
    
    Schema
    ------
    Number of coefficients        : 16
    Number of examples            : 5409
    Number of feature columns     : 15
    Number of unpacked features   : 15
    
    Hyperparameters
    ---------------
    L1 penalty                    : 0.0
    L2 penalty                    : 0.0
    
    Training Summary
    ----------------
    Solver                        : newton
    Solver iterations             : 1
    Solver status                 : SUCCESS: Optimal solution found.
    Training time (sec)           : 0.0109
    
    Settings
    --------
    Residual sum of squares       : 3.41037823404e+14
    Training RMSE                 : 251097.7281
    
    Highest Positive Coefficients
    -----------------------------
    (intercept)                   : 462426.5657
    power_2                       : 1.0287
    power_4                       : 0.0
    power_7                       : 0.0
    power_8                       : 0.0
    
    Lowest Negative Coefficients
    ----------------------------
    power_1                       : -759.2518
    power_3                       : -0.0005
    power_5                       : -0.0
    power_6                       : -0.0
    power_10                      : -0.0




![png](output_25_2.png)



```python
poly_l2_regression(set_4,15,l2_small_penalty)
```

    +-------------+-------+----------------+-----+
    |     name    | index |     value      | ... |
    +-------------+-------+----------------+-----+
    | (intercept) |  None | -170240.032842 | ... |
    |   power_1   |  None | 1247.59034541  | ... |
    +-------------+-------+----------------+-----+
    [16 rows x 4 columns]
    





    Class                         : LinearRegression
    
    Schema
    ------
    Number of coefficients        : 16
    Number of examples            : 5402
    Number of feature columns     : 15
    Number of unpacked features   : 15
    
    Hyperparameters
    ---------------
    L1 penalty                    : 0.0
    L2 penalty                    : 0.0
    
    Training Summary
    ----------------
    Solver                        : newton
    Solver iterations             : 1
    Solver status                 : SUCCESS: Optimal solution found.
    Training time (sec)           : 0.0141
    
    Settings
    --------
    Residual sum of squares       : 3.22513810183e+14
    Training RMSE                 : 244341.2932
    
    Highest Positive Coefficients
    -----------------------------
    power_1                       : 1247.5903
    power_3                       : 0.0006
    power_6                       : 0.0
    power_7                       : 0.0
    power_10                      : 0.0
    
    Lowest Negative Coefficients
    ----------------------------
    (intercept)                   : -170240.0328
    power_2                       : -1.2246
    power_4                       : -0.0
    power_5                       : -0.0
    power_8                       : -0.0




![png](output_26_2.png)


The four curves should differ from one another a lot, as should the coefficients you learned.

***QUIZ QUESTION:  For the models learned in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?***  (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

-759         1247

# Ridge regression comes to rescue

Generally, whenever we see weights change so much in response to change in data, we believe the variance of our estimate to be large. Ridge regression aims to address this issue by penalizing "large" weights. (Weights of `model15` looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.)

With the argument `l2_penalty=1e5`, fit a 15th-order polynomial model on `set_1`, `set_2`, `set_3`, and `set_4`. Other than the change in the `l2_penalty` parameter, the code should be the same as the experiment above. Also, make sure GraphLab Create doesn't create its own validation set by using the option `validation_set = None` in this call.


```python
l2_penalty=1e5
for i in [set_1,set_2,set_3,set_4]:
    poly_l2_regression(i,15,l2_penalty)
```

    +-------------+-------+---------------+-----+
    |     name    | index |     value     | ... |
    +-------------+-------+---------------+-----+
    | (intercept) |  None | 530317.024516 | ... |
    |   power_1   |  None | 2.58738875673 | ... |
    +-------------+-------+---------------+-----+
    [16 rows x 4 columns]
    
    +-------------+-------+---------------+-----+
    |     name    | index |     value     | ... |
    +-------------+-------+---------------+-----+
    | (intercept) |  None | 519216.897383 | ... |
    |   power_1   |  None | 2.04470474182 | ... |
    +-------------+-------+---------------+-----+
    [16 rows x 4 columns]
    
    +-------------+-------+---------------+-----+
    |     name    | index |     value     | ... |
    +-------------+-------+---------------+-----+
    | (intercept) |  None | 522911.518048 | ... |
    |   power_1   |  None | 2.26890421877 | ... |
    +-------------+-------+---------------+-----+
    [16 rows x 4 columns]
    
    +-------------+-------+---------------+-----+
    |     name    | index |     value     | ... |
    +-------------+-------+---------------+-----+
    | (intercept) |  None | 513667.087087 | ... |
    |   power_1   |  None | 1.91040938244 | ... |
    +-------------+-------+---------------+-----+
    [16 rows x 4 columns]
    



![png](output_31_1.png)



```python

```


```python

```


```python

```

These curves should vary a lot less, now that you applied a high degree of regularization.

***QUIZ QUESTION:  For the models learned with the high level of regularization in each of these training sets, what are the smallest and largest values you learned for the coefficient of feature `power_1`?*** (For the purpose of answering this question, negative numbers are considered "smaller" than positive numbers. So -5 is smaller than -3, and -3 is smaller than 5 and so forth.)

# Selecting an L2 penalty via cross-validation

Just like the polynomial degree, the L2 penalty is a "magic" parameter we need to select. We could use the validation set approach as we did in the last module, but that approach has a major disadvantage: it leaves fewer observations available for training. **Cross-validation** seeks to overcome this issue by using all of the training set in a smart way.

We will implement a kind of cross-validation called **k-fold cross-validation**. The method gets its name because it involves dividing the training set into k segments of roughtly equal size. Similar to the validation set method, we measure the validation error with one of the segments designated as the validation set. The major difference is that we repeat the process k times as follows:

Set aside segment 0 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
Set aside segment 1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set<br>
...<br>
Set aside segment k-1 as the validation set, and fit a model on rest of data, and evalutate it on this validation set

After this process, we compute the average of the k validation errors, and use it as an estimate of the generalization error. Notice that  all observations are used for both training and validation, as we iterate over segments of data. 

To estimate the generalization error well, it is crucial to shuffle the training data before dividing them into segments. GraphLab Create has a utility function for shuffling a given SFrame. We reserve 10% of the data as the test set and shuffle the remainder. (Make sure to use `seed=1` to get consistent answer.)


```python
(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)
```

Once the data is shuffled, we divide it into equal segments. Each segment should receive `n/k` elements, where `n` is the number of observations in the training set and `k` is the number of segments. Since the segment 0 starts at index 0 and contains `n/k` elements, it ends at index `(n/k)-1`. The segment 1 starts where the segment 0 left off, at index `(n/k)`. With `n/k` elements, the segment 1 ends at index `(n*2/k)-1`. Continuing in this fashion, we deduce that the segment `i` starts at index `(n*i/k)` and ends at `(n*(i+1)/k)-1`.

With this pattern in mind, we write a short loop that prints the starting and ending indices of each segment, just to make sure you are getting the splits right.


```python
n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)
```

    0 (0, 1938)
    1 (1939, 3878)
    2 (3879, 5817)
    3 (5818, 7757)
    4 (7758, 9697)
    5 (9698, 11636)
    6 (11637, 13576)
    7 (13577, 15515)
    8 (15516, 17455)
    9 (17456, 19395)


Let us familiarize ourselves with array slicing with SFrame. To extract a continuous slice from an SFrame, use colon in square brackets. For instance, the following cell extracts rows 0 to 9 of `train_valid_shuffled`. Notice that the first index (0) is included in the slice but the last index (10) is omitted.


```python
train_valid_shuffled[0:10] # rows 0 to 9
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">id</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">date</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">price</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">bedrooms</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">bathrooms</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_living</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_lot</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">floors</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">waterfront</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2780400035</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-05-05 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">665000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2800.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5900</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1703050500</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-03-21 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">645000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2490.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5978</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5700002325</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-06-05 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">640000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2340.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4206</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0475000510</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-11-18 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">594000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1320.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5000</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0844001052</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-01-28 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">365000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1904.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8200</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2781280290</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-04-27 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">305000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1610.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3516</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2214800630</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-11-05 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">239950.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.25</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1560.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8280</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2114700540</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-10-21 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">366000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2.5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1320.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4320</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2596400050</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2014-07-30 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">375000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1960.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7955</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4140900050</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2015-01-26 00:00:00+00:00</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">440000.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1.75</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2180.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10200</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">view</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">condition</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">grade</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_above</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_basement</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">yr_built</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">yr_renovated</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">zipcode</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">lat</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1660</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1140</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1963</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98115</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.68093246</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">9</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2490</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2003</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98074</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.62984888</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1170</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1170</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1917</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98144</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.57587004</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1090</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">230</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1920</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98107</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.66737217</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1904</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1999</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98010</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.31068733</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1610</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2006</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98055</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.44911017</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1560</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1979</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98001</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.33933392</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">660</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">660</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1918</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98106</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.53271982</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">7</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1260</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">700</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1963</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98177</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.76407345</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2000</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">180</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1966</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">98028</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">47.76382378</td>
    </tr>
</table>
<table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">long</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_living15</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sqft_lot15</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.28583258</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2580.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5900.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.02177564</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2710.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">6629.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.28796</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1360.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4725.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.36472902</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1700.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5000.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.0012452</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1560.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">12426.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.1878086</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1610.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3056.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.25864364</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1920.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8120.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.34716948</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1190.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4200.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.36361517</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1850.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">8219.0</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-122.27022456</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">2590.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">10445.0</td>
    </tr>
</table>
[10 rows x 21 columns]<br/>
</div>



Now let us extract individual segments with array slicing. Consider the scenario where we group the houses in the `train_valid_shuffled` dataframe into k=10 segments of roughly equal size, with starting and ending indices computed as above.
Extract the fourth segment (segment 3) and assign it to a variable called `validation4`.


```python
validation4=train_valid_shuffled[n*3/k:n*4/k]
```

To verify that we have the right elements extracted, run the following cell, which computes the average price of the fourth segment. When rounded to nearest whole number, the average should be $536,234.


```python
print int(round(validation4['price'].mean(), 0))
```

    536234


After designating one of the k segments as the validation set, we train a model using the rest of the data. To choose the remainder, we slice (0:start) and (end+1:n) of the data and paste them together. SFrame has `append()` method that pastes together two disjoint sets of rows originating from a common dataset. For instance, the following cell pastes together the first and last two rows of the `train_valid_shuffled` dataframe.


```python
n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
print first_two.append(last_two)
```

    +------------+---------------------------+-----------+----------+-----------+
    |     id     |            date           |   price   | bedrooms | bathrooms |
    +------------+---------------------------+-----------+----------+-----------+
    | 2780400035 | 2014-05-05 00:00:00+00:00 |  665000.0 |   4.0    |    2.5    |
    | 1703050500 | 2015-03-21 00:00:00+00:00 |  645000.0 |   3.0    |    2.5    |
    | 4139480190 | 2014-09-16 00:00:00+00:00 | 1153000.0 |   3.0    |    3.25   |
    | 7237300290 | 2015-03-26 00:00:00+00:00 |  338000.0 |   5.0    |    2.5    |
    +------------+---------------------------+-----------+----------+-----------+
    +-------------+----------+--------+------------+------+-----------+-------+------------+
    | sqft_living | sqft_lot | floors | waterfront | view | condition | grade | sqft_above |
    +-------------+----------+--------+------------+------+-----------+-------+------------+
    |    2800.0   |   5900   |   1    |     0      |  0   |     3     |   8   |    1660    |
    |    2490.0   |   5978   |   2    |     0      |  0   |     3     |   9   |    2490    |
    |    3780.0   |  10623   |   1    |     0      |  1   |     3     |   11  |    2650    |
    |    2400.0   |   4496   |   2    |     0      |  0   |     3     |   7   |    2400    |
    +-------------+----------+--------+------------+------+-----------+-------+------------+
    +---------------+----------+--------------+---------+-------------+
    | sqft_basement | yr_built | yr_renovated | zipcode |     lat     |
    +---------------+----------+--------------+---------+-------------+
    |      1140     |   1963   |      0       |  98115  | 47.68093246 |
    |       0       |   2003   |      0       |  98074  | 47.62984888 |
    |      1130     |   1999   |      0       |  98006  | 47.55061236 |
    |       0       |   2004   |      0       |  98042  | 47.36923712 |
    +---------------+----------+--------------+---------+-------------+
    +---------------+---------------+-----+
    |      long     | sqft_living15 | ... |
    +---------------+---------------+-----+
    | -122.28583258 |     2580.0    | ... |
    | -122.02177564 |     2710.0    | ... |
    | -122.10144844 |     3850.0    | ... |
    | -122.12606473 |     1880.0    | ... |
    +---------------+---------------+-----+
    [4 rows x 21 columns]
    


Extract the remainder of the data after *excluding* fourth segment (segment 3) and assign the subset to `train4`.


```python
train4=train_valid_shuffled[0:n*3/k].append(train_valid_shuffled[n*4/k:n])
```

To verify that we have the right elements extracted, run the following cell, which computes the average price of the data with fourth segment excluded. When rounded to nearest whole number, the average should be $539,450.


```python
print int(round(train4['price'].mean(), 0))
```

    539450


Now we are ready to implement k-fold cross-validation. Write a function that computes k validation errors by designating each of the k segments as the validation set. It accepts as parameters (i) `k`, (ii) `l2_penalty`, (iii) dataframe, (iv) name of output column (e.g. `price`) and (v) list of feature names. The function returns the average validation error using k segments as validation sets.

* For each i in [0, 1, ..., k-1]:
  * Compute starting and ending indices of segment i and call 'start' and 'end'
  * Form validation set by taking a slice (start:end+1) from the data.
  * Form training set by appending slice (end+1:n) to the end of slice (0:start).
  * Train a linear model using training set just formed, with a given l2_penalty
  * Compute validation error using validation set just formed


```python
def pre_process(data,feature,deg,output_name):
    poly_data=polynomial_sframe(data[feature],deg)
    features=poly_data.column_names()
    poly_data[output_name]=data[output_name]
    return poly_data,features
```


```python
def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    errors=[]
    for i in xrange(k):
        start=n*i/k
        end=n*(i+1)/k-1
        validation_set=data[start:end+1]
        training_set=data[0:start].append(data[end+1:n])
        model=graphlab.linear_regression.create(training_set,l2_penalty=l2_penalty,features=features_list,target=output_name,validation_set=None,verbose=False)
        validation_error=sum((validation_set[output_name]-model.predict(validation_set))**2)/len(validation_set) 
        #注意此处不需要加+sum(model.get("coefficients")['value']**2)，题目要求是validation error
        #并且该式是在训练模型的时候用于优化带该式的cost fucntion以防止overfitting的，而现在是用validation进行评估
        #评估阶段不需要加上该式！
        errors.append(validation_error)                        
    validation_error=sum(errors)/k                              
    return validation_error,model
```

Once we have a function to compute the average validation error for a model, we can write a loop to find the model that minimizes the average validation error. Write a loop that does the following:
* We will again be aiming to fit a 15th-order polynomial model using the `sqft_living` input
* For `l2_penalty` in [10^1, 10^1.5, 10^2, 10^2.5, ..., 10^7] (to get this in Python, you can use this Numpy function: `np.logspace(1, 7, num=13)`.)
    * Run 10-fold cross-validation with `l2_penalty`
* Report which L2 penalty produced the lowest average validation error.


Note: since the degree of the polynomial is now fixed to 15, to make things faster, you should generate polynomial features in advance and re-use them throughout the loop. Make sure to use `train_valid_shuffled` when generating polynomial features!


```python
import numpy as np
poly_data_shuffled,features=pre_process(train_valid_shuffled,'sqft_living',15,'price')
min_valid_error,best_model=k_fold_cross_validation(10,10,poly_data_shuffled,'price',features)
errors=[min_valid_error]
print type(errors)
best_l2_penalty=10
for l2_penalty in np.logspace(1,7,num=13)[1:13]:
    error,model=k_fold_cross_validation(10,l2_penalty,poly_data_shuffled,'price',features)
    errors.append(error)
    if error<=min_valid_error:
        min_valid_error=error
        best_model=model
        best_l2_penalty=l2_penalty

print best_l2_penalty
print min_valid_error
print errors
```

    <type 'list'>
    1000.0
    62483000679.2
    [253629627559.3939, 148255039063.8181, 82966236507.76212, 62946613842.037704, 62483000679.23887, 63904726964.60065, 70548961052.91782, 88537917379.45909, 118254131885.54941, 130409719652.25449, 133368686716.71367, 135501271577.79068, 136568319836.40756]


***QUIZ QUESTIONS:  What is the best value for the L2 penalty according to 10-fold validation?***

1000

You may find it useful to plot the k-fold cross-validation errors you have obtained to better understand the behavior of the method.  


```python
# Plot the l2_penalty values in the x axis and the cross-validation error in the y axis.
# Using plt.xscale('log') will make your plot more intuitive.
plt.xscale('log')
plt.plot(np.logspace(1,7,num=13),errors,'.')
```




    [<matplotlib.lines.Line2D at 0x7f5600d82dd0>]




![png](output_61_1.png)


Once you found the best value for the L2 penalty using cross-validation, it is important to retrain a final model on all of the training data using this value of `l2_penalty`.  This way, your final model will be trained on the entire dataset.


```python
model=graphlab.linear_regression.create(poly_data_shuffled,l2_penalty=best_l2_penalty,features=features,target='price',validation_set=None,verbose=False)
```

***QUIZ QUESTION: Using the best L2 penalty found above, train a model using all training data. What is the RSS on the TEST data of the model you learn with this L2 penalty? ***


```python
test_processed,features=pre_process(test,'sqft_living',15,'price')
sum( (test['price']-model.predict(test_processed))**2 )
```




    128780855058449.36


