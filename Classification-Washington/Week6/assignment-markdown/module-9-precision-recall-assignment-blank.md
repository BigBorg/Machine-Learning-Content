
# Exploring precision and recall

The goal of this second notebook is to understand precision-recall in the context of classifiers.

 * Use Amazon review data in its entirety.
 * Train a logistic regression model.
 * Explore various evaluation metrics: accuracy, confusion matrix, precision, recall.
 * Explore how various metrics can be combined to produce a cost of making an error.
 * Explore precision and recall curves.
 
Because we are using the full Amazon review dataset (not a subset of words or reviews), in this assignment we return to using GraphLab Create for its efficiency. As usual, let's start by **firing up GraphLab Create**.

Make sure you have the latest version of GraphLab Create (1.8.3 or later). If you don't find the decision tree module, then you would need to upgrade graphlab-create using

```
   pip install graphlab-create --upgrade
```
See [this page](https://dato.com/download/) for detailed instructions on upgrading.


```python
import graphlab
from __future__ import division
import numpy as np
graphlab.canvas.set_target('ipynb')
```

# Load amazon review dataset


```python
products = graphlab.SFrame('amazon_baby.gl/')
```

    This non-commercial license of GraphLab Create for academic use is assigned to 770188954@qq.com and will expire on January 14, 2017.


    [INFO] graphlab.cython.cy_server: GraphLab Create v1.10.1 started. Logging: /tmp/graphlab_server_1469297884.log


# Extract word counts and sentiments

As in the first assignment of this course, we compute the word counts for individual words and extract positive and negative sentiments from ratings. To summarize, we perform the following:

1. Remove punctuation.
2. Remove reviews with "neutral" sentiment (rating 3).
3. Set reviews with rating 4 or more to be positive and those with 2 or less to be negative.


```python
def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

# Remove punctuation.
review_clean = products['review'].apply(remove_punctuation)

# Count words
products['word_count'] = graphlab.text_analytics.count_words(review_clean)

# Drop neutral sentiment reviews.
products = products[products['rating'] != 3]

# Positive sentiment to +1 and negative sentiment to -1
products['sentiment'] = products['rating'].apply(lambda rating : +1 if rating > 3 else -1)
```

Now, let's remember what the dataset looks like by taking a quick peek:


```python
products
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">name</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">review</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">rating</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">word_count</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">sentiment</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Planetwise Wipe Pouch</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">it came early and was not<br>disappointed. i love ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 3, 'love': 1,<br>'it': 3, 'highly': 1, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Annas Dream Full Quilt<br>with 2 Shams ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Very soft and comfortable<br>and warmer than it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2, 'quilt': 1,<br>'it': 1, 'comfortable': ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This is a product well<br>worth the purchase.  I ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 3, 'ingenious':<br>1, 'love': 2, 'is': 4, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">All of my kids have cried<br>non-stop when I tried to ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2, 'all': 2,<br>'help': 1, 'cried': 1, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Stop Pacifier Sucking<br>without tears with ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">When the Binky Fairy came<br>to our house, we didn't ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2, 'cute': 1,<br>'help': 2, 'habit': 1, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A Tale of Baby's Days<br>with Peter Rabbit ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Lovely book, it's bound<br>tightly so you may no ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'shop': 1, 'be': 1,<br>'is': 1, 'bound': 1, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Perfect for new parents.<br>We were able to keep ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 2, 'all': 1,<br>'right': 1, 'able': 1, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">A friend of mine pinned<br>this product on Pinte ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'and': 1, 'fantastic':<br>1, 'help': 1, 'give': 1, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">This has been an easy way<br>for my nanny to record ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 1, 'standarad':<br>1, 'another': 1, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">Baby Tracker&amp;reg; - Daily<br>Childcare Journal, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">I love this journal and<br>our nanny uses it ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4.0</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">{'all': 2, 'nannys': 1,<br>'just': 1, 'sleep': 2, ...</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
    </tr>
</table>
[166752 rows x 5 columns]<br/>Note: Only the head of the SFrame is printed.<br/>You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.
</div>



## Split data into training and test sets

We split the data into a 80-20 split where 80% is in the training set and 20% is in the test set.


```python
train_data, test_data = products.random_split(.8, seed=1)
```

## Train a logistic regression classifier

We will now train a logistic regression classifier with **sentiment** as the target and **word_count** as the features. We will set `validation_set=None` to make sure everyone gets exactly the same results.  

Remember, even though we now know how to implement logistic regression, we will use GraphLab Create for its efficiency at processing this Amazon dataset in its entirety.  The focus of this assignment is instead on the topic of precision and recall.


```python
model = graphlab.logistic_classifier.create(train_data, target='sentiment',
                                            features=['word_count'],
                                            validation_set=None)
```


<pre>Logistic regression:</pre>



<pre>--------------------------------------------------------</pre>



<pre>Number of examples          : 133416</pre>



<pre>Number of classes           : 2</pre>



<pre>Number of feature columns   : 1</pre>



<pre>Number of unpacked features : 121712</pre>



<pre>Number of coefficients    : 121713</pre>



<pre>Starting L-BFGS</pre>



<pre>--------------------------------------------------------</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+</pre>



<pre>| Iteration | Passes   | Step size | Elapsed Time | Training-accuracy |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+</pre>



<pre>| 1         | 5        | 0.000002  | 2.341592     | 0.840754          |</pre>



<pre>| 2         | 9        | 3.000000  | 3.576867     | 0.931350          |</pre>



<pre>| 3         | 10       | 3.000000  | 4.030960     | 0.882046          |</pre>



<pre>| 4         | 11       | 3.000000  | 4.524486     | 0.954076          |</pre>



<pre>| 5         | 12       | 3.000000  | 5.003114     | 0.960964          |</pre>



<pre>| 6         | 13       | 3.000000  | 5.456605     | 0.975033          |</pre>



<pre>+-----------+----------+-----------+--------------+-------------------+</pre>



<pre>TERMINATED: Terminated due to numerical difficulties.</pre>



<pre>This model may not be ideal. To improve it, consider doing one of the following:
(a) Increasing the regularization.
(b) Standardizing the input data.
(c) Removing highly correlated features.
(d) Removing `inf` and `NaN` values in the training data.</pre>


# Model Evaluation

We will explore the advanced model evaluation concepts that were discussed in the lectures.

## Accuracy

One performance metric we will use for our more advanced exploration is accuracy, which we have seen many times in past assignments.  Recall that the accuracy is given by

$$
\mbox{accuracy} = \frac{\mbox{# correctly classified data points}}{\mbox{# total data points}}
$$

To obtain the accuracy of our trained models using GraphLab Create, simply pass the option `metric='accuracy'` to the `evaluate` function. We compute the **accuracy** of our logistic regression model on the **test_data** as follows:


```python
accuracy= model.evaluate(test_data, metric='accuracy')['accuracy']
print "Test Accuracy: %s" % accuracy
```

    Test Accuracy: 0.914536837053


## Baseline: Majority class prediction

Recall from an earlier assignment that we used the **majority class classifier** as a baseline (i.e reference) model for a point of comparison with a more sophisticated classifier. The majority classifier model predicts the majority class for all data points. 

Typically, a good model should beat the majority class classifier. Since the majority class in this dataset is the positive class (i.e., there are more positive than negative reviews), the accuracy of the majority class classifier can be computed as follows:


```python
baseline = len(test_data[test_data['sentiment'] == 1])/len(test_data)
print "Baseline accuracy (majority class classifier): %s" % baseline
```

    Baseline accuracy (majority class classifier): 0.842782577394


** Quiz Question:** Using accuracy as the evaluation metric, was our **logistic regression model** better than the baseline (majority class classifier)?  
Yes

## Confusion Matrix

The accuracy, while convenient, does not tell the whole story. For a fuller picture, we turn to the **confusion matrix**. In the case of binary classification, the confusion matrix is a 2-by-2 matrix laying out correct and incorrect predictions made in each label as follows:
```
              +---------------------------------------------+
              |                Predicted label              |
              +----------------------+----------------------+
              |          (+1)        |         (-1)         |
+-------+-----+----------------------+----------------------+
| True  |(+1) | # of true positives  | # of false negatives |
| label +-----+----------------------+----------------------+
|       |(-1) | # of false positives | # of true negatives  |
+-------+-----+----------------------+----------------------+
```
To print out the confusion matrix for a classifier, use `metric='confusion_matrix'`:


```python
confusion_matrix = model.evaluate(test_data, metric='confusion_matrix')['confusion_matrix']
confusion_matrix
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">target_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">predicted_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">3798</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1443</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1406</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">26689</td>
    </tr>
</table>
[4 rows x 3 columns]<br/>
</div>



**Quiz Question**: How many predicted values in the **test set** are **false positives**?


```python
1443
```




    1443



## Computing the cost of mistakes


Put yourself in the shoes of a manufacturer that sells a baby product on Amazon.com and you want to monitor your product's reviews in order to respond to complaints.  Even a few negative reviews may generate a lot of bad publicity about the product. So you don't want to miss any reviews with negative sentiments --- you'd rather put up with false alarms about potentially negative reviews instead of missing negative reviews entirely. In other words, **false positives cost more than false negatives**. (It may be the other way around for other scenarios, but let's stick with the manufacturer's scenario for now.)

Suppose you know the costs involved in each kind of mistake: 
1. \$100 for each false positive.
2. \$1 for each false negative.
3. Correctly classified reviews incur no cost.

**Quiz Question**: Given the stipulation, what is the cost associated with the logistic regression classifier's performance on the **test set**?


```python
1443*100+1406
```




    145706



## Precision and Recall

You may not have exact dollar amounts for each kind of mistake. Instead, you may simply prefer to reduce the percentage of false positives to be less than, say, 3.5% of all positive predictions. This is where **precision** comes in:

$$
[\text{precision}] = \frac{[\text{# positive data points with positive predicitions}]}{\text{[# all data points with positive predictions]}} = \frac{[\text{# true positives}]}{[\text{# true positives}] + [\text{# false positives}]}
$$

So to keep the percentage of false positives below 3.5% of positive predictions, we must raise the precision to 96.5% or higher. 

**First**, let us compute the precision of the logistic regression classifier on the **test_data**.


```python
precision = model.evaluate(test_data, metric='precision')['precision']
print "Precision on test data: %s" % precision
```

    Precision on test data: 0.948706099815


**Quiz Question**: Out of all reviews in the **test set** that are predicted to be positive, what fraction of them are **false positives**? (Round to the second decimal place e.g. 0.25)


```python
1-precision
```




    0.05129390018484292



**Quiz Question:** Based on what we learned in lecture, if we wanted to reduce this fraction of false positives to be below 3.5%, we would: (see the quiz)

A complementary metric is **recall**, which measures the ratio between the number of true positives and that of (ground-truth) positive reviews:

$$
[\text{recall}] = \frac{[\text{# positive data points with positive predicitions}]}{\text{[# all positive data points]}} = \frac{[\text{# true positives}]}{[\text{# true positives}] + [\text{# false negatives}]}
$$

Let us compute the recall on the **test_data**.


```python
recall = model.evaluate(test_data, metric='recall')['recall']
print "Recall on test data: %s" % recall
```

    Recall on test data: 0.949955508098


**Quiz Question**: What fraction of the positive reviews in the **test_set** were correctly predicted as positive by the classifier?

**Quiz Question**: What is the recall value for a classifier that predicts **+1** for all data points in the **test_data**?

# Precision-recall tradeoff

In this part, we will explore the trade-off between precision and recall discussed in the lecture.  We first examine what happens when we use a different threshold value for making class predictions.  We then explore a range of threshold values and plot the associated precision-recall curve.  


## Varying the threshold

False positives are costly in our example, so we may want to be more conservative about making positive predictions. To achieve this, instead of thresholding class probabilities at 0.5, we can choose a higher threshold. 

Write a function called `apply_threshold` that accepts two things
* `probabilities` (an SArray of probability values)
* `threshold` (a float between 0 and 1).

The function should return an SArray, where each element is set to +1 or -1 depending whether the corresponding probability exceeds `threshold`.


```python
def apply_threshold(probabilities, threshold):
    ### YOUR CODE GOES HERE
    # +1 if >= threshold and -1 otherwise.
    return probabilities.apply(lambda x:+1 if x>=threshold else -1)
```

Run prediction with `output_type='probability'` to get the list of probability values. Then use thresholds set at 0.5 (default) and 0.9 to make predictions from these probability values.


```python
probabilities = model.predict(test_data, output_type='probability')
predictions_with_default_threshold = apply_threshold(probabilities, 0.5)
predictions_with_high_threshold = apply_threshold(probabilities, 0.9)
```


```python
print "Number of positive predicted reviews (threshold = 0.5): %s" % (predictions_with_default_threshold == 1).sum()
```

    Number of positive predicted reviews (threshold = 0.5): 28132



```python
print "Number of positive predicted reviews (threshold = 0.9): %s" % (predictions_with_high_threshold == 1).sum()
```

    Number of positive predicted reviews (threshold = 0.9): 25630


**Quiz Question**: What happens to the number of positive predicted reviews as the threshold increased from 0.5 to 0.9?

## Exploring the associated precision and recall as the threshold varies

By changing the probability threshold, it is possible to influence precision and recall. We can explore this as follows:


```python
# Threshold = 0.5
precision_with_default_threshold = graphlab.evaluation.precision(test_data['sentiment'],
                                        predictions_with_default_threshold)

recall_with_default_threshold = graphlab.evaluation.recall(test_data['sentiment'],
                                        predictions_with_default_threshold)

# Threshold = 0.9
precision_with_high_threshold = graphlab.evaluation.precision(test_data['sentiment'],
                                        predictions_with_high_threshold)
recall_with_high_threshold = graphlab.evaluation.recall(test_data['sentiment'],
                                        predictions_with_high_threshold)
```


```python
print "Precision (threshold = 0.5): %s" % precision_with_default_threshold
print "Recall (threshold = 0.5)   : %s" % recall_with_default_threshold
```

    Precision (threshold = 0.5): 0.948706099815
    Recall (threshold = 0.5)   : 0.949955508098



```python
print "Precision (threshold = 0.9): %s" % precision_with_high_threshold
print "Recall (threshold = 0.9)   : %s" % recall_with_high_threshold
```

    Precision (threshold = 0.9): 0.969527896996
    Recall (threshold = 0.9)   : 0.884463427656


**Quiz Question (variant 1)**: Does the **precision** increase with a higher threshold?  
Yes

**Quiz Question (variant 2)**: Does the **recall** increase with a higher threshold?  
No

## Precision-recall curve

Now, we will explore various different values of tresholds, compute the precision and recall scores, and then plot the precision-recall curve.


```python
threshold_values = np.linspace(0.5, 1, num=100)
print threshold_values
```

    [ 0.5         0.50505051  0.51010101  0.51515152  0.52020202  0.52525253
      0.53030303  0.53535354  0.54040404  0.54545455  0.55050505  0.55555556
      0.56060606  0.56565657  0.57070707  0.57575758  0.58080808  0.58585859
      0.59090909  0.5959596   0.6010101   0.60606061  0.61111111  0.61616162
      0.62121212  0.62626263  0.63131313  0.63636364  0.64141414  0.64646465
      0.65151515  0.65656566  0.66161616  0.66666667  0.67171717  0.67676768
      0.68181818  0.68686869  0.69191919  0.6969697   0.7020202   0.70707071
      0.71212121  0.71717172  0.72222222  0.72727273  0.73232323  0.73737374
      0.74242424  0.74747475  0.75252525  0.75757576  0.76262626  0.76767677
      0.77272727  0.77777778  0.78282828  0.78787879  0.79292929  0.7979798
      0.8030303   0.80808081  0.81313131  0.81818182  0.82323232  0.82828283
      0.83333333  0.83838384  0.84343434  0.84848485  0.85353535  0.85858586
      0.86363636  0.86868687  0.87373737  0.87878788  0.88383838  0.88888889
      0.89393939  0.8989899   0.9040404   0.90909091  0.91414141  0.91919192
      0.92424242  0.92929293  0.93434343  0.93939394  0.94444444  0.94949495
      0.95454545  0.95959596  0.96464646  0.96969697  0.97474747  0.97979798
      0.98484848  0.98989899  0.99494949  1.        ]


For each of the values of threshold, we compute the precision and recall scores.


```python
precision_all = []
recall_all = []

probabilities = model.predict(test_data, output_type='probability')
for threshold in threshold_values:
    predictions = apply_threshold(probabilities, threshold)
    
    precision = graphlab.evaluation.precision(test_data['sentiment'], predictions)
    recall = graphlab.evaluation.recall(test_data['sentiment'], predictions)
    print threshold,precision
    precision_all.append(precision)
    recall_all.append(recall)
```

    0.5 0.948706099815
    0.505050505051 0.94905908719
    0.510101010101 0.949288256228
    0.515151515152 0.949506819072
    0.520202020202 0.949624140511
    0.525252525253 0.949805711026
    0.530303030303 0.950203324534
    0.535353535354 0.950417648319
    0.540404040404 0.950696677385
    0.545454545455 0.950877694755
    0.550505050505 0.951062459755
    0.555555555556 0.951424684994
    0.560606060606 0.951534907046
    0.565656565657 0.951761459341
    0.570707070707 0.952177656598
    0.575757575758 0.952541642734
    0.580808080808 0.952825782345
    0.585858585859 0.952950902164
    0.590909090909 0.953033408854
    0.59595959596 0.953081711222
    0.60101010101 0.953231323132
    0.606060606061 0.953525236877
    0.611111111111 0.953680340278
    0.616161616162 0.953691347784
    0.621212121212 0.954012200845
    0.626262626263 0.95415959253
    0.631313131313 0.954481362305
    0.636363636364 0.954630969609
    0.641414141414 0.954956912159
    0.646464646465 0.955217391304
    0.651515151515 0.955425794284
    0.656565656566 0.955603150978
    0.661616161616 0.955716205907
    0.666666666667 0.955933682373
    0.671717171717 0.95600756859
    0.676767676768 0.956162388494
    0.681818181818 0.956453611253
    0.686868686869 0.956670800204
    0.691919191919 0.956951949759
    0.69696969697 0.957200292398
    0.70202020202 0.95730904302
    0.707070707071 0.957558224696
    0.712121212121 0.957740800469
    0.717171717172 0.958172812328
    0.722222222222 0.958434310054
    0.727272727273 0.958762128786
    0.732323232323 0.959152130713
    0.737373737374 0.959266352387
    0.742424242424 0.95958553044
    0.747474747475 0.959906966441
    0.752525252525 0.959957149717
    0.757575757576 0.960170118343
    0.762626262626 0.96034655115
    0.767676767677 0.960716006374
    0.772727272727 0.960870855278
    0.777777777778 0.961087182534
    0.782828282828 0.961366847624
    0.787878787879 0.962202659674
    0.792929292929 0.962415603901
    0.79797979798 0.9624873268
    0.80303030303 0.962727546261
    0.808080808081 0.963204278397
    0.813131313131 0.963492362814
    0.818181818182 0.963922783423
    0.823232323232 0.964218170815
    0.828282828283 0.964581991742
    0.833333333333 0.964945559391
    0.838383838384 0.965311550152
    0.843434343434 0.965662948723
    0.848484848485 0.965982762566
    0.853535353535 0.966381418093
    0.858585858586 0.966780205901
    0.863636363636 0.966996320147
    0.868686868687 0.96737626806
    0.873737373737 0.96765996766
    0.878787878788 0.967978395062
    0.883838383838 0.968586792526
    0.888888888889 0.968960968418
    0.893939393939 0.969313939017
    0.89898989899 0.969468923029
    0.90404040404 0.969731336279
    0.909090909091 0.969926286073
    0.914141414141 0.970296640176
    0.919191919192 0.970813586098
    0.924242424242 0.971404775125
    0.929292929293 0.97203187251
    0.934343434343 0.972883121045
    0.939393939394 0.973425672411
    0.944444444444 0.974041226258
    0.949494949495 0.974463571837
    0.954545454545 0.974766393611
    0.959595959596 0.97549325026
    0.964646464646 0.976197472818
    0.969696969697 0.976871731644
    0.974747474747 0.977337354589
    0.979797979798 0.978530031612
    0.984848484848 0.980131852253
    0.989898989899 0.981307971185
    0.994949494949 0.984238628196
    1.0 0.991666666667


Now, let's plot the precision-recall curve to visualize the precision-recall tradeoff as we vary the threshold.


```python
import matplotlib.pyplot as plt
%matplotlib inline

def plot_pr_curve(precision, recall, title):
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(precision, recall, 'b-', linewidth=4.0, color = '#B0017F')
    plt.title(title)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.rcParams.update({'font.size': 16})
    
plot_pr_curve(precision_all, recall_all, 'Precision recall curve (all)')
```


![png](output_54_0.png)


**Quiz Question**: Among all the threshold values tried, what is the **smallest** threshold value that achieves a precision of 96.5% or better? Round your answer to 3 decimal places.


```python
0.838
```

**Quiz Question**: Using `threshold` = 0.98, how many **false negatives** do we get on the **test_data**? (**Hint**: You may use the `graphlab.evaluation.confusion_matrix` function implemented in GraphLab Create.)


```python
predictions = apply_threshold(probabilities, 0.98)
graphlab.evaluation.confusion_matrix(test_data['sentiment'],predictions)
```




<div style="max-height:1000px;max-width:1500px;overflow:auto;"><table frame="box" rules="cols">
    <tr>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">target_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">predicted_label</th>
        <th style="padding-left: 1em; padding-right: 1em; text-align: center">count</th>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">5826</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">22269</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">4754</td>
    </tr>
    <tr>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">-1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">1</td>
        <td style="padding-left: 1em; padding-right: 1em; text-align: center; vertical-align: top">487</td>
    </tr>
</table>
[4 rows x 3 columns]<br/>
</div>




```python
5826
```




    5826



This is the number of false negatives (i.e the number of reviews to look at when not needed) that we have to deal with using this classifier.

# Evaluating specific search terms

So far, we looked at the number of false positives for the **entire test set**. In this section, let's select reviews using a specific search term and optimize the precision on these reviews only. After all, a manufacturer would be interested in tuning the false positive rate just for their products (the reviews they want to read) rather than that of the entire set of products on Amazon.

## Precision-Recall on all baby related items

From the **test set**, select all the reviews for all products with the word 'baby' in them.


```python
baby_reviews =  test_data[test_data['name'].apply(lambda x: 'baby' in x.lower())]
```

Now, let's predict the probability of classifying these reviews as positive:


```python
probabilities = model.predict(baby_reviews, output_type='probability')
```

Let's plot the precision-recall curve for the **baby_reviews** dataset.

**First**, let's consider the following `threshold_values` ranging from 0.5 to 1:


```python
threshold_values = np.linspace(0.5, 1, num=100)
```

**Second**, as we did above, let's compute precision and recall for each value in `threshold_values` on the **baby_reviews** dataset.  Complete the code block below.


```python
precision_all = []
recall_all = []

for threshold in threshold_values:
    
    # Make predictions. Use the `apply_threshold` function 
    ## YOUR CODE HERE 
    predictions = apply_threshold(probabilities,threshold)

    # Calculate the precision.
    # YOUR CODE HERE
    precision = sum((baby_reviews['sentiment']==1)&(predictions==1))/sum(predictions==1)
    
    # YOUR CODE HERE
    recall = sum((baby_reviews['sentiment']==1)&(predictions==1))/sum(baby_reviews['sentiment']==1)
    
    print threshold,precision,recall
    # Append the precision and recall scores.
    precision_all.append(precision)
    recall_all.append(recall)
```

    0.5 0.947656392486 0.944555535357
    0.505050505051 0.948165723672 0.944373750227
    0.510101010101 0.948319941563 0.944010179967
    0.515151515152 0.948474328522 0.943646609707
    0.520202020202 0.948638274538 0.943464824577
    0.525252525253 0.948792977323 0.943101254317
    0.530303030303 0.949487554905 0.943101254317
    0.535353535354 0.949459805896 0.942555898927
    0.540404040404 0.94998167827 0.942555898927
    0.545454545455 0.949954170486 0.942010543538
    0.550505050505 0.95011920044 0.941828758408
    0.555555555556 0.950816663608 0.941828758408
    0.560606060606 0.95080763583 0.941646973278
    0.565656565657 0.950964187328 0.941283403018
    0.570707070707 0.951793928243 0.940374477368
    0.575757575758 0.951951399116 0.940010907108
    0.580808080808 0.952082565426 0.939101981458
    0.585858585859 0.952407304925 0.938556626068
    0.590909090909 0.952363367799 0.937647700418
    0.59595959596 0.952345770225 0.937284130158
    0.60101010101 0.952336966562 0.937102345028
    0.606060606061 0.952856350527 0.936920559898
    0.611111111111 0.95282146161 0.936193419378
    0.616161616162 0.952795261014 0.935648063988
    0.621212121212 0.952901909883 0.934193782949
    0.626262626263 0.953035084463 0.933284857299
    0.631313131313 0.953212031192 0.933284857299
    0.636363636364 0.953354395094 0.932557716779
    0.641414141414 0.953683035714 0.932012361389
    0.646464646465 0.954020848846 0.931648791129
    0.651515151515 0.954172876304 0.931103435739
    0.656565656566 0.954164337619 0.930921650609
    0.661616161616 0.954291044776 0.929830939829
    0.666666666667 0.954248366013 0.928922014179
    0.671717171717 0.954214165577 0.928194873659
    0.676767676768 0.95436693473 0.927649518269
    0.681818181818 0.954715568862 0.927467733139
    0.686868686869 0.954852004496 0.92655880749
    0.691919191919 0.954997187324 0.92583166697
    0.69696969697 0.955355468017 0.92583166697
    0.70202020202 0.955492957746 0.92492274132
    0.707070707071 0.955622414442 0.92383203054
    0.712121212121 0.955768868812 0.92310489002
    0.717171717172 0.95627591406 0.9223777495
    0.722222222222 0.956259426848 0.92201417924
    0.727272727273 0.956579195771 0.92110525359
    0.732323232323 0.956924239562 0.92074168333
    0.737373737374 0.956883509834 0.91983275768
    0.742424242424 0.957188861527 0.918560261771
    0.747474747475 0.957321699545 0.917469550991
    0.752525252525 0.957471046136 0.916742410471
    0.757575757576 0.957596501236 0.915469914561
    0.762626262626 0.957912778518 0.914379203781
    0.767676767677 0.958245948522 0.913652063261
    0.772727272727 0.958778625954 0.913288493001
    0.777777777778 0.959105675521 0.912379567351
    0.782828282828 0.959586286152 0.910743501182
    0.787878787879 0.960655737705 0.905471732412
    0.792929292929 0.96098126328 0.904381021632
    0.79797979798 0.961121856867 0.903290310853
    0.80303030303 0.961084220716 0.902381385203
    0.808080808081 0.961419154711 0.901472459553
    0.813131313131 0.961545931249 0.900018178513
    0.818181818182 0.962062256809 0.898927467733
    0.823232323232 0.962378167641 0.897473186693
    0.828282828283 0.962724434036 0.896746046173
    0.833333333333 0.963064295486 0.895837120524
    0.838383838384 0.963194988254 0.894382839484
    0.843434343434 0.963711259317 0.893110343574
    0.848484848485 0.964194373402 0.890928922014
    0.853535353535 0.964722112732 0.889838211234
    0.858585858586 0.964863797868 0.888565715324
    0.863636363636 0.965019762846 0.887656789675
    0.868686868687 0.965510406343 0.885475368115
    0.873737373737 0.966037735849 0.884202872205
    0.878787878788 0.966155683854 0.882203235775
    0.883838383838 0.966839792249 0.879840029086
    0.888888888889 0.967935871743 0.878022177786
    0.893939393939 0.968072289157 0.876386111616
    0.89898989899 0.968014484007 0.874750045446
    0.90404040404 0.967911200807 0.871841483367
    0.909090909091 0.967859308672 0.870387202327
    0.914141414141 0.968154158215 0.867660425377
    0.919191919192 0.968470301058 0.865479003817
    0.924242424242 0.969139587165 0.862025086348
    0.929292929293 0.969771745836 0.857298672969
    0.934343434343 0.971050454921 0.853662970369
    0.939393939394 0.971226021685 0.84675513543
    0.944444444444 0.971476510067 0.842028722051
    0.949494949495 0.972245762712 0.834211961462
    0.954545454545 0.972418216806 0.826758771133
    0.959595959596 0.973411154345 0.818578440284
    0.964646464646 0.974202011369 0.810034539175
    0.969696969697 0.97500552975 0.801308852936
    0.974747474747 0.975100942127 0.790219960007
    0.979797979798 0.976659038902 0.775858934739
    0.984848484848 0.979048964218 0.756044355572
    0.989898989899 0.980103168755 0.725322668606
    0.994949494949 0.984425349087 0.666424286493
    1.0 1.0 0.00290856207962


**Quiz Question**: Among all the threshold values tried, what is the **smallest** threshold value that achieves a precision of 96.5% or better for the reviews of data in **baby_reviews**? Round your answer to 3 decimal places.


```python
0.864
```




    0.864



**Quiz Question:** Is this threshold value smaller or larger than the threshold used for the entire dataset to achieve the same specified precision of 96.5%?

**Finally**, let's plot the precision recall curve.


```python
plot_pr_curve(precision_all, recall_all, "Precision-Recall (Baby)")
```


![png](output_73_0.png)



```python

```
