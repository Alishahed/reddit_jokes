
![Python](https://img.shields.io/badge/python-3.8-blue.svg)
<div style="text-align: center;">
    <img src="images/funny_measure_webpage.png" width="600" />
</div>

# Can a machine judge the quality of a joke?
This the question that I am trying to answer and for the start, I need to to find a way to quantify a good joke. In this repository I will try to analyze the r/jokes subreddit dataset that was uploaded to The dataset that I used is extracted and simplified from [Kaggle](https://www.kaggle.com/datasets/bwandowando/reddit-rjokes-dataset) dataset. This dataset contains ~37200 joke threads.
I chose subset of columns that I was interested in my analysis.
Here are the list of columns that I kept:
- *thread_id*: unique id of the thread containing the joke (Object)
- *thread_title*: title of the thread (Object) *sometimes this title contains start of the joke*
- *thread_selftext*: Text of the thread which includes the joke (Object)
- *thread_score*: This score between 0 and 1 (Object) *supposed to be Upvotes - downvotes but I can't see any negative values so still I don't know what is it.*
- *thread_num_comments*: Number of comments in the thread (float64)
- *thread_created_utc*: Time of the thread creation in UTC (Object)
- *thread_upvote_ratio*: Ratio of upvotes to to total votes (float64)
- *thread_over_18*: Whether the thread is over 18 or not (Object)
 and try to find a way to quantify a good joke. I will also try to build a model that can generate jokes.

 ## Data 
Three files are in the *data* folder
1) **reddit_jokes_slim.csv**: All ~37200 jokes
2) **reddit_jokes_slim_processed.csv**: All jokes but columns are converted to appropriate types
3) **reddit_jokes_slim_clean.csv** : Only clean jokes
4) **reddit_jokes_slim_plus18.csv**: Only adult jokes

## Experiments Notebooks
1) **reddit_jokes_score_analysis.ipynb** : 

Which measure in this dataset can be reflect the quality of the joke? thread_score which is assigned by reddit or thread_upvote_ratio which is the ratio of upvotes to total votes?

2) **reddit_jokes_adult_logreg_classifier.ipynb**:

Not directly related to the quality of the joke but my curiosity. I trained a simple logistic regression model to classify adult jokes. I used thread_title, thread_selftext as features and thread_over_18 as target. Good performance considering how simple the model is. Of course, more objective test of the model, I need to find a much larger dataset.

3) **reddit_jokes_adult_lstm_classifier.ipynb**:

Another attempt to detect adult jokes. This time I used LSTM model which resulted in better performance compare to logistic regression model.

4) **reddit_jokes_upvote_prediction_linreg.ipynb**:
In this notebook we try to predict the thread_upvote_ratio using thread_title and thread_selftext. We used linear regression model and we got -0.06 R2 score which is not is not adequate. Also, We explore the performance of this model using Q-Q plot of the residues as well as residue plot. All the plots indicates the poor performance of the model.

5) **reddit_jokes_upvote_prediction_lstm.ipynb**:
Continuing the prediction of upvote ratio, this time I used LSTM model. The model overfits despite all the mechanism that I have put in place, including batch normalization, regularization and dropout. The small size of dataset is to blame here I guess. However, the R-squared on test data can reach upto 16% on the test data which is much better than linear regression model. 

6) **reddit_jokes_upvote_ratio_prediction_classifier.ipynb**:
In this notebook, I continue with the problem of predicting the upvote ratio but to make it an easier problem to solve, I turn the problem to a binary classification problem by putting the jokes into two equal sized bucket (low upvote ratio and  high upvote ratio). Then I tried to train a logistic regression to predict the class using only the text of the joke. Here is the best result thst I manage to get.
                precision    recall  f1-score   support

           0       0.58      0.61      0.60      5626
           1       0.59      0.55      0.57      5539

    accuracy                           0.58     11165
   macro avg       0.58      0.58      0.58     11165
weighted avg       0.58      0.58      0.58     11165

Not very promising.

6) **reddit_jokes_upvote_ratio_prediction_classifier_2.ipynb**:

In this notebook I added lightGBM model and LSTM to test how increasing the complexity of the model affect the performance.
                
**logistic regression**                
                precision    recall  f1-score   support

           0       0.58      0.58      0.58      3765
           1       0.57      0.57      0.57      3678

    accuracy                           0.58      7443
   macro avg       0.57      0.57      0.57      7443
weighted avg       0.58      0.58      0.58      7443

**lightGBM**

                precision    recall  f1-score   support

           0       0.59      0.60      0.60      3765
           1       0.58      0.57      0.57      3678

    accuracy                           0.58      7443
   macro avg       0.58      0.58      0.58      7443
weighted avg       0.58      0.58      0.58      7443

**LSTM**
                precision    recall  f1-score   support

           0       0.60      0.55      0.57      3765
           1       0.57      0.61      0.59      3678

    accuracy                           0.58      7443
   macro avg       0.58      0.58      0.58      7443
weighted avg       0.58      0.58      0.58      7443


The results are very close and not satisfactory. Moreover, the LSTM starts to overfit pretty early in the epochs.
6) **reddit_jokes_upvote_ratio_prediction_classifier_3.ipynb**:
In this notebook we again increased the complexity of the model. This time we use BERT which is a Encoder-based transformer with ~66m parameters. The results is slightly better (~3% on total accuracy). Note that BERT overfits after the 2nd epoch.

                precision    recall  f1-score   support

           0       0.61      0.64      0.63      3765
           1       0.62      0.58      0.60      3678

    accuracy                           0.61      7443
   macro avg       0.61      0.61      0.61      7443
weighted avg       0.61      0.61      0.61      7443