## Lesson 3 notes

---

### Data Import and Transformation
This is one of the most important, if not *the* most important steps. Data in the real world is rarely clean and most machine learning algorithms are highly sensitive to the data quality. We need to iterate through evaluation of the data quality and modifying the data. The end result is publishing the dataset to make it available for training and/or prediction.

| Datastore | Dataset |
|-|-|
| * Abstraction from storage medium or compute resource<br>* Increased security<br>* Support a wide variety of cloud storage modes | * Specific sets of files<br>* Can point to datastores<br>* Can track data versions |
| Actual storage of data | Window to the stored data |

While setting up an ingestion pipeline we should consider how the data may change over time. The real data the model is exposed to will not be the same as the training data (otherwise there would be no point training a model) but *how* different? **Data Monitors** help to keep track of how the real data changes over time compared to the training data. If that variance exceeds a threshold the model should be retrained or disposed.

---

### Feature Engineering
Real data is unlikely to be perfect for mapping to your target variable. Feature engineering - the process of adding new features or removing unnecessary ones - can be performed at the data source, during data ingestion, through the manipulation process, even during the training process if the model algorithm creates new features itself and uses them as part of the trained model.
* **Not enough**
Sometimes you may not have enough information to predict your target variable and models you train are either poor models or don't generalize well. In this case we can try creating new features by joining to other data, or performing manipulations on other features.
* **Too many**
Many algorithms perform poorly with too much information just as with too little. We can discard features that may not correlate well with the target variable, or we could combine them with other features through 'dimensionality reduction' such as PCA or LDA.

#### Correlated Features
If a two features by nature vary together across the dataset they are [correlated](https://www.theanalysisfactor.com/multicollinearity-explained-visually/). Many machine learning models will treat both variables as if they were independent and lend more weight to the underlying information than they should. The effect of this is that model parameters will be learned that are influenced by both features in the same direction i.e. the model is being pulled harder in that direction at the expense of other feature dimensions.

#### Dimensionality Reduction
When you have a large number of features / columns, the dataset is said to have *high dimensionality*. Why is this bad? Because as you increase the number of dimensions, the distance between points increases, eventully getting to a point where points are so far apart that identifying similarities and patterns becomes almost meaningless.

###### Resources that help understand this concept
https://www.youtube.com/watch?v=zwAD6dRSVyI&t=550
https://stats.stackexchange.com/questions/169156/explain-curse-of-dimensionality-to-a-child

### Model Drift

In the real world, input data doesn't stay the same. Things change. A solution must be able to measure change in the input data, else the model will no longer reflect the right patterns.

### Model Training

1. Choose hyperparameters
1. Split the data into a training set (train the model), a validation set (optional, validate that the chosen trauned model works), and a test set (to evaluate the training hyperparameters chosen)

### Evaluation

#### Classification
A confusion matrix shows the correct and incorrect classification of items. Actual class is usually on the top and Predicted on the left.

### Multiple learners
As amazing as machine learning is, a model will never be perfect. Indeed some algorithms will only ever fit real world data poorly (for example linear regression models). Just as looking at a sample of data points helps to understand the population from which it was taken, training multiple models from the same data, either using the same algorithm with some random parameters, or using multiple algorithms, the collective power can be increased.
#### Ensemble Learning
* **Bootstrap Aggregation (Bagging)** means training multiple models on subsamples of the training data and averaging the predictions.
* **Boosting** means training multiple models using the same training data but different hyperparameters.
* **Stacking** means training multiple models using different algorithms and combining the outputs.
#### Automated Machine Learning (AutoML)
AutoML uses automation to repeatedly train and score a number of models using different feature sets, cleaning steps, algorithms and hyperparameters, etc. We can then review the scores and pick the model that performed the best based on our use case.
