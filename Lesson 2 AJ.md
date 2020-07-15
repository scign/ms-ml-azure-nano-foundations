## Lesson 2 notes

---

> Machine learning is a data science technique used to extract patterns from data, allowing computers to identify related data, and forecast future outcomes, behaviors, and trends.

| In | You use | To turn | Into |
| - | - | - | - |
| Traditional programming | Rules | Data | Answers |
| Machine learning | Algorithms | Data and answers | Rules |

---

### Some Applications
* Disease recognition: Identifying whether or not given symptoms or scans are indicative of a particular condition
* Next best actions: Suggesting next actions based on latest actions and learned paths from previous interactions
* Personalized automated customer interactions: Using natural language techniques to help systems interact with humans in an aesthetic way
* Triage routing: Sending incoming messages or requests to the right destination, based on content, consistently and efficiently

---

### History
Early explosions in the 1950s occurred when people [discussed](https://en.wikipedia.org/wiki/Dartmouth_Workshop) whether computers might be able to mimic human intelligence. The [hype](https://en.wikipedia.org/wiki/2001:_A_Space_Odyssey_(film)) led people to overpromise which resulted in funding for projects that didn't deliver on expectations. Funding enthusiasm shrivelled and the first ["AI Winter"](https://en.wikipedia.org/wiki/AI_winter) began around 1974. 6 years later, in 1980, the British government started funding more [projects](https://en.wikipedia.org/wiki/Alvey). Instead of the more generalized approach, specific use cases were targeted looking at learning rules from data under new names such as 'knowledge systems', 'inference engineering' and 'machine learning', but when popularity rose again, the hype followed and disappointment triggered [another AI winter](https://towardsdatascience.com/history-of-the-second-ai-winter-406f18789d45) around 1987.

In 1993 researchers who had been struggling through the dry spell finally started achieving some of the original goals such as [intelligent resource management](https://en.wikipedia.org/wiki/Dynamic_Analysis_and_Replanning_Tool) and [behaviour-based robotics](https://en.wikipedia.org/wiki/Polly_(robot)). Computing power increased rapidly with [Moore's Law](https://en.wikipedia.org/wiki/Moore%27s_law#/media/File:Moore's_Law_Transistor_Count_1971-2018.png) and the [availability of data](https://datafloq.com/read/big-data-history/239) to learn the rules also increased as systems moved the world into the digital age (also fueled by advances in technology that [dropped the price](https://jcmit.net/disk2015.htm) of data storage).

The Deep Learning boom came in 1999 when, thanks to the demand for increased realism in gaming, [dedicated chips](https://xoticpc.com/blogs/news/history-of-gpus) were developed for rendering graphics. GPUs used [parallelization techniques](http://homepages.math.uic.edu/~jan/mcs572/mcs572notes/lec27.html) to compute matrix calculations much faster than CPUs - the [same kinds](https://deeplizard.com/learn/video/6stDhEA0wFQ) of matrix calculations that are used in training neural networks.

---

### The Data Science Process
1. Data collection
1. Data preparation
   * Data ingestion
   * Data cleaning
   * Data manipulation
1. Model training
1. Model evaluation
1. Model deployment
   * Model monitoring
   * Model disposal

This pipeline is iterative and from any step one can return to steps #1, #2, or #3 to improve the model.

---

### Types of Data

Most machine learning algorithms at their core are numerical in nature so it's important to understand what forms data can take on ingestion and how they can be converted to numerical forms for model training.

##### Numerical
* Integers: whole numbers, representing countable properties like quantity or rank
* Floats: decimals, representing continuous properties like size or monetary value

Data can be structured in tabular form:
* Rows could be referred to as entities, observations, instances, points, etc.
* Columns could be referred to as features, attributes, variables, vectors, fields, etc.

A _vector_ is a 1-dimensional array of numbers.

##### Time Series
A series of data points in a 1-dimensionally ordered set. The data points can be n-dimensional but the ordering is a single dimension. The dimension is usually time, but the key concept is that there is some dependency between the value at any specific point, and the values immediately preceeding and following that point. This dependency means that the individual data points have some collinearity and therefore cannot be treated as independent features.

Time series data can be manipulated to reduce the correlation using restructuring techniques like those illustrated in the following links:
* https://machinelearningmastery.com/time-series-forecasting-supervised-learning/#:~:text=Given%20a%20sequence%20of%20numbers,step%20as%20the%20output%20variable.
* https://towardsdatascience.com/how-not-to-use-machine-learning-for-time-series-forecasting-avoiding-the-pitfalls-19f9d7adf424
* https://www.linkedin.com/pulse/how-use-machine-learning-time-series-forecasting-vegard-flovik-phd-1f/

##### Categorical
A value, which could be numerical or not, denoting some group to which the observation belongs. There are [a number of ways](https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63) to treat categorical data, including ordinal encoding (replacing each category label with an integer value) and one-hot encoding (creating a separate true/false field for each label denoting whether or not the observation is in that group).

##### Text
Humans communicate very inefficiently so text data is messy and noisy. **Normalization** may include removing 'stop words' (words that convey little to no information in relation to the problem), and converting words to their base form, e.g. "wildly" to "wild", or "complicating" to "complicate".

Text data can be transformed into numerical values (or **vectorized**) in a wide variety of ways. Here are a few common ones:
* Token counts - one of the simplest ways is to simply chop up the text into "tokens" (letters, words, word pieces, phrases) and count them. Features could be constructed as the number of tokens or "terms" in a piece of text or "document", or flipped and be the number of documents that contain the term, depending on what you are trying to model.
* TF-IDF - A challenge with token counts is that terms that occur in many documents appear significant due to the high counts relative to other words but, depending on the use case, could convey relatively little information on how that document relates to the intended output. One solution is dividing the TF (term frequency) by the inverse of the number of documents it occurs in (inverse document frequency). This results in a measure of the relative "importance" of the term in differentiating that document from the rest.
* Embedding vectors - Another way to convert text into numbers is let another machine learned algorithm do it for you! You can use a pre-trained embedding such as GloVe, or add an 'embedding' layer to a neural net model and have it learn the optimal conversion function for your use case.

##### Image
Images are stored as grids of numbers. A small picture may measure 400 pixels ("picture elements") across and 400 pixels down, and each pixels encodes the colour of that part of the grid, encoded as a mix of red, green and blue light (3 numbers). This picture could therefore be represented by a string of (400 x 400 x 3) 480,000 numbers. The red, green and blue are 'channels'. Grayscale images have just one channel.

Before training, images are often:
* Cropped to a uniform aspect ratio
* Normalized (e.g. mean pixel value in a channel subtracted from each pixel value in that channel)
* Rotated
* Resized
* Denoised and
* Centered.

---

### Scaling Data

| | |
| - | - |
| Standardization | Subtract the mean from each element and divide by the standard deviation |
| Normalization | Subtract the minimum value from each element and divide by the difference between the maximum and minimum |

---

### The ML Ecosystem

* **Libraries** allow you to leverage the work of others so you are not reinventing the wheel.
* **Development environments** help you manage your code so you can focus on functionality rather than refactoring; some also help with writing the code with things like templates and autocompletion, and some help to run code within the environment itself which helps with prototyping and rapid iteration.
* **Cloud services** offer a segregated standardized environment to run pipelines and train models.

---

### Libraries

* Pandas - data ingestion, storage and manipulation
* Numpy - low level mathematical routines optimized for matrix calculations
* Scikit-learn - machine learning models and supporting tooling
* Tensorflow - Google's platform for neural net training
* PyTorch - Facebook's platform for neural net training
* Scipy - scientific, mathematical and engineering functions
* Matplotlib - base level visualization framework
* Seaborn - builds on matplotlib with cleaner outputs and more variation
* Plotly - interactive visualization library
* Bokeh - another interactive visualization library

---

### Learning a Function
The objective of machine learning is to learn a mapping function between the input variables map to an output variable.

Y = f(X)

During the training process the parameters of the mapping function are tweaked through an automated process such that the output Y' is as close to Y as possible for all values of X. The difference between Y and Y' is the model error. This is distinct from the irreducible error which is the error inherent in the assumptions made to model the mapping function.

### Regression

In regression the objective is to learn the relationship that maps the input variables to a continuous variable such as size or score. Linear regression assumes that the mapping function will take the form

Y = c + m1.x + m2.x2 + m3.x3 + ...

Where each x is a feature, each m1,m2,... are the coefficients of their respective features and c is a constant. The 'c' and the 'm's are the parameters that are learned from the data. "Simple" linear regression involves one constant, one coefficient, one input variable and one output variable. Adding more variables becomes [Multiple Linear Regression](https://en.wikipedia.org/wiki/Linear_regression#Simple_and_multiple_linear_regression). This should not be confused with Generalized Linear Regression which is where there are multiple output variables.

To calculate how far off the model is from truth we use a **cost function** which gives a scalar value from the model error for multiple predictions. The objective is to find values for the parameters such that the output of the cost function is minimized.

There are several of assumptions involved in using a linear regression model:
1. The relationship between the features and output is linear. If not, use a different model algorithm.
1. Each feature is independent of all the others. If any features are related the model function will essentially double-count the features and give them more importance than it should.
1. The distribution of the data in each dimension is close to gaussian (normal). If not, try and transform the data so that the distribution is gaussian.

Some tips for data preparation when using linear models:
1. With large values or differently scaled values the model parameters can get very large as the model tries to compensate for different scales. Scaling all features and output to the same range (either through standardization or normalization, or both) helps to keep the parameters manageable and reduce the multi-dimensional distance between the points. Also with standardizing, centering and scaling don't necessarily have to be done using the mean and the standard deviation.
2. Consider whether you can remove noise. Outliers can have a lot of influence on the model parameters.

---

### Parametric vs Non-parametric Algorithms
Parametric algorithms assume that the mapping function takes a known form with parameters that are learned during training. Non-parametric algorithms have a less well-defined structure such that the resulting function may take a wider range of forms.

| | Parametric algorithms | Non-parametric Algorithms |
|-|-|-|
| Pros | Easier to interpret<br>Faster to train<br>Require less data | Fit a large number of forms<br>Fewer assumptions about the data<br>Might fit data more closely |
| Cons | Constrained to a specific form<br>Limited in model complexity<br>Fit real world data poorly | Slower to train<br>Require more data to fit<br>Easier to overfit<br>Can be less interpretable |
| Examples | Logistic Regression<br>Linear Discriminant Analysis<br>Perceptron<br>Naive Bayes<br>Simple Neural Networks | k-Nearest Neighbors<br>Decision Trees like CART and C4.5<br>Support Vector Machines |

---

### Classical ML vs Deep Learning
Deep learning refers specifically to machine learning using neural networks with more than one hidden layer. In this form you can find more complex mapping functions but they are often difficult to interpret and explain, they can easily be overfit and the larger the model the more computational power is required to train the model and generate predictions.

---

### Approaches to Machine Learning
Supervised learning is where each set of input variables is linked to its respective truth output and the model is trained to learn the mapping. Forms of supervised learning includ classification, regression, and similarity learning.

Unsupervised learning is where input data is presented to the algorithm which attempts to extract latent structure from the data. Common types of unsupervised learning are dimensionality reduction such as PCA and LDA, clustering and anomaly detection.

Reinforcement learning is where parameter updates to the model during training come from a reward function rather than an error between predictions and truth. This process is often used in modeling actions of an agent in an environment. It is a less passive process as the actions performed by the agent may influence future inputs.

---

### Trade Offs
#### Overfitting vs Underfitting
A model can be overfit to data meaning that it performs accurately on the data on which it was trained but performs poorly on previously unseen data. A model can also be underfit where it fails to perform well even on the training data. A poor choice of model, poor choice of features or a poor training process can all result in underfitting or overfitting.

#### Bias vs Variance
If you train different models using the same algorithm but different training data (from the same population), you are likely to end up with different models. If you consider the model error with respect to each of the models, the bias is how far off the truth the models are in aggregate, and the variance is the spread of the model error. The more complex a model the higher the chance you'll get a different model error therefore the higher the variance, but the more tightly the complex function will fit the data so the lower the bias.