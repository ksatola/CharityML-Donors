# Finding Donors for CharityML

## Overview
CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. With nearly 15 million working Californians, CharityML wants an algorithm to best identify potential donors and reduce overhead cost of sending mail.

This project was completed as part of Udacity's [Data Scientist Nanodegree](https://eu.udacity.com/course/data-scientist-nanodegree--nd025) certification.

## Objective
The goal of this project is to evaluate and optimize several different supervised learners to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent. Specifically, the goal is to construct and optimize a model that accurately predicts whether an individual makes more than $50,000 (using data collected from the 1994 U.S. Census).

## Data Origin
The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. The paper is available [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**
- `income`: Income Class (<=50K, >50K)

## Tools, Software and Libraries
This project uses the following software and Python libraries:
- Python 3.6
- NumPy
- Pandas
- Scikit-learn (preprocessing, model_selection, metrics, ensemble, naive_bayes, linear_model, base)
- Matplotlib
- Seaborn
- Jupyter Lab

## Results
- The provided data were explored and prepared as input for machine learning algorithms (skewed data distributions transformation, numerical features normalization, data preprocessing of categorical features with one-hot encoding, shuffling and data split into test and train datasets).
- Three models were selected from available in Sklearn supervised learning models (RandomForestClassifier, GaussianNB, LogisticRegression).
- The selected models were incorporated into a training and predicting pipeline and an initial models evaluations was performed based on model evaluation matrics like accuracy, F beta score (combining precision and recall into a mathematical formula).
- The logistic regression model was chosen and its hyper-parameters were fine-tuned using GridSearch.
- After the best hyperparameters identification, feature importance was checked. Finally no features were removed from the datased based on worse model performance with the limited set of features (with the F-score of 0.5204, comparing to a model trained on full data set score of 0.68).

## Details
- [HTML Preview](https://ksatola.github.io/projects/finding_donors.html)
- [Jupyter Notebook](https://github.com/ksatola/ml-introduction/blob/master/finding_donors.ipynb)
