In this project, we explored the problem of predicting the NIL Deal for top college football recruits.

This project uses a machine learning model to predict what kind of NIL (Name, Image, Likeness) deal a high school football recruit might get at a college.

It looks at the recruit’s star rating, position, and how much NIL money the school usually spends. Then it tells you if the player would get a Tier A, Tier B, or Tier C NIL offer.


## Them Boys - Our project objectives

1. To classify each recruit into one of our three established NIL tiers:

A Tier: NIL Budget >= $10 Million
B Tier: $5-10 million
C Tier: $5 million

# dataset overview

- top 100 football recruits, plus adding synthetics playesr for balance
- includes player stats (overall rankings, position, position ranking, committed school, and star rating)
-  each specific recruiting school and their estimated NIL Budgetp
- dataset is at "data/nil_dataset_diverse.csv"


# theres two python files:

- main.py
trains the model using a csv file with real and synthetic recruiting and NIL data

- app.py
a simple program when you enter info about your own made-up player, and it predicts their NIL tier

# tools/libraries used

- pandas
working with the data

- scikit-learnn
for machine learning

- matplotlib
for making the feature importance graph

- joglib
saving and loading the model

- logistic regression
used to compare how a simpler model would perform

# how to run

1. Install the right libraries, run this in your terminal
pip install pandas scikit-learn matplotlib joblib

2. train the model first, run this line
python3 main.py

3. use this app to predict nil tiers for your made-up player
python3 app.py


# this model is trained on star rating, position and the players committed school's NIL budget. We also added a new features that combines the rating and budget to make predictiosn more realistic


# results

1. Random Forest Accuracy:
97% with cross-validation

2. Feature importance:
The budget and rating combo had the biggest impact

3. Graph outpuit
Saved to output/feature_importance.png

# Notes
1. we added more diverse players to the dataset to have more accurate prediction
