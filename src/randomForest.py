import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

def randomForest_train_predict(x_train, y_train, x_test):

    #create model
    model = RandomForestRegressor()

    param_random = {
        'n_estimators' : randint(50,500),
        'max_depth' : randint(5, 50),
        'max_features' : ['log2', 'sqrt' ],
        'min_samples_split' : [2, 5, 10],
        'min_samples_leaf' : [1, 2, 4],
    }

    #RandomizedSearchCV to get the best model with the best hyperparamters
    randomSearch = RandomizedSearchCV(model, param_distributions=param_random, cv = 3, n_jobs= -1, verbose= 2, n_iter= 20)

    #train the model
    randomSearch.fit(x_train, y_train)

    model = randomSearch.best_estimator_ #return the best model

    #predict the data
    predictions = model.predict(x_test)

    return predictions
