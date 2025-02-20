import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR

def svr_train_predict(x_train, y_train, x_test):
    #scale the data
    #data preprocessing
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler() 

    X_train_scaled = scaler_X.fit_transform(x_train)
    X_test_scaled = scaler_X.transform(x_test)

    Y_train_scaled = scaler_Y.fit_transform(y_train.values.reshape(-1,1))

    #create the svr model
    model = SVR()
    
    #define the parameters using RandomSearchCV
    param_random = {
        'kernel' : ['linear', 'rbf'],
        'C' : [0.1, 1000],
        'gamma' : ['scale', 0.01, 0.1, 1]
    }

    #perform random search rather than grid search because large dataset
    randomSearch = RandomizedSearchCV(model, param_distributions = param_random, cv = 3, n_jobs = -1, verbose = 2, n_iter = 20 )#could use random_state for more consistent results

    #fit the model
    randomSearch.fit(X_train_scaled, Y_train_scaled.ravel())

    model = randomSearch.best_estimator_#return the best model

    #predict the data
    predictions_scaled = model.predict(X_test_scaled)

    #inverse the scaling to get actual predictions
    predictions = scaler_Y.inverse_transform(predictions_scaled.reshape(-1,1))

    return predictions

    