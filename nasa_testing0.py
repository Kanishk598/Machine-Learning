import numpy as np
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import random
def nasa():
    mydata = pd.read_csv("C:\\Users\\yvish\\Desktop\\nasa-asteroids-classification\\nasa.csv")
#print(mydata.keys())

#mydata.dtypes

    del mydata['Equinox']
    del mydata['Orbiting Body']
    del mydata['Orbit ID']
    del mydata['Neo Reference ID']
    del mydata['Name']

    mydata['Close Approach Date'] = pd.to_datetime(mydata['Close Approach Date'])
    mydata['Orbit Determination Date'] = pd.to_datetime(mydata['Orbit Determination Date'])

    mydata['Close Approach Date'] = pd.to_numeric(mydata['Close Approach Date'])
    mydata['Orbit Determination Date'] = pd.to_numeric(mydata['Orbit Determination Date'])

    x_train, x_test, y_train, y_test = train_test_split(mydata[['Absolute Magnitude', 'Est Dia in KM(min)', 'Est Dia in KM(max)','Est Dia in M(min)', 'Est Dia in M(max)', 'Est Dia in Miles(min)',
       'Est Dia in Miles(max)', 'Est Dia in Feet(min)', 'Est Dia in Feet(max)',
       'Close Approach Date', 'Epoch Date Close Approach',
       'Relative Velocity km per sec', 'Relative Velocity km per hr',
       'Miles per hour', 'Miss Dist.(Astronomical)', 'Miss Dist.(lunar)',
       'Miss Dist.(kilometers)', 'Miss Dist.(miles)',
       'Orbit Determination Date', 'Orbit Uncertainity',
       'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant',
       'Epoch Osculation', 'Eccentricity', 'Semi Major Axis', 'Inclination',
       'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance',
       'Perihelion Arg', 'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly',
       'Mean Motion']], mydata['Hazardous'], random_state = random.randint(1,100), shuffle=True)


    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_test)
    return("Accuracy of the model out of 1.00 is: {}".format(np.mean(prediction == y_test)))
    