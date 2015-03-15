## Intro

This is essentially a python code written to predict the max temperature of 500 locations, for everyday over the entire year of 2010.  

## Neural Net (neural_weather.py)

The code would first read in temperatures of the 500 locations, for everyday from 1980 to 2009, from the file "weather_data.csv".  These data will then be used to train/cross-validate/test an artificial neural network (one separate network for every location) to predict the 2010 temperatures.  Outputs are written to output.csv.  

It's entirely questionable why anyone would use a neural net for this particular machine learning task.  In the end, it's just a fun project with interesting data constraints that combines neural network, optimization of network parameters via cross-validation, and making predictions.  In addition, it's very parallelizable, so it's a nice starting point to test out Spark.

To run the code, simply type

```
python neural_weather.py
```

To run on Spark (locally, with 2 cores), 

```
MASTER=local[2] /usr/local/bin/spark/bin/spark-submit neural_weather.py
```

One can change the number_of_locations in the code to as high as 500.  Also, cross-validation/test errors are computed in the object, but not exported.


## Kalman Filter (kalman_weather.py)

The Kalman filter implemented here assumes there's no control vector; as such, it's used essentially as a noise filtering technique (see the ipython notebook in this repo).  One can run the filter through the temperature measurements for each location first, before feeding the filtered data into the neural network for training/prediction.

Alternatively, one could also assume that the underlying state is stationary, and everything observed is just noise.  Then the prediction for next year's max temperature, for a given location and date, is just the current tempearture state is that location/date.


## Simple Average
Within the PandasMaster class, the native learning algorithm is a simple average of the first 29 years, tested on the 30th.  When tested over 20 locations over the entire year of 2010, this simple average turns out to yield the best prediciton (as shown in the ipython notebook):

```
e(simple avg) < e(kalman + simple avg) < e(kalman + neural net) < e(neural net) 
```

