This is essentially a python code written to predict the max temperature of 500 locations, for everyday over the entire year of 2010.  

The code would first read in temperatures of the 500 locations, for everyday from 1980 to 2009, from the file "training.csv".  These data will then be used to train/cross-validate/test an artificial neural network (one separate network for every location) to predict the 2010 temperatures.  Outputs are written to output.csv.  

It's entirely questionable why anyone would use a neural net for this particular machine learning task.  A Kalman filter might make more sense, given the time-series nature of it.  In the end, it's just a fun project with interesting data constraints that combines neural network, optimization of network parameters via cross-validation, and making predictions.  In addition, it's very parallelizable, so it's a nice starting point to test out Spark.

To run the code, simply type

python neural_weather.py

To run on Spark (locally, with 2 cores), 

MASTER=local[2] /usr/local/bin/spark/bin/spark-submit neural_weather.py

One can change the number_of_locations in the code to as high as 500.  Also, cross-validation/test errors are computed in the object, but not exported.