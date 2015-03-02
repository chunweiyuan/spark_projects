from pandas_master import *
from kalman import Kalman1D

###-------------------------------------------------------------------------------------
class KalmanWeather(PandasMaster):

      def __init__(self, file='weather_data.csv', number_of_locations=5, 
                   pfile='predict.csv', xfile='xv.csv', tfile='test.csv'): # number of locations is an input
          # Sets up the data frame.  The locations are labeled from 1 to 500
          super(KalmanWeather, self).__init__(file, number_of_locations, pfile, xfile, tfile)


      def learn(self, location, month, day):
          # Simply applies the filter over all past data to get the "state" (temperature).
          # The state is hence the predicted value.
          tr_set, xv_set, ts_set = self.get_sets(location, month, day, n_prior_years=0)
          data_set = tr_set + xv_set + ts_set  # I actually want all data strung together to improve the filter
          years, temperatures = self.day_history(location, month, day)
          E = float(max(temperatures) - min(temperatures)) / 2.0 # approximate the noise covariance
          kalman = Kalman1D( A = numpy.matrix([1]), B = numpy.matrix([0]), H = numpy.matrix([1]),
                             x0 = numpy.matrix(data_set[0][1]), P0 = numpy.matrix([E]),
                             Q = numpy.matrix([E]), R = numpy.matrix([E]) ) # initialize the filter
          u = numpy.matrix([0]) # the control vector is 0 in this case
          for data in data_set: # if n_prior_years = 30, this should really just be one 
              kalman.update(u, numpy.matrix(data[1]))
          x, P = kalman.current_state()
          return dict(obj=kalman, n_prior_years=0, predict=x,
                      xverr=numpy.sqrt(P), terr=numpy.sqrt(P) )

###--------------------------------------------------------------------------------                
       

if __name__=='__main__':

   weatherman = KalmanWeather(file='weather_data.csv', number_of_locations=5,\
                              pfile='predict.csv', xfile='xv.csv', tfile='test.csv') # initialize the class object
   weatherman.next_year_forecast() # perform the forecast and output to csv file
           
