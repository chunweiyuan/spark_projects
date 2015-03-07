import csv
import sys
import numpy as np
import datetime
import pandas as pd
from kalman import Kalman1D

###-------------------------------------------------------------------------------------
class PandasMaster(object):
      # this class stores the weather data in a Pandas data frame, and set up learning/exporting.

      def __init__(self, file='weather_data.csv', number_of_locations=5, pfile='predict.csv', xfile='xv.csv', tfile='test.csv'): 
          # number of locations is an input
          # Sets up the data frame.  The locations are labeled from 1 to 500
          self.pfile    = pfile
          self.xfile    = xfile
          self.tfile    = tfile
          self.nloc = number_of_locations
          self.df = pd.read_csv(file, header=None)
          self.df = self.df.rename(columns={0:'date'})
          self.df.set_index(['date'], drop=True, inplace=True)
          self.df.index = self.df.index.tolist()
          self.years = map(lambda x: int(x[0:4]),filter(lambda x: x[-5:]=='01-01',self.df.index))


      def month_day_string(self, month, day):
          month_string = str(month) if month >= 10 else '0'+str(month)
          day_string   = str(day) if day >= 10 else '0'+str(day)
          return month_string + '-' + day_string


      def get_temperature(self, loc, year, month, day):
          # location = 1 to 500
          # this is a routine that tells the temperature of a given location, at a particular year-month-day
          # if there's no temperature recording for that day, or if such a day does not exist, it returns None         
          date = str(year) + '-' + self.month_day_string(month, day)
          temperature = self.df.loc[date,loc] if (date in self.df.index) and (loc in self.df.columns) else None
          return temperature
          
      
      def day_history(self, loc, month, day):
          # this is a routine that looks at a particular location, for a particular month-day,
          # all the past recorded temperatures
          monthday = self.month_day_string(month, day)
          return map(lambda x: int(x[0:4]),filter(lambda x: x[-5:]==monthday, self.df.index)),\
                 self.df.loc[filter(lambda x: x[-5:]==monthday, self.df.index),loc].tolist()
         
      
      def filter(self, series, n): # filters the series up to the n-th element
          E = (max(series) - min(series)) / 2.0
          kalman =  Kalman1D( A = np.matrix([1]), B = np.matrix([0]), H = np.matrix([1]), 
                              x0 = np.matrix([series[0]]), P0 = np.matrix([E]), 
                              Q = np.matrix([E]), R = np.matrix([E]) )
          for i in range(n):
              kalman.update(np.matrix([0]),np.matrix([series[i]]))
              series[i] = kalman.current_state()[0]
          return series


      def get_sets(self, location, month, day, n_prior_years, filter=False):
          # creating the training, cross validation, and test sets
          years, temperatures = self.day_history(location, month, day)
          inputs, outputs = ((np.array(temperatures)).tolist(), (np.array(temperatures)).tolist())
          n_observations = len(years) - n_prior_years  # number of patterns for this location
          n_training_set = int(n_observations * 0.6) # the size of training set
          n_xv_set       = int(n_observations * 0.2) # the size of cross validation set
          n_test_set     = n_observations - n_training_set - n_xv_set # the size of test set
          if filter: inputs = self.filter(inputs, len(inputs))
          training_set = [[inputs[i:(i+n_prior_years)],[outputs[i+n_prior_years]]] for i in range(n_training_set)]
          xv_set       = [[inputs[i:(i+n_prior_years)],[outputs[i+n_prior_years]]] for i in range(n_training_set,n_training_set+n_xv_set)]
          test_set     = [[inputs[i:(i+n_prior_years)],[outputs[i+n_prior_years]]] for i in range(n_training_set+n_xv_set,n_observations)]
          return training_set, xv_set, test_set


      def learn(self, location, month, day):
          # this is just a placeholder here.  Will be overridden by subclasses.
          return [0,0,0,0,0]


      def next_year_forecast(self, filter=False):
          # this is a routine that performs next year's forecast, and then pipes them out to csv files
          # it does not forecast February 29s    
          self.setup_output_files() 
          next_year = max(self.years) + 1 # this is the "next year"
          four_years_ago = next_year - 4 # well, just in case this is also a leap year           
          for month in range(1,13): # go through the 12 months
              for day in range(1,32): # go through 31 days for each month
                  date = str(next_year) + '-' + self.month_day_string(month, day) # the date string
                  prow, xvrow, trow  = ([date],[date],[date]) # initialize the "row" to be outputed
                  for location in range(1,self.nloc+1): # location index
                      if self.get_temperature(location, four_years_ago, month, day) is not None:
                         results = self.learn(location, month, day, filter=filter)
                         prow.append( str(results['predict']) )  # append to row
                         xvrow.append( str(results['xverr']) )
                         trow.append( str(results['terr']) )
                  if len(prow) > 1: self.write_output_files(prow, xvrow, trow)


      def setup_output_files(self):
          self.predicter = csv.writer(open(self.pfile,'w'))
          self.xver      = csv.writer(open(self.xfile,'w'))
          self.tester    = csv.writer(open(self.tfile,'w'))

    
      def write_output_files(self, prow, xvrow, trow):
          self.predicter.writerow(prow)
          self.xver.writerow(xvrow)
          self.tester.writerow(trow)


###--------------------------------------------------------------------------------                
       

if __name__=='__main__':
   pass
