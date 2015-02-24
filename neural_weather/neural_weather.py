import csv
import sys
import numpy
import datetime
import pandas as pd
import bpnn

###-------------------------------------------------------------------------------------
class NeuralWeather:

      def __init__(self, filename, number_of_locations): # number of locations is an input
          # Sets up the data frame.  The locations are labeled from 1 to 500
          self.filename = filename
          self.nloc = number_of_locations
          self.df = pd.read_csv(filename, header=None)
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
         
      
      def get_sets(self, location, month, day, n_prior_years):
          # creating the training, cross validation, and test sets
          years, temperatures = self.day_history(location, month, day)
          temp = (numpy.array(temperatures) / 100.0).tolist() # because the bpnn network works for values between -1 and 1
          n_observations = len(years) - n_prior_years  # number of patterns for this location
          n_training_set = int(n_observations * 0.6) # the size of training set
          n_xv_set       = int(n_observations * 0.2) # the size of cross validation set
          n_test_set     = n_observations - n_training_set - n_xv_set # the size of test set
          training_set = [[temp[i:(i+n_prior_years)],[temp[i+n_prior_years]]] for i in range(n_training_set)]
          xv_set       = [[temp[i:(i+n_prior_years)],[temp[i+n_prior_years]]] for i in range(n_training_set,n_training_set+n_xv_set)]
          test_set     = [[temp[i:(i+n_prior_years)],[temp[i+n_prior_years]]] for i in range(n_training_set+n_xv_set,n_observations)]
          return training_set, xv_set, test_set


      def prediction_error(self, network, data_set):
          # this routine tabulates the absolute difference between the predictions and actuals from cross validation (or test)
          errors = [numpy.abs(data[1][0]-network.update(data[0])[0]) for data in data_set]
          return float( numpy.mean(errors) )


      def optimize_network(self, location, month, day, iterations):
          # Uses cross validation to optimize the network, by finding the minimum error as a function of n_inputs
          # "n_prior_years" denotes the number of training input nodes.
          # Hence, if temperatures[0:10] are the training inputs, then temperatures[10] is the target
          # So, for 30 data points, I would have 20 training sessions, when the target value goes from temperatures[10] to [29]
          # "iterations" denotes the number of iterations during training.  The higher the better, but also more time expensive.
          networks, n_inputs, predicts, xv_errors, test_errors = ([], range(2,11), [], [], [])
          for n_prior_years in n_inputs:  # I only try between 2-10 input units       
              tr_set, xv_set, ts_set = self.get_sets(location, month, day, n_prior_years) # get the training, validation, and test sets
              network = bpnn.NN(n_prior_years, n_prior_years, 1) # hidden layer has same number of units as inputs.  Output = 1 unit
              network.train(tr_set, iterations=iterations) # training the network
              networks.append(network)
              xv_errors.append( self.prediction_error(network, xv_set) * 100.0)
              test_errors.append( self.prediction_error(network, ts_set) * 100.0)
              predicts.append( int( round( network.update(ts_set[-1][0])[0] * 100.0 ) ) )        
          xv_error, i = min((val, idx) for (idx, val) in enumerate(xv_errors)) # finds the minimum cross-validation error
          network, n_prior_years, predict, xv_error, test_error  = (networks[i], n_inputs[i], predicts[i], xv_error, test_errors[i])
          return network, n_prior_years, predict, xv_error, test_error


      def next_year_forecast(self):
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
                         results = self.optimize_network(location, month, day, iterations=100)
                         prow.append( str(results[2]) )  # append to row
                         xvrow.append( str(results[3]) )
                         trow.append( str(results[4]) )
                  if len(prow) > 1: self.write_output_files(prow, xvrow, trow)


      def setup_output_files(self):
          self.predicter = csv.writer(open('prediction.csv','w'))
          self.xver      = csv.writer(open('xverror.csv','w'))
          self.tester    = csv.writer(open('testerror.csv','w'))

    
      def write_output_files(self, prow, xvrow, trow):
          self.predicter.writerow(prow)
          self.xver.writerow(xvrow)
          self.tester.writerow(trow)
            
###--------------------------------------------------------------------------------                
       

if __name__=='__main__':

   weatherman = NeuralWeather('weather_data.csv', number_of_locations = 5) # initialize the class object
   weatherman.next_year_forecast() # perform the forecast and output to csv file
           
