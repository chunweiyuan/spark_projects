import csv
import sys
import numpy
import datetime
#from scipy import stats
#import numpy.linalg
import bpnn

###-------------------------------------------------------------------------------------
class NeuralWeather:
      # This class was developed before I learned Pandas.
      # A lot of the data manipulation would've been simpler with Pandas.

      def __init__(self, filename, number_of_locations): # number of locations is an input
          
          self.filename = filename
          self.number_of_locations = number_of_locations
          self.number_of_months = 12
          self.number_of_days = 31
          self.years = []  # the years of the inputs will be stored here
          self.leap_years = [] # this stores the leap years that are covered
          self.biglist = []  # this is the "database", to be initialized in the following loop
 
          # this generates my "database" in order of "location", "day", "month",
          # so that when I look up self.biglist[i][j][k], 
          # it'll give me, for this location and this day of the year,
          # all the temperatures from the past years
          # this is because I use temperatures from the past years to extrapolate for the next year
          for i in range(self.number_of_locations):
              self.biglist.append([])
              for j in range(self.number_of_days):
                  self.biglist[i].append([])
                  for k in range(self.number_of_months):
                      self.biglist[i][j].append([])
          self.fill_data_list()
          self.years.sort()


      def fill_data_list(self):
          f = open(self.filename, 'rU')
          reader = csv.reader(f, delimiter=',', quotechar='"')
          for row in reader: 
              date = row[0]
              year, month, day = date.split('-')
              temperatures = numpy.array(row[1:],dtype='int') # convert temperatures to a list of integers
              self.insert(int(year),int(month),int(day),temperatures) # fill up the "database"

      
      def insert(self,year,month,day,temperatures): 
          # this is the routine that inserts data into the "database"     
          # I'm here assuming that the input rows are not sorted by date,
          # such that the data would be given in random years, 2002 followed by 1983, etc....
          # hence I'd need these if-else statements          
          if self.years.count(year) == 0: # if this is a year that has not shown up before
             self.years.append(year)
             if (month==2 and day==29):  # account for the leap years
                self.leap_years.append(year) 
             for i in range(self.number_of_locations): # here we insert the temperature for all locations, at this day and this month
                 self.biglist[i][day-1][month-1].append(temperatures[i])
          else: # if this is a year that's already had something on the record 
             index = self.years.index(year)
             if (month==2 and day==29):  # it feels a bit silly accounting for leap years, but oh well.
                self.leap_years.append(year)
                index = self.leap_years.index(year)
             for i in range(self.number_of_locations):  # insert temperature for all locations
                 self.biglist[i][day-1][month-1].insert(index,temperatures[i])

      
      def tell(self,location,year,month,day):
          # this is a routine that tells the temperature of a given location, at a particular year-month-day
          # all values returned in lists
          # if there's no temperature recording for that day, if such a day does not exist,
          # it returns []
          temperatures = self.biglist[location-1][day-1][month-1]
          if len(temperatures) == len(self.years): # data point exists, not a leap year
             year_index = self.years.index(year)
             return [temperatures[year_index]]
          else:
             if (month==2 and day==29): # if it's 2/29
                if year in self.leap_years: # check to see if it's actually a leap year
                   year_index = self.leap_years.index(year) # yes, then find the index
                   return [temperature[year_index]]
                else:
                   return []  # data point does not exist
             else:
                return []  # data point does not exist
          
      
      def day_history(self,location,month,day):
          # this is a routine that looks at a particular location, for a particular month-day,
          # all the past recorded temperatures
          if (month==2 and day==29):
             return self.leap_years, self.biglist[location-1][day-1][month-1]
          else:
             return self.years, self.biglist[location-1][day-1][month-1]

      
      def get_sets(self, location, month, day, n_prior_years):
          # creating the training, cross validation, and test sets
          years, temperatures = self.day_history(location, month, day)
          temp = numpy.array(temperatures) / 100.0 # because the bpnn network works for values between -1 and 1
          temp = temp.tolist()  # convert back to list
          L = len(years)  # this is the number of data points 
          n_observations = L - n_prior_years  # number of patterns for this location
          n_training_set = int(n_observations * 0.6) # the size of training set
          n_xv_set       = int(n_observations * 0.2) # the size of cross validation set
          n_test_set     = n_observations - n_training_set - n_xv_set # the size of test set
          training_set = []
          xv_set       = []
          test_set     = []
          for i in range(n_training_set):             
              training_set.append([temp[i:(i+n_prior_years)],[temp[i+n_prior_years]]]) # contruct pattern to be learned
          for i in range(n_training_set, n_training_set + n_xv_set):
              xv_set.append([temp[i:(i+n_prior_years)],[temp[i+n_prior_years]]])
          for i in range(n_training_set + n_xv_set, n_observations):
              test_set.append([temp[i:(i+n_prior_years)],[temp[i+n_prior_years]]])
          return training_set, xv_set, test_set


      def prediction_error(self, network, data_set):
          # this routine tabulates the absolute difference between the predictions and actuals from cross validation (or test)
          errors = []
          for data in data_set:
              prediction = network.update(data[0])
              delta      = numpy.abs(data[1][0] - prediction[0])
              errors.append(delta)
          return float( numpy.mean(errors) )


      def optimize_network(self, location, month, day, iterations):
          # Uses cross validation to optimize the network, by finding the minimum error as a function of n_inputs
          # "n_prior_years" denotes the number of training input nodes.
          # Hence, if temperatures[0:10] are the training inputs, then temperatures[10] is the target
          # So, for 30 data points, I would have 20 training sessions, when the target value goes from temperatures[10] to [29]
          # "iterations" denotes the number of iterations during training.  The higher the better, but also more time expensive.
          xv_errors = []
          test_errors = []
          networks = []
          predicts = []
          n_inputs = range(2,11)
          for n_prior_years in n_inputs:  # I only try between 2-10 inputs       
              tr_set, xv_set, ts_set = self.get_sets(location, month, day, n_prior_years) # get the training, validation, and test sets
              network = bpnn.NN(n_prior_years, n_prior_years, 1) # hidden layer has same number of units as inputs.  Output = 1 unit
              network.train(tr_set, iterations=iterations) # training the network
              networks.append(network)
              xv_errors.append( self.prediction_error(network, xv_set) * 100.0)
              test_errors.append( self.prediction_error(network, ts_set) * 100.0)
              predicts.append( int( round( network.update(ts_set[-1][0])[0] * 100.0 ) ) )        
          xv_error, index = min((val, idx) for (idx, val) in enumerate(xv_errors)) # finds the minimum cross-validation error
          network  = networks[index]
          test_error = test_errors[index]
          n_prior_years = n_inputs[index]
          prediction = predicts[index]
          return network, n_prior_years, prediction, xv_error, test_error


      # this is a routine that performs next year's forecast, and then pipes them out to stdout
      def next_year_forecast(self):
          self.setup_output_files()      
          next_year = max(self.years) + 1 # this is the "next year"
          four_years_ago = next_year - 4 # well, just in case this is also a leap year           
          for i in range(self.number_of_months):
              month = i + 1
              for j in range(self.number_of_days):
                  day = j + 1
                  date = str(next_year) + '-' + str(month) + '-' + str(day) # the date string
                  prow, xvrow, trow  = ([date],[date],[date]) # initialize the "row" to be outputed
                  for k in range(len(self.biglist)): # go through the locations
                      location = k + 1 # index of the locations
                      # because j goes from 1 to 31, and not all months have 31 days, so 
                      # I need the following conditional statement, such that
                      # I only evaluate the days that exist
                      if len( self.tell(location, four_years_ago, month, day) ) > 0:
                         results = self.optimize_network(location,month,day,iterations=100)
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
           
