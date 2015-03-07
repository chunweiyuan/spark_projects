from pandas_master import *
import bpnn

###-------------------------------------------------------------------------------------
class NeuralWeather(PandasMaster):

      def __init__(self, file='weather_data.csv', number_of_locations=5, pfile='predict.csv', xfile='xv.csv', tfile='test.csv'): # number of locations is an input
          # Sets up the data frame.  The locations are labeled from 1 to 500
          super(NeuralWeather, self).__init__(file, number_of_locations, pfile, xfile, tfile)


      def prediction_error(self, network, data_set):
          # this routine tabulates the absolute difference between the predictions and actuals from cross validation (or test)
          errors = [np.abs(data[1][0]-network.update(data[0])[0]) for data in data_set]
          return float( np.mean(errors) )

      
      def normalize(self, nested_data_list):
          # this takes the nested lists provided by self.get_set and divides every element by 100.0
          return [[(np.asarray(x)/100.0).tolist() for x in sublist] for sublist in nested_data_list]


      def learn(self, location, month, day, filter=False, iterations=100):
          # Uses cross validation to optimize the network, by finding the minimum error as a function of n_inputs
          # "n_prior_years" denotes the number of training input nodes.
          # Hence, if temperatures[0:10] are the training inputs, then temperatures[10] is the target
          # So, for 30 data points, I would have 20 training sessions, when the target value goes from temperatures[10] to [29]
          # "iterations" denotes the number of iterations during training.  The higher the better, but also more time expensive.
          networks, n_inputs, predicts, xv_errors, test_errors = ([], range(2,11), [], [], [])
          for n_prior_years in n_inputs:  # I only try between 2-10 input units       
              tr_set, xv_set, ts_set = self.get_sets(location, month, day, n_prior_years, filter=filter) # get the training, validation, and test sets
              tr_set, xv_set, ts_set = (self.normalize(tr_set), self.normalize(xv_set), self.normalize(ts_set))
              network = bpnn.NN(n_prior_years, n_prior_years, 1) # hidden layer has same number of units as inputs.  Output = 1 unit
              network.train(tr_set, iterations=iterations) # training the network
              networks.append(network)
              xv_errors.append( self.prediction_error(network, xv_set) * 100.0)
              test_errors.append( self.prediction_error(network, ts_set) * 100.0)
              predicts.append( int( round( network.update(ts_set[-1][0])[0] * 100.0 ) ) )       
          xv_error, i = min((val, idx) for (idx, val) in enumerate(xv_errors)) # finds the minimum cross-validation error
          network, n_prior_years, predict, xv_error, test_error  = (networks[i], n_inputs[i], predicts[i], xv_error, test_errors[i])
          return dict(obj=network, n_prior_years=n_prior_years, predict=predict, xverr=xv_error, terr=test_error)

###--------------------------------------------------------------------------------                
       

if __name__=='__main__':

   weatherman = NeuralWeather(file='weather_data.csv', number_of_locations=5,\
                              pfile='predict.csv', xfile='xv.csv', tfile='test.csv') # initialize the class object
   weatherman.next_year_forecast(filter=False) # perform the forecast and output to csv file
           
