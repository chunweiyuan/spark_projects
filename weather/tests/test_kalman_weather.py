"""
PandasMaster test set
"""
import unittest
import os
import sys
sys.path.insert(0,'../')
from kalman_weather import KalmanWeather

class TestNeuralWeather(unittest.TestCase):
   
      def setUp(self):
          """
          Essentially testing that init goes through
          """
          self.file    = '../weather_data.csv'
          self.num_col = 20       
          self.pfile   = 'predict_test.csv'
          self.xfile   = 'xv_test.csv'
          self.tfile   = 'test_test.csv'
          self.weatherman = KalmanWeather(file = self.file,
                                         number_of_locations = self.num_col,
                                         pfile = self.pfile,
                                         xfile = self.xfile,
                                         tfile = self.tfile)
          self.series = range(10)


      def tearDown(self):
          pass
          #os.remove(self.pfile)
          #os.remove(self.xfile)
          #os.remove(self.tfile)


      def test_learn(self):
          """
          Tests whether learn is returning the right format
          """
          results = self.weatherman.learn(1,1,1,filter=False)
          self.assertTrue(-50 < results['predict'] <= 50)
          self.assertTrue(isinstance(results['xverr'],float))
          self.assertTrue(isinstance(results['terr'],float))


      def test_create_date_list(self):
          dates = self.weatherman.create_date_list(year=1985)
          self.assertTrue( len(dates) == 365 )
          dates = self.weatherman.create_date_list(year=2008)
          self.assertTrue( len(dates) == 366 )


if __name__ == '__main__':
   unittest.main()
