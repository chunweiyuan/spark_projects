"""
PandasMaster test set
"""
import unittest
import os
import sys
sys.path.insert(0,'../')
from neural_weather import NeuralWeather

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
          self.weatherman = NeuralWeather(file = self.file,
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
                             
 
      def test_normalize(self):
          """
          makes sure the temperatures are divided by 100
          """
          nested_data_list = [[[12,38,-99],[8]],[[52,13,-87],[25]],[[36,-2,10],[99]]]
          newlist = self.weatherman.normalize(nested_data_list)
          self.assertEqual(len(nested_data_list),len(newlist))
          self.assertTrue( max(newlist[0][0]) <= 1.0)
          self.assertTrue( min(newlist[0][0]) >= -1.0)
          self.assertTrue( max(newlist[2][1]) <= 1.0)


      def test_learn(self):
          """
          Tests whether learn is returning the right format
          """
          results = self.weatherman.learn(1,1,1,filter=False)
          self.assertTrue(-50 < results['predict'] <= 50)
          self.assertTrue(isinstance(results['xverr'],float))
          self.assertTrue(isinstance(results['terr'],float))


if __name__ == '__main__':
   unittest.main()
