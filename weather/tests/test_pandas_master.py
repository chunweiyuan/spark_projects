"""
PandasMaster test set
"""
import unittest
import os
import sys
sys.path.insert(0,'../')
from pandas_master import PandasMaster

class TestPandasMaster(unittest.TestCase):
   
      def setUp(self):
          """
          Essentially testing that init goes through
          """
          self.file    = '../weather_data.csv'
          self.num_col = 20       
          self.pfile   = 'predict_test.csv'
          self.xfile   = 'xv_test.csv'
          self.tfile   = 'test_test.csv'
          self.weatherman = PandasMaster(file = self.file,
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
                             
 
      def test_date_format(self):
          """
          Confirms the dates are in YYYY-MM-DD format
          """
          date = self.weatherman.df.index[0]
          year, month, day = date.split('-')
          year = int(year)
          month = int(month)
          day = int(day)
          self.assertTrue(year > 1900)
          self.assertTrue(1 <= month <= 12)
          self.assertTrue(1 <= day <= 31)


      def test_month_day_string(self):
          """
          Test the MM-DD date string construction
          """
          date = self.weatherman.month_day_string(12,31)
          self.assertEqual(date, '12-31')
          date = self.weatherman.month_day_string(1,1)
          self.assertEqual(date, '01-01')


      def test_get_temperature(self):
          """
          Test the temperature retrieval process
          """
          t = self.weatherman.get_temperature(loc = 1,
                                              year = 2000,
                                              month = 5,
                                              day = 25)
          self.assertTrue(isinstance(t, int))
          self.assertTrue(-100 < t < 100)


      def test_filter(self):
          """
          Tests whether filter returns a list of same length as input list.
          Also tests whether it filters only up to the index indicated
          """
          s = self.weatherman.filter(self.series, len(self.series)-1)
          self.assertTrue(len(s), len(self.series))
          self.assertEqual(s[-1],self.series[-1])
          

      def test_get_sets(self):
          """
          Tests whether the inputs are constructed with equal length across sets
          """
          trset, xvset, teset = self.weatherman.get_sets(1,1,1,3,filter=False)
          self.assertTrue(len(trset[0][0]) == len(xvset[0][0]) == len(teset[0][0]))


      def test_learn(self):
          """
          Tests whether learn is returning the right format
          """
          results = self.weatherman.learn(1,1,1,filter=False)
          self.assertTrue(isinstance(results['predict'],float))
          self.assertTrue(isinstance(results['xverr'],float))
          self.assertTrue(isinstance(results['terr'],float))


if __name__ == '__main__':
   unittest.main()
