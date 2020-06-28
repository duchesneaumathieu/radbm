import unittest
from time import time, sleep

from radbm.utils.time import *

class TestChronometer(unittest.TestCase):
    def setUp(self):
        self.chrono = Chronometer()
            
    def test_basic_timing(self):
        self.chrono.start()
        sleep(.1)
        self.chrono.stop()
        err = abs(self.chrono.time() - .1)
        self.assertTrue(err < .001)
    
        #waste time when stopped
        sleep(.5)

        #test time addition (second start) and not stopped time() call
        self.chrono.start()
        sleep(.2)
        err = abs(self.chrono.time() - .3)
        self.assertTrue(err < .001)
        
        #test second timing without stop 
        sleep(.25)
        err = abs(self.chrono.time() - .55)
        self.assertTrue(err < .002)
        
    def test_reset(self):
        self.chrono.start()
        sleep(0.1)
        self.chrono.reset()
        self.assertTrue(self.chrono.time()==0)
        
    def test_double_start_error(self):
        self.chrono.start()
        with self.assertRaises(ValueError):
            self.chrono.start()
        
    def test_double_stop_error(self):
        with self.assertRaises(ValueError):
            self.chrono.stop()
            
    def test_repr(self):
        self.assertTrue(repr(self.chrono.start()).endswith('(start)'))
        self.assertTrue(repr(self.chrono.stop()).endswith('(stop)'))