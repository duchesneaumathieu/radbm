import os, shutil, unittest
from pathlib import Path
from radbm.utils.gdrive import available_files
from radbm.utils.fetch import (
    fetch_file,
    DATASETS_DIR,
    MODELS_DIR,
    PACKAGE_VAR,
)

class Test_fetch_file(unittest.TestCase):
    def setUp(self):
        Path('./testdump').mkdir(exist_ok=True)
        with open('./testdump/existing_data', 'w'):
            pass
        
    def tearDown(self):
        shutil.rmtree('./testdump')
    
    def test_fetch_file(self):
        _environ = os.environ
        try:
            os.environ[DATASETS_DIR] = '/dataset_dir'
            os.environ[MODELS_DIR] = '/model_dir'
            os.environ[PACKAGE_VAR] = '/package_dir'
            os.environ['HOME'] = '/home_dir'

            path = './testdump/'
            results = fetch_file('existing_data', path, data_type='dataset')
            expected = [os.path.abspath('./testdump/existing_data')]
            self.assertEqual(results, expected)

            self.assertTrue('mnist.pkl.gz' in available_files())
            results = fetch_file('mnist.pkl.gz', path, data_type='model', subdirs=['Mnist'])
            expected = [os.path.abspath('./testdump/Mnist/mnist.pkl.gz')]
            self.assertEqual(results, expected)
            self.assertTrue(os.path.isfile('./testdump/Mnist/mnist.pkl.gz'))
            
            with self.assertRaises(ValueError):
                results = fetch_file('existing_data', path, data_type='wrong', download=True)

            with self.assertRaises(FileNotFoundError):
                results = fetch_file('missing_data', path, data_type='dataset', download=True)

            with self.assertRaises(FileNotFoundError):
                results = fetch_file('missing_data', path, data_type='dataset', download=False)
        finally:
            os.environ.clear()
            os.environ.update(_environ)