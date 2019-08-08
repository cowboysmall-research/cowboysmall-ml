import glob

import unittest


def create_suite():
    tests   = [test_file.split('/') for test_file in glob.glob('tests/**/test_*.py', recursive = True)]
    modules = ['.'.join(test)[:-3] for test in tests]
    suites  = [unittest.defaultTestLoader.loadTestsFromName(module) for module in modules]
    return unittest.TestSuite(suites)



if __name__ == '__main__':
   suite = create_suite()

   runner=unittest.TextTestRunner()
   runner.run(suite)