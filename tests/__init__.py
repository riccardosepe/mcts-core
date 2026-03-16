import unittest

def run_all_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=".", pattern="test*.py")
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == "__main__":
    run_all_tests()