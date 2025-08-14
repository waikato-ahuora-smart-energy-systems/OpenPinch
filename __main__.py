from .src import main

if __name__ == "__main__":
    filenames = main.get_test_filenames()
    for filename in filenames:
        main.run_pinch_analysis_comparison(filename)    
