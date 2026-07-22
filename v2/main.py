# main.py
"""
Main script to run peak prediction analysis
"""

import polars as pl
from datetime import datetime, timedelta
import os
from config import (
    DATA_SOURCES, PEAK_PREDICT_PARAMS, INIT_PEAK_FINDER_PARAMS,
    TIMEZONE, OUTPUT_DIR, OUTPUT_CSV_PEAK_PREDICT, OUTPUT_CSV_INIT_PEAK
)
from init_peak_finder import InitPeakFinder
from peak_predictor import PeakPredictor
from helper import fetch_df
from ice_data_py import cds


class PeakAnalysisRunner:
    """Main runner for peak analysis"""
    
    def __init__(self):
        """
        Initialize runner
        """
        self.cds = cds
        self.init_peak_finder = InitPeakFinder(
            alpha=INIT_PEAK_FINDER_PARAMS['default_alpha'],
            lookback_years=INIT_PEAK_FINDER_PARAMS['lookback_years']
        )
        self.peak_predictor = PeakPredictor(
            zone=PEAK_PREDICT_PARAMS['zone'],
            threshold_mw=PEAK_PREDICT_PARAMS['threshold_mw'],
            derate_7day_forecast=PEAK_PREDICT_PARAMS['derate_7day_forecast']
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    def run_init_peak_finder(self, t1, t2):
        """
        Run initial peak finder for a date range
        
        Args:
            t1: start datetime
            t2: end datetime
        
        Returns:
            Polars DataFrame with results
        """
        print(f"Running InitPeakFinder from {t1} to {t2}")
        
        results = []
        current_date = t1.date() if hasattr(t1, 'date') else t1
        end_date = t2.date() if hasattr(t2, 'date') else t2
        
        while current_date <= end_date:
            current_datetime = datetime(
                current_date.year, current_date.month, current_date.day, 0, 0, 0
            )
            
            print(f"Processing date: {current_date}")
            
            result = self.init_peak_finder.run(DATA_SOURCES, current_datetime, self.cds)
            
            results.append({
                'date': current_date,
                'timestamp': result['timestamp'],
                'month': result['month'],
                'n95': result['n95'],
                'current_year_max': result['current_year_max'],
                'historical_min': result['historical_min'],
                'init_peak': result['init_peak']
            })
            
            current_date += timedelta(days=1)
        
        results_df = pl.DataFrame(results)
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_INIT_PEAK)
        results_df.write_csv(output_path)
        print(f"Saved init peak finder results to {output_path}")
        
        return results_df
    
    def run_peak_predictor(self, t1, t2):
        """
        Run peak predictor for a date range
        
        Args:
            t1: start datetime
            t2: end datetime
        
        Returns:
            Polars DataFrame with predictions
        """
        print(f"Running PeakPredictor from {t1} to {t2}")
        
        results = []
        current_date = t1.date() if hasattr(t1, 'date') else t1
        end_date = t2.date() if hasattr(t2, 'date') else t2
        
        while current_date <= end_date:
            current_datetime = datetime(
                current_date.year, current_date.month, current_date.day, 0, 0, 0
            )
            
            print(f"Processing date: {current_date}")
            
            df = self.peak_predictor.run(DATA_SOURCES, current_datetime, self.cds)
            
            if not df.is_empty():
                results.append(df)
            
            current_date += timedelta(days=1)
        
        if results:
            results_df = pl.concat(results)
        else:
            results_df = pl.DataFrame()
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_PEAK_PREDICT)
        results_df.write_csv(output_path)
        print(f"Saved peak predictor results to {output_path}")
        
        return results_df
    
    def fetch_and_save_load_data(self, t1, t2):
        """
        Fetch and save load_mw data
        
        Args:
            t1: start datetime
            t2: end datetime
        """
        print(f"Fetching load_mw data from {t1} to {t2}")
        
        load_data_source = {
            "load_mw": DATA_SOURCES["load_mw"]
        }
        
        df = fetch_df(load_data_source, t1, t2, self.cds)
        
        if not df.is_empty():
            # Select only relevant columns
            df = df.select(['datetime', 'load_mw', 'month', 'year'])
            
            # Save to CSV
            output_path = os.path.join(OUTPUT_DIR, "load_data.csv")
            df.write_csv(output_path)
            print(f"Saved load data to {output_path}")
        else:
            print("No load data fetched")
    
    def run_all(self, t1, t2):
        """
        Run all analyses
        
        Args:
            t1: start datetime
            t2: end datetime
        """
        print("=" * 80)
        print("PEAK ELECTRICAL USAGE ANALYSIS")
        print("=" * 80)
        
        # Fetch and save load data
        self.fetch_and_save_load_data(t1, t2)
        
        init_peak_results = self.run_init_peak_finder(t1, t2)
        print("\nInitial Peak Finder Results:")
        print(init_peak_results)
        
        print("\n" + "=" * 80 + "\n")
        
        peak_predict_results = self.run_peak_predictor(t1, t2)
        print("\nPeak Predictor Results:")
        print(peak_predict_results)
        
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print(f"Results saved to {OUTPUT_DIR}/")


# Example usage
if __name__ == "__main__":
    # Initialize runner
    runner = PeakAnalysisRunner()
    
    # Define date range - using 2024 data
    t1 = datetime(2024, 1, 1)
    t2 = datetime(2024, 2, 28)
    
    # Run analysis
    runner.run_all(t1, t2)
