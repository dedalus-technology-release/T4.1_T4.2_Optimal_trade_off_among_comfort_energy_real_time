#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pareto Optimization Runner (No Target Version)

This script implements a multi-objective optimization approach to determine the optimal
balance between indoor thermal comfort and energy consumption,
using historical data to generate three solution strategies:

S1: Optimal Comfort - prioritizes thermal comfort
S2: Balanced Comfort-Energy - balances comfort and energy efficiency
S3: Acceptable Comfort with Optimal Energy - prioritizes energy savings while maintaining acceptable comfort

The script loads data, performs Pareto optimization, and generates visualizations for the three solutions.
This version does not use target data for comparisons.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import csv
import math

# Add NewScinario to the system path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import utility functions
from scripts.optimization.utils import (
    # Constants
    K, COMFORT_THRESHOLD, SOLUTION_TYPE_S1, SOLUTION_TYPE_S2, SOLUTION_TYPE_S3,
    # Classes
    OptimalSolution,
    # Functions
    load_data, process_data, read_csv_data, create_k_tuples,
    compute_distances_and_metrics, get_pareto_set, filter_solutions,
    select_best_solution, create_artificial_solution,
    plot_solution, plot_solutions_comparison
)


def prepare_data_for_optimization(data: pd.DataFrame) -> Tuple:
    """
    Prepare data for optimization by converting to list format and creating reference values.
    
    Args:
        data: Processed data.
        
    Returns:
        Tuple containing data list, reference temperature and energy values.
    """
    # Convert DataFrame to list of lists for compatibility with utility functions
    data_list = data.values.tolist()
    
    # Add column names as strings to match the expected format
    for i, row in enumerate(data_list):
        # Convert timestamp to string format if it's a datetime object
        if isinstance(row[0], pd.Timestamp):
            data_list[i][0] = str(row[0])
    
    # Create reference temperature and energy values
    # Since we don't have target data, we'll use averages from the last 24 hours of data
    # as reference to maintain compatibility with the original function
    last_24_hours = data.tail(24)
    reference_temp = last_24_hours['t_r'].tolist()
    reference_energy = last_24_hours['energy_average'].tolist()
    
    # Ensure we have exactly 24 values (pad if necessary)
    while len(reference_temp) < 24:
        reference_temp.append(reference_temp[-1] if reference_temp else 21.0)
    while len(reference_energy) < 24:
        reference_energy.append(reference_energy[-1] if reference_energy else 100.0)
    
    return data_list, reference_temp, reference_energy


def optimize(data_list: List[List], reference_temp: List[float], 
             reference_energy: List[float]) -> Tuple[Optional[OptimalSolution], 
                                                    Optional[OptimalSolution], 
                                                    Optional[OptimalSolution]]:
    """
    Optimize the data to find the best solutions for S1, S2, and S3.
    
    Args:
        data_list: List of data rows.
        reference_temp: Reference temperature values.
        reference_energy: Reference energy values.
        
    Returns:
        Tuple of OptimalSolution objects (S1, S2, S3).
    """
    print("Starting optimization process...")
    
    # Create K-tuples from the data
    records = create_k_tuples(data_list, K)
    print(f"Created {len(records)} K-tuples from data")
    
    # Compute distances and metrics
    records = compute_distances_and_metrics(records, reference_energy, reference_temp, K)
    print("Computed distances and metrics for all records")
    
    # Get Pareto optimal set
    pareto_set = get_pareto_set(records)
    print(f"Found {len(pareto_set)} Pareto optimal solutions")
    
    # Find solutions for each strategy
    s1_filtered = filter_solutions(pareto_set, SOLUTION_TYPE_S1)
    print(f"Found {len(s1_filtered)} candidates for S1 (Optimal Comfort)")
    
    s2_filtered = filter_solutions(pareto_set, SOLUTION_TYPE_S2)
    print(f"Found {len(s2_filtered)} candidates for S2 (Balanced)")
    
    s3_filtered = filter_solutions(pareto_set, SOLUTION_TYPE_S3)
    print(f"Found {len(s3_filtered)} candidates for S3 (Energy Saving)")
    
    # Select best solutions
    s1_solution = select_best_solution(s1_filtered, SOLUTION_TYPE_S1, data_list)
    s3_solution = select_best_solution(s3_filtered, SOLUTION_TYPE_S3, data_list)
    
    # For S2, either use best from filtered set or create artificial solution
    if s2_filtered:
        s2_solution = select_best_solution(s2_filtered, SOLUTION_TYPE_S2, data_list)
        print("Selected best S2 solution from Pareto set")
    elif s1_solution and s3_solution:
        # Create artificial solution as fallback
        s2_solution = create_artificial_solution(s1_solution, s3_solution)
        print("Created artificial S2 solution as weighted average of S1 and S3")
    else:
        s2_solution = None
        print("Could not create S2 solution")
    
    # Print results
    print("\nOptimization Results:")
    if s1_solution:
        print(f"\nS1 - Optimal Comfort:")
        print(f"  Comfort Percentage: {s1_solution.comfort_percentage:.2f}%")
        print(f"  Energy Consumption: {s1_solution.energy_consumption:.2f} Wh")
        print(f"  Average Temperature: {s1_solution.avg_indoor_temp:.2f} C")
        print(f"  Average Humidity: {s1_solution.avg_indoor_humidity:.2f}%")
    
    if s2_solution:
        print(f"\nS2 - Balanced:")
        print(f"  Comfort Percentage: {s2_solution.comfort_percentage:.2f}%")
        print(f"  Energy Consumption: {s2_solution.energy_consumption:.2f} Wh")
        print(f"  Average Temperature: {s2_solution.avg_indoor_temp:.2f} C")
        print(f"  Average Humidity: {s2_solution.avg_indoor_humidity:.2f}%")
    
    if s3_solution:
        print(f"\nS3 - Energy Saving:")
        print(f"  Comfort Percentage: {s3_solution.comfort_percentage:.2f}%")
        print(f"  Energy Consumption: {s3_solution.energy_consumption:.2f} Wh")
        print(f"  Average Temperature: {s3_solution.avg_indoor_temp:.2f} C")
        print(f"  Average Humidity: {s3_solution.avg_indoor_humidity:.2f}%")
    
    return s1_solution, s2_solution, s3_solution


def save_solutions_to_csv(solutions: List[OptimalSolution], output_dir: str) -> None:
    """
    Save solutions to CSV files.
    
    Args:
        solutions: List of solutions to save.
        output_dir: Directory to save CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set hours to 24
    hours = 24
    
    # Save summary CSV
    summary_path = os.path.join(output_dir, "solutions_summary.csv")
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Solution", "Comfort (%)", "Energy (Wh)", 
            "Avg Temp (C)", "Avg RH (%)", "Energy Savings (%)"
        ])
        
        for sol in solutions:
            writer.writerow([
                sol.solution_type, 
                f"{sol.comfort_percentage:.2f}", 
                f"{sol.energy_consumption:.2f}",
                f"{sol.avg_indoor_temp:.2f}", 
                f"{sol.avg_indoor_humidity:.2f}",
                f"{sol.energy_savings:.2f}"
            ])
    
    # Save detailed CSV for each solution
    for sol in solutions:
        detail_path = os.path.join(output_dir, f"{sol.solution_type}_details.csv")
        with open(detail_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Hour", "Temperature (C)", "Humidity (%)", "Energy (Wh)"
            ])
            
            for i in range(24):
                row = [i, f"{sol.temp_values[i]:.2f}", f"{sol.rh_values[i]:.2f}", f"{sol.energy_values[i]:.2f}"]
                writer.writerow(row)
    
    print(f"\nSolution summary saved to: {summary_path}")
    print(f"Detailed solution data saved to: {output_dir}")


def run_pareto_optimization(input_file: str, output_dir: str) -> None:
    
    print("Starting Pareto Optimization...")
    
    # Set up file paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = input_file
    
    output_base_dir = os.path.join(output_dir, "optimization_results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #output_dir = os.path.join(output_base_dir, timestamp)
    os.makedirs(output_base_dir, exist_ok=True)
    
    #plot_dir = os.path.join(output_dir, "plots")
    #os.makedirs(plot_dir, exist_ok=True)
    
    csv_dir = output_base_dir
    os.makedirs(csv_dir, exist_ok=True)
    
    # Load and preprocess data
    print(f"Loading data from {data_path}")
    df = load_data(data_path)
    print(f"Loaded {len(df)} records, columns: {df.columns.tolist()}")
    
    # Process data without splitting
    processed_data = process_data(df)
    print(f"Processed {len(processed_data)} data records")
    
    # Prepare data for optimization
    data_list, reference_temp, reference_energy = prepare_data_for_optimization(processed_data)
    
    # Run optimization
    s1_solution, s2_solution, s3_solution = optimize(data_list, reference_temp, reference_energy)
    
    # Collect valid solutions
    solutions = [sol for sol in [s1_solution, s2_solution, s3_solution] if sol is not None]
    
    if solutions:
        # Generate comparison plots only (no individual solution plots)
        #plot_solutions_comparison(solutions, plot_dir)
        
        # Save solutions to CSV
        save_solutions_to_csv(solutions, csv_dir)
        
        print("\nOptimization completed successfully!")
        print(f"Output directory: {output_dir}")
        #print(f"Plots saved to: {plot_dir}")
        print(f"CSV files saved to: {csv_dir}")
        
        # Print energy savings in log
        print("\nEnergy Savings Summary:")
        for solution in solutions:
            print(f"{solution.solution_type} - Energy Savings: {solution.energy_savings:.2f}%")
    else:
        print("No valid solutions found. Please check your data and parameters.")
