#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for Pareto optimization and data processing.
"""

import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

# CSV column indices
CSV_TIME = 0                   # Timestamp
CSV_SPMV = 1                   # sPMV value
CSV_BASELINE = 2               # Baseline energy
CSV_FLEX_BELOW = 3             # Flexibility below
CSV_ENERGY_AVG = 4             # Energy average
CSV_INDOOR_T = 5               # Indoor temperature (t_r)
CSV_INDOOR_RH = 6              # Indoor relative humidity (rh_r)
CSV_HOUR = 7                   # Hour of day
CSV_DAY_OF_WEEK = 8            # Day of week
CSV_IS_WEEKEND = 9             # Weekend flag
CSV_FORECASTED_SPMV = 10       # Forecasted sPMV

# Tuple indices for optimization records
PARETO_OPT = 0                 # Pareto optimal solution flag
PROG = 1                       # Row index
DIST = 2                       # Temperature profile distance
KPI_ENERGY = 3                 # Energy in the period
KPI_FLEXIBILITY = 4            # Flexibility in the period
KPI_SPMV = 5                   # Comfort based on sPMV in the period
KPI_FORECASTED_SPMV = 6        # Forecasted comfort in the period
ENERGY_SAVINGS = 7             # Energy savings percentage
COMFORT_PERCENTAGE = 8         # Percentage of periods in comfort (based on sPMV)
ENERGY_VALUES = 9              # Tuple of K energy consumption values in the period
TEMP_VALUES = 10               # Tuple of K temperature values in the period
SPMV_VALUES = 11               # Tuple of K sPMV values in the period
RH_VALUES = 12                 # Tuple of K relative humidity values in the period

# Configuration parameters
K = 24                         # Number of prediction intervals (24 hours)
COMFORT_THRESHOLD = 0.5        # Threshold for acceptable thermal comfort

# Solution types
SOLUTION_TYPE_S1 = "S1"        # Optimal Comfort (99.00-100.00%)
SOLUTION_TYPE_S2 = "S2"        # Balanced Comfort-Energy (97.00-98.99%)
SOLUTION_TYPE_S3 = "S3"        # Energy Saving with Acceptable Comfort (95.00-96.99%)

@dataclass
class OptimalSolution:
    """Class for storing information about optimal solutions."""
    solution_type: str
    comfort_percentage: float
    energy_consumption: float
    avg_indoor_temp: float
    avg_indoor_humidity: float
    start_index: int
    temp_values: List[float]
    rh_values: List[float]
    energy_values: List[float]
    energy_savings: float
    day_of_week: str  # Day of week
    date: str  # Date of the solution


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the data.
    
    Args:
        file_path: Path to the data file.
        
    Returns:
        Preprocessed DataFrame.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Drop CO2 data column
    if 'co2_ppm' in df.columns:
        df = df.drop(columns=['co2_ppm'])
    
    return df


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data without splitting into historical and target.
    
    Args:
        df: DataFrame with all data.
        
    Returns:
        Processed DataFrame.
    """
    # Process all data without separating target day
    processed_data = df.copy()
    
    return processed_data


def read_csv_data(file_path: str, debug: bool = False) -> List[List[str]]:
    """
    Read data from a CSV file and store it as a list of rows.
    
    Args:
        file_path: Path to the CSV file.
        debug: Whether to print debug information.
        
    Returns:
        List of rows from the CSV file.
    """
    data_list = []
    
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        
        for row in reader:
            data_list.append(row)
    
    if debug:
        print(f"Read {len(data_list)} records from {file_path}")
        print(f"First row: {data_list[0]}")
    
    return data_list


def create_k_tuples(data_list: List[List[str]], k: int) -> List[Tuple]:
    """
    Create K-tuples from the input data, calculating averages for energy and comfort metrics.
    
    Args:
        data_list: List of data rows.
        k: Size of each tuple.
        
    Returns:
        List of K-tuples with calculated metrics.
    """
    new_list = []
    
    for i in range(len(data_list) - k + 1):
        # Calculate averages for the k-period window
        sum_energy = sum(float(data_list[j][CSV_ENERGY_AVG]) for j in range(i, i + k)) / k
        sum_baseline = sum(float(data_list[j][CSV_BASELINE]) for j in range(i, i + k)) / k
        sum_flexibility = sum(float(data_list[j][CSV_FLEX_BELOW]) for j in range(i, i + k)) / k
        sum_spmv = sum(abs(float(data_list[j][CSV_SPMV])) for j in range(i, i + k)) / k
        sum_forecasted_spmv = sum(abs(float(data_list[j][CSV_FORECASTED_SPMV])) for j in range(i, i + k)) / k
        
        # Collect values for the k-period window
        energy_values = [float(data_list[j][CSV_ENERGY_AVG]) for j in range(i, i + k)]
        temp_values = [float(data_list[j][CSV_INDOOR_T]) for j in range(i, i + k)]
        spmv_values = [float(data_list[j][CSV_SPMV]) for j in range(i, i + k)]
        rh_values = [float(data_list[j][CSV_INDOOR_RH]) for j in range(i, i + k)]
        
        # Create tuple with all metrics
        # Initial energy savings and comfort percentage are set to 0, will be calculated later
        merged_tuple = (
            '-',  # Not yet marked as Pareto optimal
            i,    # Starting index
            0,    # Distance (to be calculated)
            sum_energy,
            sum_flexibility,
            sum_spmv,
            sum_forecasted_spmv,
            0,    # Energy savings % (to be calculated)
            0,    # Comfort % (to be calculated)
            energy_values,
            temp_values,
            spmv_values,
            rh_values
        )
        
        new_list.append(merged_tuple)
    
    return new_list


def euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vector1: First vector.
        vector2: Second vector.
        
    Returns:
        Euclidean distance between the vectors.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))


def compute_distances_and_metrics(records: List[Tuple], reference_energy: List[float], 
                                 reference_temp: List[float], k: int) -> List[Tuple]:
    """
    Compute distance between temperature profiles and calculate energy savings and comfort metrics.
    
    Args:
        records: List of K-tuples.
        reference_energy: Reference energy values.
        reference_temp: Reference temperature values.
        k: Size of each tuple.
        
    Returns:
        Updated list of K-tuples with calculated distances and metrics.
    """
    updated_list = []
    
    # Calculate total reference energy for comparison
    energy_input = sum(reference_energy)
    
    for record in records:
        # Calculate Euclidean distance between temperature profiles
        distance = euclidean_distance(record[TEMP_VALUES], reference_temp)
        
        # Calculate energy savings percentage
        total_energy = sum(record[ENERGY_VALUES])
        energy_savings = (energy_input - total_energy) / energy_input * 100 if energy_input > 0 else 0
        
        # Calculate comfort percentage (percentage of periods within comfort threshold)
        comfort_count = sum(1 for spmv in record[SPMV_VALUES] if abs(spmv) <= COMFORT_THRESHOLD)
        comfort_percentage = comfort_count / k * 100
        
        # Create updated record with calculated metrics
        updated_record = (
            record[PARETO_OPT],
            record[PROG],
            distance,
            record[KPI_ENERGY],
            record[KPI_FLEXIBILITY],
            record[KPI_SPMV],
            record[KPI_FORECASTED_SPMV],
            energy_savings,
            comfort_percentage,
            record[ENERGY_VALUES],
            record[TEMP_VALUES],
            record[SPMV_VALUES],
            record[RH_VALUES]
        )
        
        updated_list.append(updated_record)
    
    return updated_list


def get_pareto_set(records: List[Tuple]) -> List[Tuple]:
    """
    Identify Pareto optimal solutions from the set of records.
    A solution is Pareto optimal if no other solution is strictly better in all objectives.
    
    Args:
        records: List of records to analyze.
        
    Returns:
        List of Pareto optimal solutions.
    """
    list_items = [list(t) for t in records]

    for item in list_items:
        item[PARETO_OPT] = '*'  # Initially mark all solutions as Pareto optimal
        
    # Iterate through all solutions to identify dominated ones
    for i, item1 in enumerate(list_items):
        for j, item2 in enumerate(list_items):
            if i != j:
                # Check if item1 is dominated by item2
                # For minimization objectives: DIST, KPI_ENERGY, KPI_SPMV, KPI_FORECASTED_SPMV
                # Lower values are better
                if (item1[DIST] >= item2[DIST] and
                    item1[KPI_ENERGY] >= item2[KPI_ENERGY] and
                    item1[KPI_SPMV] >= item2[KPI_SPMV] and
                    item1[KPI_FORECASTED_SPMV] >= item2[KPI_FORECASTED_SPMV] and
                    # At least one objective must be strictly better
                    (item1[DIST] > item2[DIST] or
                     item1[KPI_ENERGY] > item2[KPI_ENERGY] or
                     item1[KPI_SPMV] > item2[KPI_SPMV] or
                     item1[KPI_FORECASTED_SPMV] > item2[KPI_FORECASTED_SPMV])):
                    # Mark as dominated (not Pareto optimal)
                    item1[PARETO_OPT] = ' '
                    break
    
    # Convert back to tuples
    updated_records = [tuple(item) for item in list_items]
    
    # Filter to return only Pareto optimal solutions
    pareto_optimal = [record for record in updated_records if record[PARETO_OPT] == '*']
    
    return pareto_optimal


def filter_solutions(solutions: List[Tuple], solution_type: str) -> List[Tuple]:
    """
    Filter solutions based on the solution type.
    
    Args:
        solutions: List of all solutions.
        solution_type: Type of solution (S1, S2, S3).
        
    Returns:
        List of filtered solutions.
    """
    filtered = []
    
    if solution_type == SOLUTION_TYPE_S1:
        # S1: Optimal Comfort (99.00-100.00%)
        filtered = [sol for sol in solutions if sol[COMFORT_PERCENTAGE] >= 99.0]
    elif solution_type == SOLUTION_TYPE_S2:
        # S2: Balanced Comfort-Energy (97.00-98.99%)
        filtered = [sol for sol in solutions if 97.0 <= sol[COMFORT_PERCENTAGE] < 99.0]
    elif solution_type == SOLUTION_TYPE_S3:
        # S3: Energy Saving with Acceptable Comfort (95.00-96.99%)
        filtered = [sol for sol in solutions if 95.0 <= sol[COMFORT_PERCENTAGE] < 97.0]
    
    return filtered


def sort_solutions(filtered_solutions: List[Tuple], solution_type: str) -> List[Tuple]:
    """
    Sort solutions based on the solution type to find the best candidates.
    
    Args:
        filtered_solutions: Filtered solutions to sort.
        solution_type: Type of solution to sort for (S1, S2, or S3).
        
    Returns:
        Sorted list of solutions with the best candidate at index 0.
    """
    if solution_type == SOLUTION_TYPE_S1:
        # For S1, prioritize comfort (lowest forecasted sPMV)
        sorted_solutions = sorted(filtered_solutions, key=lambda x: x[KPI_FORECASTED_SPMV])
    elif solution_type == SOLUTION_TYPE_S2:
        # For S2, balance comfort and energy
        # Define a custom sorting key that gives equal weight to comfort and energy
        sorted_solutions = sorted(filtered_solutions, 
                                 key=lambda x: (x[KPI_FORECASTED_SPMV] + x[KPI_ENERGY]) / 2)
    elif solution_type == SOLUTION_TYPE_S3:
        # For S3, prioritize energy efficiency while maintaining acceptable comfort
        sorted_solutions = sorted(filtered_solutions, key=lambda x: x[KPI_ENERGY])
    else:
        sorted_solutions = filtered_solutions
    
    return sorted_solutions


def select_best_solution(filtered_solutions: List[Tuple], solution_type: str, 
                         data_list: List[List[str]]) -> Optional[OptimalSolution]:
    """
    Select the best solution from filtered solutions based on solution type.
    
    Args:
        filtered_solutions: List of filtered solutions.
        solution_type: Type of solution (S1, S2, or S3).
        data_list: Original data list to extract day information.
        
    Returns:
        Best OptimalSolution or None if no solutions found.
    """
    if not filtered_solutions:
        return None
    
    # Sort solutions based on solution type
    sorted_solutions = sort_solutions(filtered_solutions, solution_type)
    
    # Select the best solution (first after sorting)
    best_solution = sorted_solutions[0]
    
    # Create OptimalSolution object
    return create_optimal_solution(best_solution, solution_type, data_list)


def create_optimal_solution(record: Tuple, solution_type: str, data_list: List[List[str]]) -> OptimalSolution:
    """
    Create an OptimalSolution object from a record.
    
    Args:
        record: Record to create optimal solution from.
        solution_type: Type of solution (S1, S2, or S3).
        data_list: Original data list to extract day information.
        
    Returns:
        OptimalSolution object.
    """
    start_index = record[PROG]
    
    # Calculate average temperature and humidity
    avg_temp = sum(record[TEMP_VALUES]) / len(record[TEMP_VALUES])
    avg_rh = sum(record[RH_VALUES]) / len(record[RH_VALUES])
    
    # Calculate total energy consumption
    energy_consumption = sum(record[ENERGY_VALUES])
    
    # Extract day of week and date information
    day_of_week = data_list[start_index][CSV_DAY_OF_WEEK]
    date = data_list[start_index][CSV_TIME].split()[0]
    
    # Map day of week number to name
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_name = day_names[int(day_of_week) % 7]
    
    return OptimalSolution(
        solution_type=solution_type,
        comfort_percentage=record[COMFORT_PERCENTAGE],
        energy_consumption=energy_consumption,
        avg_indoor_temp=avg_temp,
        avg_indoor_humidity=avg_rh,
        start_index=start_index,
        temp_values=list(record[TEMP_VALUES]),
        rh_values=list(record[RH_VALUES]),
        energy_values=list(record[ENERGY_VALUES]),
        energy_savings=record[ENERGY_SAVINGS],
        day_of_week=day_name,
        date=date
    )


def create_artificial_solution(s1_solution: OptimalSolution, s3_solution: OptimalSolution) -> OptimalSolution:
    """
    Create an artificial solution as a weighted blend of S1 and S3 solutions.
    
    Args:
        s1_solution: Optimal comfort solution (S1)
        s3_solution: Optimal energy solution (S3)
        
    Returns:
        An artificial balanced solution (S2)
    """
    # Weight for blending (0.6 weight to comfort, 0.4 to energy)
    w_comfort = 0.6
    w_energy = 0.4
    
    # Create blended temperature and humidity profiles
    temp_values = [w_comfort*t1 + w_energy*t3 for t1, t3 in zip(s1_solution.temp_values, s3_solution.temp_values)]
    rh_values = [w_comfort*h1 + w_energy*h3 for h1, h3 in zip(s1_solution.rh_values, s3_solution.rh_values)]
    energy_values = [w_comfort*e1 + w_energy*e3 for e1, e3 in zip(s1_solution.energy_values, s3_solution.energy_values)]
    
    # Calculate averages
    avg_temp = sum(temp_values) / len(temp_values)
    avg_rh = sum(rh_values) / len(rh_values)
    
    # Calculate energy consumption and savings
    energy_consumption = sum(energy_values)
    energy_savings = w_comfort * s1_solution.energy_savings + w_energy * s3_solution.energy_savings
    
    # Calculate comfort percentage
    comfort_percentage = w_comfort * s1_solution.comfort_percentage + w_energy * s3_solution.comfort_percentage
    
    return OptimalSolution(
        solution_type=SOLUTION_TYPE_S2,
        comfort_percentage=comfort_percentage,
        energy_consumption=energy_consumption,
        avg_indoor_temp=avg_temp,
        avg_indoor_humidity=avg_rh,
        start_index=s1_solution.start_index,  # Use S1's start index
        temp_values=temp_values,
        rh_values=rh_values,
        energy_values=energy_values,
        energy_savings=energy_savings,
        day_of_week=s1_solution.day_of_week,  # Use S1's day of week
        date=s1_solution.date  # Use S1's date
    )


def plot_solution(solution: OptimalSolution, output_dir: str = 'plots') -> None:
    """
    Generate and save visualizations for the optimal solution.
    
    Args:
        solution: OptimalSolution to plot.
        output_dir: Directory to save plots.
    """
    import os
    import matplotlib.pyplot as plt
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create x-axis values for time series plots
    hours = list(range(24))
    
    # Temperature plot
    plt.figure(figsize=(10, 6))
    plt.plot(hours, solution.temp_values, 'b-', linewidth=2, label='Optimal Temperature')
    plt.xlabel('Hour of Day')
    plt.ylabel('Temperature (°C)')
    plt.title(f'{solution.solution_type} - Temperature Profile - {solution.date}')
    plt.legend()
    plt.grid(True)
    plt.xticks(hours)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{solution.solution_type}_temperature.png'))
    plt.close()
    
    # Plot humidity profile
    plt.figure(figsize=(10, 6))
    plt.plot(hours, solution.rh_values, 'b-', linewidth=2, label='Optimal Humidity')
    plt.xlabel('Hour of Day')
    plt.ylabel('Relative Humidity (%)')
    plt.title(f'{solution.solution_type} - Humidity Profile - {solution.date}')
    plt.legend()
    plt.grid(True)
    plt.xticks(hours)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{solution.solution_type}_humidity.png'))
    plt.close()
    
    # Energy plot
    plt.figure(figsize=(10, 6))
    plt.plot(hours, solution.energy_values, 'g-', linewidth=2, label='Optimal Energy')
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy (Wh)')
    plt.title(f'{solution.solution_type} - Energy Profile - {solution.date}')
    plt.legend()
    plt.grid(True)
    plt.xticks(hours)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{solution.solution_type}_energy.png'))
    plt.close()


def plot_solutions_comparison(solutions: List[OptimalSolution], output_dir: str = 'plots') -> None:
    """
    Generate and save comparison visualizations for all solutions.
    
    Args:
        solutions: List of solutions to compare.
        output_dir: Directory to save plots.
    """
    import os
    import matplotlib.pyplot as plt
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create x-axis values for time series plots
    hours = list(range(24))
    
    # Temperature comparison
    plt.figure(figsize=(12, 8))
    
    # Plot each solution's temperature
    for solution in solutions:
        plt.plot(hours, solution.temp_values, linewidth=2, label=f'{solution.solution_type}')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Profiles Comparison')
    plt.legend()
    plt.grid(True)
    plt.xticks(hours)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_temperature.png'))
    plt.close()
    
    # Energy comparison
    plt.figure(figsize=(12, 8))
    
    # Plot each solution's energy
    for solution in solutions:
        plt.plot(hours, solution.energy_values, linewidth=2, label=f'{solution.solution_type}')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy (Wh)')
    plt.title('Energy Profiles Comparison')
    plt.legend()
    plt.grid(True)
    plt.xticks(hours)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'energy_comparison.png'))
    plt.close()
    
    # Create bar chart comparing key metrics
    metrics = ['Comfort (%)', 'Energy (kWh)', 'Avg. Temp (°C)', 'Avg. RH (%)']
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(metrics))
    width = 0.2
    multiplier = 0
    
    for solution in solutions:
        # Scale energy to kWh for better visualization
        energy_kwh = solution.energy_consumption / 1000
        
        values = [solution.comfort_percentage, 
                 energy_kwh, 
                 solution.avg_indoor_temp, 
                 solution.avg_indoor_humidity]
        
        offset = width * multiplier
        plt.bar(x + offset, values, width, label=solution.solution_type)
        multiplier += 1
    
    # Add a bar for the target data regardless of length
    
    
    offset = width * multiplier
    plt.bar(x + offset, width, label='Target')
    
    plt.ylabel('Value')
    plt.title('Comparison of Key Metrics')
    plt.xticks(x + width, metrics)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()
