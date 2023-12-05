import os
import csv

def write_metrics_to_file(metrics_list, filename='metrics.csv'):
    if len(metrics_list)!=0:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = metrics_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for metrics in metrics_list:
                writer.writerow(metrics)
