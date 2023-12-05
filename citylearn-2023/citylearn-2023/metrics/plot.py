import csv
import matplotlib.pyplot as plt
import ast

def read_metrics_from_file(filename='E:\TAMU\Courses\Fall 2023\CSCE 642 600 Deep Reinforcement Learning\project\citylearn-2023-starter-kit-master\citylearn-2023-starter-kit-master\ppo_agent_metrics.csv'):
    metrics_list = []

    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            metrics_list.append(row)

    return metrics_list

def plot_graph(metrics_list, metric_name):
    episode_count = []
    average_scores = []

    window_size = 1
    total_score = 0

    for i, row in enumerate(metrics_list):
        try:
            score_dict = ast.literal_eval(row[metric_name])
            value = score_dict['value']
            total_score += float(value)

            if (i + 1) % window_size == 0:
                episode_count.append(i + 1)
                average_scores.append(total_score / window_size)
                total_score = 0

        except (ValueError, KeyError, SyntaxError) as e:
            print(f"Skipping row {i + 1} with invalid 'average_score' value: {row['average_score']} ({e})")

    plt.plot(episode_count, average_scores, marker='o')
    plt.xlabel('Episode Count')
    plt.ylabel('Users’ comfort factor')
    # plt.title(f'Average Score (Every {window_size} Episodes) vs Episode Count')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    metrics_list = read_metrics_from_file()
    # plot_graph(metrics_list, 'average_score')
    # plot_graph(metrics_list, 'carbon_emissions_total')
    plot_graph(metrics_list, 'discomfort_proportion')
