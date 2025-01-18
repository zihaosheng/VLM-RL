import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_eval(eval_csv_paths, output_name=None):
    episode_numbers = pd.read_csv(eval_csv_paths[0])['episode'].unique()

    # Get a list of unique episode numbers
    cols = ['Steer', 'Throttle', 'Speed (km/h)', 'Reward', 'Center Deviation (m)', 'Distance (m)',
            'Angle next waypoint (grad)', 'Trayectory']

    # Create a figure with subplots for each episode
    fig, axs = plt.subplots(len(episode_numbers), len(cols), figsize=(4 * len(cols), 3 * len(episode_numbers)))

    if len(eval_csv_paths) == 1:
        eval_plot_path = eval_csv_paths[0].replace(".csv", ".png")
    else:
        os.makedirs('tensorboard/eval_plots', exist_ok=True)
        eval_plot_path = f'./tensorboard/eval_plots/{output_name}'

    models = ['Waypoints']

    # Load the dataframe
    for e, path in enumerate(eval_csv_paths):
        df = pd.read_csv(path)
        model_id = df.loc[df['model_id'] != 'route', 'model_id'].unique()[0]
        models.append(model_id)
        # Loop over each episode number
        for i, episode_number in enumerate(episode_numbers):
            # Select the rows for the current episode
            episode_df = df[(df['episode'] == episode_number) & (df['model_id'] != 'route')]
            route_df = df[(df['episode'] == episode_number) & (df['model_id'] == 'route')]

            # Plot the steer progress
            axs[i, 0].plot(episode_df['step'], episode_df['steer'], label=model_id)
            axs[i, 0].set_xlabel('Step')
            axs[i, 0].set_ylim(-1, 1)  # clip y-axis limits to -1 and 1

            # Plot the throttle progress
            axs[i][1].plot(episode_df['step'], episode_df['throttle'], label=model_id)
            axs[i][1].set_xlabel('Step')
            axs[i, 1].set_ylim(0, 1)  # clip y-axis limits to -1 and 1

            axs[i][2].plot(episode_df['step'], episode_df['speed'], label=model_id)
            axs[i][2].set_xlabel('Step')
            axs[i, 2].set_ylim(0, 40)  # clip y-axis limits to -1 and 1

            # Plot the reward progress
            axs[i][3].plot(episode_df['step'], episode_df['reward'], label=model_id)
            axs[i][3].set_xlabel('Step')
            axs[i, 3].set_ylim(-0.2, 1)  # clip y-axis limits to -1 and 1

            axs[i][4].plot(episode_df['step'], episode_df['center_dev'], label=model_id)
            axs[i][4].set_xlabel('Step')
            axs[i, 4].set_ylim(0, 3)  # clip y-axis limits to -1 and 1

            axs[i][5].plot(episode_df['step'], episode_df['distance'], label=model_id)
            axs[i][5].set_xlabel('Step')

            axs[i][6].plot(episode_df['step'], episode_df['angle_next_waypoint'], label=model_id)
            axs[i][6].set_xlabel('Step')

            if e == 0:
                axs[i][7].plot(route_df['route_x'].head(1), route_df['route_y'].head(1), 'go',
                               label='Start')
                axs[i][7].plot(route_df['route_x'].tail(1), route_df['route_y'].tail(1), 'ro',
                               label='End')
                axs[i][7].plot(route_df['route_x'], route_df['route_y'], label='Waypoints', color="green")

                axs[i, 7].set_xlim(left=min(-5, min(route_df['route_x'] - 3)))
                axs[i, 7].set_xlim(right=max(5, max(route_df['route_x'] + 3)))
            axs[i][7].plot(episode_df['vehicle_location_x'], episode_df['vehicle_location_y'], label=model_id)

    # Add legend

    pad = 5  # in points
    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axs[:, 0], episode_numbers):
        ax.annotate(f"Episode {row}", xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    # Adjust the spacing between subplots
    # fig.subplots_adjust(bottom=0.062*len(labels))

    handles, labels = axs[0][7].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.1 + 0.02 * len(labels), 1, 1))

    # Adjust the bottom margin to make room for the legend
    # Show the plot
    plt.savefig(eval_plot_path)


def summary_eval(eval_csv_path):
    df = pd.read_csv(eval_csv_path)
    df_route = df[df['model_id'] == 'route']
    df = df[df['model_id'] != 'route']
    df = df.drop(['model_id', 'route_x', 'route_y'], axis=1)

    # Get the total distance traveled from each episode based on last row
    df_distance = df.groupby(['episode'], as_index=False).last()[['episode', 'distance']].rename(
        columns={'distance': 'total_distance'})

    # Get the total reward from each episode summing all the rewards
    df_reward = df.groupby(['episode'], as_index=False).sum()[['episode', 'reward']].rename(columns={'reward': 'total_reward'})

    # Get the mean and std from: speed, center_dev and reward
    df_mean_std = df.groupby(['episode'], as_index=False).agg(
        {'speed': ['mean', 'std'], 'center_dev': ['mean', 'std'], 'reward': ['mean', 'std']})
    df_mean_std.columns = ['episode', 'speed_mean', 'speed_std', 'center_dev_mean', 'center_dev_std', 'reward_mean', 'reward_std']

    # Get the RC, CS, "collision_interval", "CPS", "CPM" from each episode based on last row
    df_routes_completed = df.groupby(['episode'], as_index=False).last()[['episode', 'routes_completed']]
    df_routes_completed['routes_completed'] = df_routes_completed['routes_completed'].clip(upper=1)

    df_collision_speed = df.groupby(['episode'], as_index=False).last()[['episode', 'collision_speed']]
    df_collision_interval = df.groupby(['episode'], as_index=False).last()[['episode', 'collision_interval']]
    df_CPS = df.groupby(['episode'], as_index=False).last()[['episode', 'CPS']]
    df_CPM = df.groupby(['episode'], as_index=False).last()[['episode', 'CPM']]

    # Calculate if the episode was successful based on the distance between waypoints and the vehicle of the last row

    # First from the route dataframe get the last row of each episode
    df_waypoint = df_route.groupby(['episode'], as_index=False).last()[['episode', 'route_x', 'route_y']]
    df_success = df.groupby(['episode'], as_index=False).last()[['episode', 'vehicle_location_x', 'vehicle_location_y']]
    df_success = pd.merge(df_success, df_waypoint, on='episode')

    # If the distance between the last waypoint and the vehicle is less than 5 meters, the episode was successful
    df_success['success'] = df_success.apply(
        lambda x: eucldist(x['vehicle_location_x'], x['vehicle_location_y'], x['route_x'], x['route_y']) < 5, axis=1)
    df_success = df_success[['episode', 'success']]
    # Merge all the dataframes
    df_summary = pd.merge(df_distance, df_reward, on='episode')
    df_summary = pd.merge(df_summary, df_routes_completed, on='episode')
    df_summary = pd.merge(df_summary, df_collision_speed, on='episode')
    df_summary = pd.merge(df_summary, df_collision_interval, on='episode')
    df_summary = pd.merge(df_summary, df_CPS, on='episode')
    df_summary = pd.merge(df_summary, df_CPM, on='episode')

    df_summary = pd.merge(df_summary, df_mean_std, on='episode')
    df_summary = pd.merge(df_summary, df_success, on='episode')

    # Turn the episode column into a string
    df_summary['episode'] = df_summary['episode'].astype(str)

    # Create a new row called where the episode is total with the mean of all the columns except total reward and total distance without modifying the index
    df_summary.loc['total'] = df_summary.mean(numeric_only=True)
    df_summary.loc['total', 'episode'] = 'total'
    df_summary.loc['total', 'total_reward'] = df_summary['total_reward'].iloc[:-1].sum()
    df_summary.loc['total', 'total_distance'] = df_summary['total_distance'].iloc[:-1].sum()

    output_path = eval_csv_path.replace("eval.csv", "eval_summary.csv")
    df_summary.to_csv(output_path, index=False)
    print(f"Saving summary to {output_path}")


def eucldist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results from different models")
    parser.add_argument("--models", nargs='+', type=str, default="", help="Path to a model evaluate")
    args = vars(parser.parse_args())

    compare_models = args['models']
    eval_csv_paths = []
    for model in compare_models:
        model_id, steps = model.split("-")
        eval_csv_paths.append(os.path.join("tensorboard", model_id, "eval", f"model_{steps}_steps_eval.csv"))
    plot_eval(eval_csv_paths, output_name="+".join(compare_models))


if __name__ == '__main__':
    main()
