import os
import matplotlib.pyplot as plt
import numpy as np
from cahp.data_types.decorators import verbose_decorator

@verbose_decorator
def plot_mss_curve(k_values, mss_values, knee=None, folder_name="MSS(k)", file_name="MSS vs k"):
    """
    Visualizes the Mean Sample Silhouette (MSS) values across different numbers of clusters (k).

    Args:
        k_values (list[int]): A list of integers representing the range of cluster 
                              counts (number of medoids) evaluated.
        mss_values (list[float]): A list of MSS scores corresponding to each $k$. 
                                  Can be longer than k_values if padded with 1.0s.
        knee (int, optional): The detected knee point representing the optimal number 
                              of components. If provided, it is marked on the plot.
        folder_name (str, optional): The target directory for saving the plot. 
                                     Defaults to "MSS(k)".
        file_name (str, optional): The name of the output image file. 
                                   Defaults to "MSS vs k".
    """
    # Ensure matched lengths (handles early 1.0 padding cases)
    n = min(len(k_values), len(mss_values))
    ks = list(k_values)[:n]
    mss = list(mss_values)[:n]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, mss, marker='o', linewidth=1.5)
    plt.xlabel("k (number of medoids)")
    plt.ylabel("MSS")
    plt.title("MSS vs k")
    plt.grid(True, alpha=0.3)

    if knee is not None and len(ks) > 0:
        ks_arr = np.asarray(ks)
        idx = int(np.argmin(np.abs(ks_arr - knee)))
        knee_k = int(ks_arr[idx])
        plt.axvline(knee_k, linestyle="--", alpha=0.7, color='r')
        plt.scatter([knee_k], [mss[idx]], s=60, zorder=5)
        plt.annotate(f"knee={knee_k}\nMSS={mss[idx]:.3f}",
                     xy=(knee_k, mss[idx]),
                     xytext=(10, 10), textcoords="offset points")

    plt.tight_layout()
    save_plot(plot=plt, folder_name=folder_name, file_name=file_name)


def save_plot(plot: plt, folder_name: str, file_name: str):
    """
    Saves the given plot to the specified folder and file name.

    Args:
        plot (plt): The matplotlib plot to be saved.
        folder_name (str): The name of the folder to save the plot in.
        file_name (str): The name of the file to save the plot as (without the extension).
    """
    try:
        base_folder = os.path.join(os.getcwd(), 'outputs')
        plot_save_path = os.path.join(base_folder, __get_most_recent_folder(), 'plots', folder_name, f'{file_name}.png')

        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plot.savefig(plot_save_path)
        plot.close()
    except Exception as e:
        print(e)
        return


def __get_most_recent_folder():
    """
    Identifies the most recently created folder within the base folder.
    """
    base_folder = os.path.join(os.getcwd(), 'outputs')
    folders = [
        os.path.join(base_folder, d) for d
        in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, d))
    ]

    if not folders:
        raise FileNotFoundError(f"No folders found in {base_folder}")

    most_recent_folder = max(folders, key=os.path.getctime)
    return most_recent_folder
