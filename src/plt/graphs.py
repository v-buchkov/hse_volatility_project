"""Graphs for strategy output visualization"""
import matplotlib.pyplot as plt
from typing import List, Union


def plot_barchart(dates: list, data: List[Union[int, float]], name: str) -> None:
    """
    Creates a barchart (x-axis: dates, y-axis: values).

        Parameters:
            dates (list): List of x-axis labels
            data (list): List of numerical values for y-axis
            name (str): Graph name

        Returns:
            None
    """
    f, axis = plt.subplots()
    f.suptitle(name, fontsize=20)

    axis.bar(dates, data, color='red')
    plt.xticks(rotation=90)

    f.set_tight_layout(True)
    f.savefig(f'output/{"_".join(name.split(" "))}.jpeg')


def plot_line_chart(dates: list, data: List[Union[int, float]], name: str, label: str) -> None:
    """
    Creates a line chart (x-axis: dates, y-axis: values).

        Parameters:
            dates (list): List of x-axis labels
            data (list): List of numerical values for y-axis
            name (str): Graph name
            label (str): function label

        Returns:
            None
    """
    f, axis = plt.subplots()
    f.suptitle(name, fontsize=20)

    axis.plot(dates, data, color='red', label=label)
    axis.legend(shadow=True)
    plt.xticks(rotation=90)

    f.set_tight_layout(True)
    f.savefig(f'output/{"_".join(name.split(" "))}.jpeg')
