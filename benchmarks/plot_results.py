import matplotlib.pyplot as plt
import numpy as np
import json
import glob


def results_matrix(results, libraries, tasks):
    A = np.zeros((len(libraries), len(tasks)))

    for res in results:
        if res["library"] in libraries:
            y = libraries.index(res["library"])
            for line in res["results"]:
                if line["task"] in tasks:
                    x = tasks.index(line["task"])
                    A[y, x] = line["fps"]


def barplot(libraries, tasks, fps, title="", spacing=0.0):
    x = np.arange(len(tasks))  # the label locations
    width = (1.0 - spacing) / fps.shape[1]  # the width of the bars

    params = {
        'font.size': 12,
        "figure.figsize": [10, 8]
    }
    plt.rcParams.update(params)

    fig, ax = plt.subplots()
    rects = []
    for i, lib in enumerate(libraries):
        px = x + (i - len(libraries) // 2) * width
        rects.append(ax.bar(px, fps[i], width, label=lib))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Images/second')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(0.5, 1), loc='upper center', ncol=len(libraries))

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{int(np.round(height))}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    for rect in rects:
        autolabel(rect)

    plt.margins(0.05, 0.15)


def main():
    results = [json.load(open(p)) for p in glob.glob("*_results.json")]

    title = results[0]["gpu"]

    libraries = ["pyronn", "TorchRadon", "TorchRadon half"]
    tasks = ["parallel forward", "parallel backward", "fanbeam forward", "fanbeam backward"]

    fps = results_matrix(results, libraries, tasks)
