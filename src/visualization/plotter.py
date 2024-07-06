import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display

#################################################################################
# HistogramPlotter
# ClusterPlotter
# BarPlotter
###############################################################################


# Class for creating a Histogram plot with flexible options
class HistogramPlotter:
    def __init__(
        self,
        data,
        title,
        xlabel,
        ylabel,
        filename=None,
        xticks=None,
        bins=None,
        color=None,
        figsize=None,
        alpha=None,
        label=None,
    ):
        self.data = data
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.filename = filename
        self.xticks = xticks
        self.bins = bins
        self.color = color
        self.figsize = figsize
        self.alpha = alpha
        self.label = label

    def plot(self):
        fig, ax = plt.subplots()
        ax.hist(
            self.data,
            bins=self.bins,
            color=self.color,
            alpha=self.alpha,
            label=self.label,
        )
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.bins is not None:
            ax.hist(self.data, bins=self.bins, color=self.color)
        else:
            ax.hist(self.data, color=self.color)
        if self.xticks:
            ax.set_xticks(self.xticks)
        if self.label:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="upper right")
        if self.filename is not None:
            plt.savefig(self.filename)
        else:
            plt.show()
        plt.close(fig)


# Class for creating a Cluster plot with flexible options
class ClusterPlotter:
    def __init__(
        self,
        X_pca,
        clusters,
        title,
        xlabel,
        ylabel,
        filename=None,
        plot_type="2d",
        zlabel=None,
        figsize=(10, 8),
        cmap="viridis",
        edgecolor="k",
        s=50,
        alpha=0.7,
    ):
        self.X_pca = X_pca
        self.clusters = clusters
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel
        self.filename = filename
        self.plot_type = plot_type
        self.figsize = figsize
        self.cmap = cmap
        self.edgecolor = edgecolor
        self.s = s
        self.alpha = alpha

    def plot(self):
        if self.plot_type == "2d":
            self._plot_2d()
        elif self.plot_type == "3d":
            self._plot_3d()
        else:
            raise ValueError("plot_type must be '2d' or '3d'")

    def _plot_2d(self):
        plt.figure(figsize=self.figsize)
        ax = plt.subplot(111)
        scatter = ax.scatter(
            self.X_pca[:, 0],
            self.X_pca[:, 1],
            c=self.clusters,
            cmap=self.cmap,
            edgecolor=self.edgecolor,
            s=self.s,
            alpha=self.alpha,
        )
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        legend = plt.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        if self.filename is not None:
            plt.savefig(self.filename)
        plt.show()
        plt.close()

    def _plot_3d(self):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            self.X_pca[:, 0],
            self.X_pca[:, 1],
            self.X_pca[:, 2],
            c=self.clusters,
            cmap=self.cmap,
            edgecolor=self.edgecolor,
            s=self.s,
            alpha=self.alpha,
        )
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_zlabel(self.zlabel)
        legend = plt.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        if self.filename is not None:
            plt.savefig(self.filename)
        plt.show()
        plt.close()


# Class for creating a bar plot
class BarPlotter:
    def __init__(
        self,
        data,
        title,
        xlabel,
        ylabel,
        filename=None,
        xticks=None,
        color="skyblue",
        edgecolor="k",
        figsize=(10, 6),
    ):
        self.data = data
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.filename = filename
        self.xticks = xticks
        self.color = color
        self.edgecolor = edgecolor
        self.figsize = figsize

    def plot(self):
        plt.figure(figsize=self.figsize)
        plt.bar(
            self.data.index,
            self.data.values,
            color=self.color,
            edgecolor=self.edgecolor,
        )
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        if self.xticks is not None:
            plt.xticks(self.xticks)
        plt.grid(axis="y")
        if self.filename is not None:
            plt.savefig(self.filename)
        plt.show()
        plt.close()
