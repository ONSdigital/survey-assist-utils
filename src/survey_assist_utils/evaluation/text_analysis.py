# pylint: disable=too-many-instance-attributes,attribute-defined-outside-init
"""This module provides the TextAnalyser class for analysing free-text responses.

The class is designed to take a pandas DataFrame with a specified text column,
and then perform several analysis steps:
- Embed the text into a semantic vector space (currently using GCP's Vertex AI).
- Identify and filter out null or irrelevant responses.
- Cluster the text embeddings using K-Means to group similar responses.
- Provide methods to investigate the optimal number of clusters (elbow plot).
- Visualise the clusters in a 2D space using t-SNE.
- Identify representative comments for each cluster.
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google import genai
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


class TextAnalyser:
    """A class to handle/assist with text analysis on free-text responses.

    This class automates the process of embedding, clustering, and visualising
    text data from a pandas DataFrame. It is designed to help understand
    the thematic structure within a corpus, aimed at as survey responses.

    Args:
        dataset (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing text to analyse.
        project_id (str): The Google Cloud Project ID (for Vertex AI).
        additional_kwargs (dict | None): Optional dictionary for advanced
            configuration. See the __init__ method for details on available keys.

    Attributes:
        df (pd.DataFrame): The DataFrame holding the text and analysis results.
        text_column (str): The name of the column being analysed.
        project_id (str): The GCP project ID used.
        vectoriser (genai.Client): The client for the embedding model.
        number_of_clusters (int): The number of clusters used in K-Means.
        kmeans (KMeans): The fitted KMeans model object.
        cluster_representatives (list[str]): A list of the most representative
            comments for each cluster.

    Methods:
        embed(texts): Embeds a list of texts into vectors.
        get_distance(vec1, vec2): Calculates the distance between two vectors.
        reset_null_behaviour(example_null_responses, null_marker_threshold):
            Configures and applies the logic for marking null/irrelevant responses.
        drop_null_responses(): Removes rows marked as null from the DataFrame.
        investigate_clusters(kmin, kmax, ...): Creates an elbow plot to help
            determine the optimal number of clusters for K-Means.
        apply_kmeans(k): Applies K-Means clustering with a specified number of
            clusters.
        visualise_dim_reduced(cluster_plot_outfile, cluster_descriptions):
            Generates a 2D t-SNE plot of the clusters and representative comments.
        write_single_cluster_comments(cluster_id): Prints to console all
            comments belonging to a specified cluster.

    Notes:
        Some attributes are created and/or updated by methods. This behaviour will
        be clarified in the docstrings of the particular methods.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        text_column: str,
        project_id: str,
        additional_kwargs: dict | None = None,
    ):  # pylint: disable=too-many-arguments
        """Initialises the TextAnalyser with a dataset, chosen text column,
        and GCP project ID.

        This constructor sets up the internal DataFrame, configures the embedding
        model, and performs initial embedding and null response marking based on
        provided or default parameters.

        Args:
            dataset (pd.DataFrame): The input DataFrame containing the text data.
            text_column (str): The name of the column in `dataset` that holds
                the free-text responses.
            project_id (str): The Google Cloud Project ID required for
                authenticating and using Vertex AI services for embeddings.
            additional_kwargs (dict | None): An optional dictionary to override
                default configuration settings. Supported keys include:
                - 'model_name' (str): The name of the embedding model to use
                    (default: "text-embedding-004").
                - 'model_task_type' (str): The task type for the embedding model
                    (default: "CLASSIFICATION").
                - 'max_batch_size' (int): The maximum number of texts to embed
                    in a single API call (default: 250).
                - 'cleaning_func' (callable): A function to apply to the text
                    column for cleaning before embedding (default: lambda x: x).
                - 'example_null_responses' (list[str]): A list of example
                    responses considered null or irrelevant (default: []).
                - 'null_marker_threshold' (float): The distance threshold for
                    marking responses as null (default: 0.0).
                - 'random_state' (int): Seed for the random number generator for
                    K-Means initialization, ensuring reproducibility (default: 1234).
        """
        self.df = dataset.copy()
        self.text_column = text_column
        self.project_id = project_id
        self.vectoriser = genai.Client(
            vertexai=True, project=project_id, location="europe-west2"
        )
        kwargs = {
            "model_name": "text-embedding-004",
            "model_task_type": "CLASSIFICATION",
            "max_batch_size": 250,
            "cleaning_func": lambda x: x,
            "example_null_responses": [],
            "null_marker_threshold": 0.0,
            "random_state": 1234,
        }
        if additional_kwargs is not None:
            kwargs.update(additional_kwargs)
        self.cleaning_func = kwargs["cleaning_func"]
        self.df[self.text_column] = self.df[self.text_column].apply(self.cleaning_func)
        self.model_name: str = kwargs["model_name"]  # type: ignore[assignment]
        self.task_type: str = kwargs["model_task_type"]  # type: ignore[assignment]
        self.max_batch_size: int = kwargs["max_batch_size"]  # type: ignore[assignment]
        self.random_state: int = kwargs["random_state"]  # type: ignore[assignment]
        self.df["embeddings"] = self.embed(self.df[self.text_column].to_list())
        self.reset_null_behaviour(
            kwargs["example_null_responses"],  # type: ignore[arg-type]
            kwargs["null_marker_threshold"],  # type: ignore[arg-type]
        )

    def embed(self, texts: str | list[str]):
        """Embeds a text string / list of text strings into a semantic vector space.

        This method uses the configured embedding model (Vertex AI) to convert
        text responses into numerical embeddings, processing them in batches
        to account for VertexAI's API limitations.

        Args:
            texts (str | list[str]): A list of strings to be embedded.

        Returns:
            list[list[float]]: A list of embedding vectors, where each vector
            is a list of floats.
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for batch_id in range(0, 1 + len(texts) // self.max_batch_size):
            batch = texts[
                self.max_batch_size * batch_id : self.max_batch_size * (batch_id + 1)
            ]
            embedding_objs = self.vectoriser.models.embed_content(
                model=self.model_name,
                contents=batch,  # type: ignore[arg-type]
                config=genai.types.EmbedContentConfig(task_type=self.task_type),
            )
            if embedding_objs.embeddings is not None:
                embeddings.extend([res.values for res in embedding_objs.embeddings])
        return embeddings

    def get_distance(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculates the Euclidean distance between two vectors.

        Args:
            vec1 (list[float]): The first vector.
            vec2 (list[float]): The second vector.

        Returns:
            float: The Euclidean distance between the two vectors.
        """
        return np.sqrt(np.sum(np.abs(np.array(vec1) - np.array(vec2)) ** 2))

    def reset_null_behaviour(
        self,
        example_null_responses: list | None = None,
        null_marker_threshold: float = 0.0,
    ):
        """Configures and applies the logic for marking null or irrelevant responses.

        If `example_null_responses` are provided, this method embeds them and then
        marks responses in the DataFrame that are semantically close to these
        examples as 'null'. If no examples are given, all responses are considered
        non-null.

        Args:
            example_null_responses (list | None): A list of strings that represent
                null or irrelevant responses. If None, no responses will be marked
                as null.
            null_marker_threshold (float): The maximum distance a response's
                embedding can be from an `example_null_responses` embedding to
                be considered null.

        Modifies:
            self.df (pd.DataFrame): Adds or updates the 'is_null_response' column.
            self: Adds or updates the 'embedded_null_responses' and
                  'null_marker_threshold' attributes.
        """
        if example_null_responses is not None:
            self.null_responses = example_null_responses
            self.null_marker_threshold = null_marker_threshold
            self.embedded_null_responses = self.embed(self.null_responses)
            self._mark_null_responses()
        else:
            self.df["is_null_response"] = False

    def _mark_null_responses(self):
        """Internal method to mark rows in the DataFrame as null responses.

        A response is marked as null if its embedding is within
        `self.null_marker_threshold` distance of any of the
        `self.embedded_null_responses`.

        Modifies:
            self.df (pd.DataFrame): Adds or updates the 'is_null_response' column
                with boolean values.
        """

        def row_has_null_response(row):
            """Helper function to determine if a single row's text is a null response.

            Args:
                row (pd.Series): A row from the DataFrame, expected to contain
                    an 'embeddings' column.

            Returns:
                bool: True if the response is considered null, False otherwise.
            """
            distances = [
                self.get_distance(row["embeddings"], null_val)
                for null_val in self.embedded_null_responses
            ]
            return min(distances) <= self.null_marker_threshold

        self.df["is_null_response"] = self.df.apply(row_has_null_response, axis=1)

    def drop_null_responses(self):
        """Removes rows from the DataFrame that have been marked as null responses.

        This method filters `self.df` to exclude any rows where the
        'is_null_response' column is True.

        Modifies:
            self.df (pd.DataFrame): The DataFrame is updated to contain only
                non-null responses.
        """
        self.df = self.df[~self.df["is_null_response"]]

    def _fit_kmeans(self, k: int):
        """Internal method to fit a K-Means clustering model to the embeddings.

        Args:
            k (int): The number of clusters to form.

        Returns:
            sklearn.cluster.KMeans: A fitted KMeans model object.
        """
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=self.random_state)
        kmeans.fit(self.df["embeddings"].to_list())
        return kmeans

    def investigate_clusters(
        self,
        kmin: int = 2,
        kmax: int = 25,
        elbow_plot_outfile: str = "elbow.png",
    ):
        """Generates an elbow plot to help determine the optimal number of clusters.

        This method fits K-Means models for a range of `k` values and plots
        the inertia (sum of squared distances of samples to their closest
        cluster center) against the number of clusters. The "elbow" point
        in the plot can suggest an appropriate number of clusters.

        Args:
            kmin (int): The minimum number of clusters to consider (inclusive).
            kmax (int): The maximum number of clusters to consider (exclusive).
            elbow_plot_outfile (str): The filename (including path) where the
                elbow plot image will be saved.

        Modifies:
            self.kmeans_range (tuple[int, int]): Stores the range of k values used.

        Outputs:
            A PNG image file of the elbow plot.
        """
        self.kmeans_range = (kmin, kmax)
        inertia_values = []
        k_values = range(self.kmeans_range[0], self.kmeans_range[1])
        for k in k_values:
            kmeans = self._fit_kmeans(k)
            inertia_values.append(kmeans.inertia_)
        fig, ax = plt.subplots()
        ax.plot(k_values, inertia_values, marker="o")
        ax.set_xlabel("Number of clusters (k)", fontsize=14)
        ax.set_ylabel("Inertia", fontsize=14)
        ax.set_title(
            "Elbow Method for Feedback Comment\nK-Means clustering\n(based on semantic embeddings)"
        )
        ax.grid(True)
        fig.tight_layout()
        plt.savefig(elbow_plot_outfile, dpi=250)

    def apply_kmeans(self, k: int):
        """Applies K-Means clustering to the text embeddings with a specified `k`.

        This method fits a K-Means model with `k` clusters to the embeddings,
        assigns cluster labels to each text response in the DataFrame, and
        identifies a representative comment for each cluster.

        Args:
            k (int): The desired number of clusters.

        Modifies:
            self.number_of_clusters (int): Stores the number of clusters used.
            self.kmeans (sklearn.cluster.KMeans): The fitted KMeans model.
            self.df (pd.DataFrame): Adds a 'feedback_comment_labels' column
                with the assigned cluster for each response.
            self.cluster_representatives (list[str]): A list of strings, where
                each string is the most representative comment for its cluster.
        """
        self.number_of_clusters = k
        self.kmeans = self._fit_kmeans(self.number_of_clusters)
        feedback_comment_labels = self.kmeans.predict(self.df["embeddings"].to_list())
        self.df["feedback_comment_labels"] = feedback_comment_labels
        self.cluster_representatives = self._get_representative_comments()

    def _get_representative_comments(self) -> list[str]:
        """Internal method to find the most representative comment for each cluster.

        The representative comment for a cluster is identified as the comment
        whose embedding is closest to the cluster's centroid.

        Returns:
            list[str]: A list of strings, where each string is the most
            representative comment for its corresponding cluster.
        """
        distances_from_centroids = [
            np.array([self.get_distance(c, e) for e in self.df["embeddings"].to_list()])
            for c in self.kmeans.cluster_centers_
        ]
        ids_most_representative_pt = [np.argmin(e) for e in distances_from_centroids]
        return [
            self.df.iloc[pt_id][self.text_column]
            for pt_id in ids_most_representative_pt
        ]

    def visualise_dim_reduced(
        self,
        cluster_plot_outfile: str = "clusters_visualised.png",
        cluster_descriptions: list | None = None,
    ):  # pylint: disable=too-many-locals
        """Generates and saves a 2D t-SNE plot of the clustered embeddings.

        This method reduces the dimensionality of the embeddings using t-SNE
        and visualizes the clusters. It also displays either representative
        comments or user-provided descriptions for each cluster alongside the plot.

        Args:
            cluster_plot_outfile (str): The filename (including path) where the
                cluster visualization image will be saved.
            cluster_descriptions (list | None): An optional list of strings,
                where each string provides a custom description for a cluster.
                If provided, the length of this list must match the number of
                clusters. If None or empty, representative comments will be used.

        Raises:
            ValueError: If `cluster_descriptions` is provided but its length
                does not match the number of clusters.

        Modifies:
            self.tsne_embeddings (np.ndarray): Stores the 2D t-SNE coordinates
                of the individual embeddings.
            self.tsne_centroids (np.ndarray): Stores the 2D t-SNE coordinates
                of the cluster centroids.
        Outputs:
            A PNG image file of the cluster visualization.
        """
        tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=50)
        embeddings_with_centroids = np.vstack(
            [self.df["embeddings"].to_list(), self.kmeans.cluster_centers_]
        )
        dim_reduced_ewc = tsne.fit_transform(embeddings_with_centroids)
        self.tsne_embeddings = dim_reduced_ewc[: len(self.df["embeddings"])]
        self.tsne_centroids = dim_reduced_ewc[len(self.df["embeddings"]) :]
        fig, (ax, ax2) = plt.subplots(  # pylint: disable=unused-variable
            ncols=2, figsize=(13, 5)
        )
        colour_map = {
            0: "blue",
            1: "magenta",
            2: "indigo",
            3: "green",
            4: "red",
            5: "brown",
            6: "orange",
            7: "black",
        }
        for group, group_count in zip(
            *np.unique(self.df["feedback_comment_labels"].to_list(), return_counts=True)
        ):
            idx = np.where(self.df["feedback_comment_labels"].to_list() == group)
            ax.scatter(
                self.tsne_embeddings.T[0][idx],
                self.tsne_embeddings.T[1][idx],
                c=colour_map[group],
                label=f"Cluster {group} | {group_count}",
                s=10,
                alpha=0.333,
            )
        ax.legend()
        ax.set_title(
            "Dimension reduced embedded clusters\n"
            f"(K-Means with {self.number_of_clusters} clusters, "
            "omitting responses detected as 'null')"
        )
        if cluster_descriptions is None or len(cluster_descriptions) == 0:
            ax2.text(
                0,
                0.98,
                "Representative Comments\n--------------------------------------",
                color="k",
            )
            for group in np.unique(self.df["feedback_comment_labels"].to_list()):
                ax2.text(
                    0,
                    0.94 - 0.18 * group,
                    f"cluster {group}: "
                    f"{textwrap.fill(self.cluster_representatives[group], width=60)}",
                    color=colour_map[group],
                )
        elif len(cluster_descriptions) == self.number_of_clusters:
            ax2.text(
                0,
                0.98,
                "Cluster Descriptions\n--------------------------------------",
                color="k",
            )
            for group in np.unique(self.df["feedback_comment_labels"].to_list()):
                ax2.text(
                    0,
                    0.94 - 0.18 * group,
                    f"cluster {group}: {textwrap.fill(cluster_descriptions[group], width=60)}",
                    color=colour_map[group],
                )
        else:
            raise ValueError(
                "Provided cluster descriptions do not match the number of "
                f"clusters. {len(cluster_descriptions)} =/= {self.number_of_clusters}."
            )
        ax2.axis("off")
        plt.tight_layout()
        plt.savefig(cluster_plot_outfile, dpi=250)

    def write_single_cluster_comments(self, cluster_id: int):
        """Prints all comments belonging to a specific cluster to the console.

        This method also prints the most representative comment for the specified
        cluster, followed by all other comments assigned to that cluster.

        Args:
            cluster_id (int): The ID of the cluster whose comments are to be printed.

        Raises:
            ValueError: If `cluster_id` is out of the valid range of clusters.

        Outputs:
            Prints formatted strings to the console.
        """
        if cluster_id < 0 or cluster_id >= self.number_of_clusters:
            raise ValueError(
                f"Cluster ID {cluster_id} is out of range. "
                f"Must be between 0 and {self.number_of_clusters - 1}."
            )
        print(
            "Most representative comment:\n"
            f"{textwrap.fill(self.cluster_representatives[cluster_id])}\n\n"
        )
        for c in self.df[self.df["feedback_comment_labels"] == cluster_id][
            self.text_column
        ]:
            print(f"{textwrap.fill(c, width=100)}\n")
