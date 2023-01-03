from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from math import floor
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sko.PSO import PSO


def display_heat_map(data: pd.DataFrame, title: str):
    plt.figure(figsize=(10, 10))
    sns.heatmap(data, linewidth=1, annot=True)
    plt.title(title)
    plt.show()


class PsoMultiCol:

    def set_data(self, df):
        self.df = df
        # calculate the spearmanr correlation of the dataframe's features
        corr = spearmanr(df).correlation
        # make sure it is symmetric
        corr = (corr + corr.T) / 2
        # fill the diagonal with 1s
        np.fill_diagonal(corr, 1)
        # transform the matrix to a dataframe that represents how similar each feature it is to another
        self.dis_matrix = pd.DataFrame(data=(1 - np.abs(corr)), columns=list(df.columns), index=list(df.columns))
        # have a dictionary mapping the column's order to its name
        self.columns_dict = dict(list(zip(range(len(df.columns)), df.columns)))
        # set the number of features for later reference
        self.num_feats = len(df.columns)
        # save the column names for later reference
        self.columns = list(df.columns)

    def __init__(self, df: pd.DataFrame = None, max_iter: int = 200, vif_threshold: float = 2.5, epsilon: int = 0.1,
                 min_fraction=0.40, max_fraction=0.76, step=0.05):  # add other parameters to the game
        self.max_iter = max_iter
        self.pso = None
        # the value that determine whether columns are multi-collinear or not
        self.vif_threshold = vif_threshold
        # an epsilon value used in the evaluation function
        self.epsilon = epsilon
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.step = step
        self.pso = None
        if df is not None:
            self.set_data(df)

    def _get_vif(self, df=None):
        if df is None:
            df = self.df

        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
        vif['variables'] = df.columns
        return vif.set_index('variables')

    def _get_clusters(self, particle: np.array):
        particle_size = len(particle)
        discrete_particle = np.array([int(x) for x in particle])
        cluster_feats = {}
        for i in range(particle_size):
            # if the value of the cluster is not in the dictinary, initialize the list
            if discrete_particle[i] not in cluster_feats:
                cluster_feats[discrete_particle[i]] = []
            # the cluster_feats will be a map between numbers representing clusters
            # and columns representing
            cluster_feats[discrete_particle[i]].append(i)

        return cluster_feats

    def _cluster_scores(self, cluster_feats: dict):
        cluster_names = {}
        new_order = []
        title = ""

        # map each cluster to its column names
        for c, feats in cluster_feats.items():
            cluster_names[c] = [self.columns_dict[i] for i in feats]
            new_order.extend(feats)
            title += f"{len(feats)}-"

        # how similar the points are inside a single cluster
        inner_cluster_score = 0
        for c, names in cluster_feats.items():
            inner_cluster_score += (1 + np.exp(self.dis_matrix.iloc[names, names].values.sum())) \
                                   / np.log(len(names) + np.exp(1))

        # new_dist_matrix = self.dis_matrix.loc[new_order[::-1], new_order]
        # display_heat_map(new_dist_matrix, title)
        return inner_cluster_score  # / inter_cluster_score

    def _pso_function(self, particle: np.array):
        return self._cluster_scores(self._get_clusters(particle))

    def _cluster_pso(self, num_clusters):
        # determine the function object to pass to the PSO algorithm
        pso_function = lambda x: self._pso_function(x)
        # bounds
        lower_bound = np.zeros(self.num_feats)
        upper_bound = np.full(shape=self.num_feats, fill_value=num_clusters, dtype="float")

        pso = sko.PSO.PSO(func=pso_function, n_dim=self.num_feats, pop=15, max_iter=self.max_iter, lb=lower_bound,
                          ub=upper_bound, c1=1.5, c2=1.5)
        pso.run()

        x, y = pso.gbest_x, pso.gbest_y

        cluster_feats = self._get_clusters(x)

        new_order = []
        title = ""
        for _, f in cluster_feats.items():
            new_order.extend(f)
            title += f'{len(f)}-'

        print(y)
        new_dist_matrix = self.dis_matrix.loc[new_order[::-1], new_order]
        display_heat_map(new_dist_matrix, title)

        return x, y

    def _find_best_cluster(self):
        best_score = np.inf
        best_x = None
        last_num_clusters = 0
        for fraction in np.arange(self.min_fraction, self.max_fraction, self.step):
            num_clusters = max(floor(fraction * self.num_feats), 3)

            if num_clusters == last_num_clusters:
                continue

            last_num_clusters = num_clusters

            x, y = self._cluster_pso(num_clusters)
            if y < best_score:
                best_score = y
                best_x = x

        return best_x

    def _get_new_df(self, best_particle):
        # define a PCA object to combine the clustered
        pca = PCA(n_components=1)

        # get the clusters out of the particle
        clusters = self._get_clusters(best_particle)
        # get the cluster
        new_dfs = []

        for _, feats in clusters.items():
            # reduce the clusted features into a single more informative feature
            new_feats = pd.DataFrame(data=pca.fit_transform(self.df.loc[:, feats]), index=list(self.df.index))
            new_dfs.append(new_feats)

        # return the features concatenated horizontally
        return pd.concat(new_dfs, axis=1, ignore_index=True)

    def eliminate_multicol(self, df: pd.DataFrame):
        # first of all determine the vifs of the different columns
        vif = self._get_vif(df)
        # retrieve multicollinear variables
        collinear = list(vif[vif['VIF'] >= self.vif_threshold].index)
        collinear_df = df.loc[:, collinear]

        # retrieve the non-collinear part
        non_collinear = [c for c in df.columns if c not in collinear]
        non_collinear_df = df.loc[:, non_collinear]

        # if there are no collinear columns, no further preprocessing is needed
        if not collinear:
            return df

        # set the df field to the fraction of the dataframe with only multicollinear columns
        self.set_data(collinear_df)
        # retrieve the best particle
        best_x = self._find_best_cluster()
        # retrieve the new representation of the collinear features
        new_collinear_df = self._get_new_df(best_x)
        # concatenate the two parts to form the final dataframe
        return pd.concat([non_collinear_df, new_collinear_df], axis=1)


if __name__ == "__main__":
    pass
