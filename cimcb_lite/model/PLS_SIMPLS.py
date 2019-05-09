import math
import numpy as np
import pandas as pd
from copy import deepcopy
from itertools import combinations
from sklearn.cross_decomposition import PLSRegression
from bokeh.plotting import output_notebook, show
from bokeh.layouts import gridplot
from bokeh.plotting import ColumnDataSource, figure
from .BaseModel import BaseModel
from ..plot import scatter, distribution, roc_calculate, roc_plot, boxplot


class PLS_SIMPLS(BaseModel):
    """ Partial least-squares regression using the SIMPLS algorithm.

    Parameters
    ----------
    n_components : int, (default 2)
        Number of components to keep.

    Methods
    -------
    train : Fit model to data.

    test : Apply model to test data.

    evaluate : Evaluate model.

    calc_bootci : Calculate bootstrap intervals for plot_featureimportance.

    plot_featureimportance : Plot coefficient and Variable Importance in Projection (VIP).

    plot_permutation_test : Perform a permutation test and plot.
    """

    bootlist = ["model.vip_", "model.coef_"]  # list of metrics to bootstrap

    def __init__(self, n_components=2):
        self.model = PLSRegression()  # Should change this to an empty model
        self.n_component = n_components

    def train(self, X, Y):
        """ Fit the PLS model, save additional stats (as attributes) and return Y predicted values.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.

        Y : array-like, shape = [n_samples, 1]
            Response variables, where n_samples is the number of samples.

        Returns
        -------
        y_pred_train : array-like, shape = [n_samples, 1]
            Predicted y score for samples.
        """

        # Convert to numpy array if a DataFrame
        if isinstance(X, pd.DataFrame or pd.Series):
            X = np.array(X)
            Y = np.array(Y).ravel()

        # Error checks
        if np.isnan(X).any() is True:
            raise ValueError("NaNs found in X.")
        if len(np.unique(Y)) != 2:
            raise ValueError("Y needs to have 2 groups. There is {}".format(len(np.unique(Y))))
        if np.sort(np.unique(Y))[0] != 0:
            raise ValueError("Y should only contain 0s and 1s.")
        if np.sort(np.unique(Y))[1] != 1:
            raise ValueError("Y should only contain 0s and 1s.")
        if len(X) != len(Y):
            raise ValueError("length of X does not match length of Y.")

        # Calculates and store attributes of PLS SIMPLS
        Xscores, Yscores, Xloadings, Yloadings, Weights, Beta = self.pls_simpls(X, Y, ncomp=self.n_component)
        self.model.x_scores_ = Xscores
        self.model.y_scores_ = Yscores
        self.model.x_loadings_ = Xloadings
        self.model.y_loadings_ = Yloadings
        self.model.x_weights_ = Weights
        self.model.beta_ = Beta

        # Calculate pctvar, flatten coef_ and vip for future use
        meanX = np.mean(X, axis=0)
        X0 = X - meanX
        self.model.pctvar_ = sum(abs(self.model.x_loadings_) ** 2) / sum(sum(abs(X0) ** 2)) * 100
        self.model.coef_ = Beta[1:]
        W0 = Weights / np.sqrt(np.sum(Weights ** 2, axis=0))
        sumSq = np.sum(Xscores ** 2, axis=0) * np.sum(Yloadings ** 2, axis=0)
        self.model.vip_ = np.sqrt(len(Xloadings) * np.sum(sumSq * W0 ** 2, axis=1) / np.sum(sumSq, axis=0))

        # Calculate and return Y predicted value
        newX = np.insert(X, 0, np.ones(len(X)), axis=1)
        y_pred_train = np.matmul(newX, Beta)

        # Storing X, Y, and Y_pred
        self.X = X
        self.Y = Y
        self.Y_pred = y_pred_train
        return y_pred_train

    def test(self, X):
        """Calculate and return Y predicted value.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Test variables, where n_samples is the number of samples and n_features is the number of predictors.

        Returns
        -------
        y_pred_test : array-like, shape = [n_samples, 1]
            Predicted y score for samples.
        """
        # Convert to X to numpy array if a DataFrame
        if isinstance(X, pd.DataFrame or pd.Series):
            X = np.array(X)

        # Calculate and return Y predicted value
        newX = np.insert(X, 0, np.ones(len(X)), axis=1)
        y_pred_test = np.matmul(newX, self.model.beta_)
        return y_pred_test

    def plot_projections(self, label=None, size=12):
        """ Plots latent variables projections against each other in a Grid format.

        Parameters
        ----------
        label : DataFrame or None, (default None)
            hovertool for scatterplot.

        size : positive integer, (default 12)
            size specifies circle size for scatterplot.
        """

        num_x_scores = len(self.model.x_scores_.T)

        # If there is only 1 x_score, Need to plot x_score vs. peak (as opposided to x_score[i] vs. x_score[j])
        if num_x_scores == 1:
            # Violin plot
            violin_bokeh = boxplot(self.Y_pred.flatten(), self.Y, title="", xlabel="Class", ylabel="Predicted Score", violin=True, color=["#FFCCCC", "#CCE5FF"], width=320, height=315)
            # Distribution plot
            dist_bokeh = distribution(self.Y_pred, group=self.Y, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315)
            # ROC plot
            fpr, tpr, tpr_ci = roc_calculate(self.Y, self.Y_pred, bootnum=100)
            roc_bokeh = roc_plot(fpr, tpr, tpr_ci, width=310, height=315)
            # Score plot
            y = self.model.x_scores_[:, 0].tolist()
            # get label['Idx'] if it exists
            try:
                x = label["Idx"].values.ravel()
            except:
                x = []
                for i in range(len(y)):
                    x.append(i)
            scatter_bokeh = scatter(x, y, label=label, group=self.Y, ylabel="LV {} ({:0.1f}%)".format(1, self.model.pctvar_[0]), xlabel="Idx", legend=True, title="", width=950, height=315, hline=0, size=int(size / 2), hover_xy=False)

            # Combine into one figure
            fig = gridplot([[violin_bokeh, dist_bokeh, roc_bokeh], [scatter_bokeh]])

        else:
            comb_x_scores = list(combinations(range(num_x_scores), 2))

            # Width/height of each scoreplot
            width_height = int(950 / num_x_scores)
            circle_size_scoreplot = size / num_x_scores
            label_font = str(13 - num_x_scores) + "pt"

            # Create empty grid
            grid = np.full((num_x_scores, num_x_scores), None)

            # Append each scoreplot
            for i in range(len(comb_x_scores)):
                # Make a copy (as it overwrites the input label/group)
                label_copy = deepcopy(label)
                group_copy = self.Y.copy()

                # Scatterplot
                x, y = comb_x_scores[i]
                xlabel = "LV {} ({:0.1f}%)".format(x + 1, self.model.pctvar_[x])
                ylabel = "LV {} ({:0.1f}%)".format(y + 1, self.model.pctvar_[y])
                gradient = self.model.y_loadings_[0][y] / self.model.y_loadings_[0][x]

                max_range = max(np.max(np.abs(self.model.x_scores_[:, x])), np.max(np.abs(self.model.x_scores_[:, y])))
                new_range_min = -max_range - 0.05 * max_range
                new_range_max = max_range + 0.05 * max_range
                new_range = (new_range_min, new_range_max)

                grid[y, x] = scatter(self.model.x_scores_[:, x].tolist(), self.model.x_scores_[:, y].tolist(), label=label_copy, group=group_copy, title="", xlabel=xlabel, ylabel=ylabel, width=width_height, height=width_height, legend=False, size=circle_size_scoreplot, label_font_size=label_font, hover_xy=False, xrange=new_range, yrange=new_range, gradient=gradient)

            # Append each distribution curve
            for i in range(num_x_scores):
                xlabel = "LV {} ({:0.1f}%)".format(i + 1, self.model.pctvar_[i])
                grid[i, i] = distribution(self.model.x_scores_[:, i], group=group_copy, kde=True, title="", xlabel=xlabel, ylabel="density", width=width_height, height=width_height, label_font_size=label_font)

            # Append each roc curve
            for i in range(len(comb_x_scores)):
                x, y = comb_x_scores[i]

                # Get the optimal combination of x_scores based on rotation of y_loadings_
                theta = math.atan(self.model.y_loadings_[0][y] / self.model.y_loadings_[0][x])
                x_rotate = self.model.x_scores_[:, x] * math.cos(theta) + self.model.x_scores_[:, y] * math.sin(theta)

                # ROC Plot with x_rotate
                fpr, tpr, tpr_ci = roc_calculate(group_copy, x_rotate, bootnum=100)
                grid[x, y] = roc_plot(fpr, tpr, tpr_ci, width=width_height, height=width_height, xlabel="1-Specificity (LV{}/LV{})".format(x + 1, y + 1), ylabel="Sensitivity (LV{}/LV{})".format(x + 1, y + 1), legend=False, label_font_size=label_font)

            # Bokeh grid
            fig = gridplot(grid.tolist())

        output_notebook()
        show(fig)

    @staticmethod
    def pls_simpls(X, Y, ncomp=2):
        """PLS SIMPLS method. Refer to https://doi.org/10.1016/0169-7439(93)85002-X"""

        # Error check that X and Y match
        n, dx = X.shape
        ny = len(Y)
        if ny != n:
            raise ValueError("X and Y must have the same number of rows")

        # Error check for ncomp < maxncomp
        maxncomp = min(n - 1, dx)
        if ncomp > maxncomp:
            raise ValueError("ncomp must be less than or equal to {} for these data.".format(maxncomp))

        # Center both predictors and response
        meanX = np.mean(X, axis=0)
        meanY = np.mean(Y, axis=0)
        X0 = X - meanX
        Y0 = Y - meanY
        n, dx = X0.shape
        dy = 1

        # Empty arrays for loadings, scores, and weights
        Xloadings = np.zeros([dx, ncomp])
        Yloadings = np.zeros([dy, ncomp])
        Xscores = np.zeros([n, ncomp])
        Yscores = np.zeros([n, ncomp])
        Weights = np.zeros([dx, ncomp])

        # An orthonormal basis for the X loadings
        V = np.zeros([dx, ncomp])
        Cov = np.matmul(X0.T, Y0)
        Cov = Cov.reshape(len(Cov), 1)

        for i in range(ncomp):
            # Find unit length ti=X0*ri and ui=Y0*ci whose covariance, ri'*X0'*Y0*ci, is jointly maximized, subject to ti'*tj=0 for j=1:(i-1).
            ri, si, ci = np.linalg.svd(Cov)
            ri = ri[:, 0]
            si = si[0]
            ci = ci[0]
            ti = np.matmul(X0, ri)
            normti = np.linalg.norm(ti)
            ti = ti / normti
            qi = si * ci / normti

            Xloadings[:, i] = np.matmul(X0.T, ti)
            Yloadings[:, i] = qi
            Xscores[:, i] = ti
            Yscores[:, i] = (Y0 * qi).tolist()
            Weights[:, i] = ri / normti  #

            # Update the orthonormal basis with modified Gram Schmidt
            vi = Xloadings[:, i]
            for repeat in range(2):
                for j in range(i):
                    vj = V[:, j]
                    vi = vi - np.matmul(vj.T, vi) * vj

            vi = vi / np.linalg.norm(vi)
            V[:, i] = vi

            # Deflate Cov
            vim = vi * np.matmul(vi.T, Cov)
            Cov = Cov - vim.reshape(len(vim), 1)
            Vi = V[:, 0 : i + 1]
            Vim = np.dot(Vi, np.matmul(Vi.T, Cov)).flatten()
            Cov = Cov - Vim.reshape(len(Vim), 1)

        # Use modified Gram-Schmidt, repeated twice.
        for i in range(ncomp):
            ui = Yscores[:, i]
            for repeat in range(2):
                for j in range(i):
                    tj = Xscores[:, j]
                    ui = ui - np.dot(np.matmul(tj.T, ui), tj)
            Yscores[:, i] = ui

        Beta = np.matmul(Weights, Yloadings.T)
        Beta_add = meanY - np.dot(meanX, Beta)
        Beta = np.insert(Beta, 0, Beta_add)
        return Xscores, Yscores, Xloadings, Yloadings, Weights, Beta
