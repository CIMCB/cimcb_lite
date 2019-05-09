import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import Circle, HoverTool, TapTool, LabelSet
from tqdm import tqdm
from bokeh.plotting import output_notebook, show
from .BaseCrossVal import BaseCrossVal
from ..utils import binary_metrics


class kfold(BaseCrossVal):
    """ Exhaustitive search over param_dict calculating binary metrics.

    Parameters
    ----------
    model : object
        This object is assumed to store bootlist attributes in .model (e.g. modelPLS.model.x_scores_).

    X : array-like, shape = [n_samples, n_features]
        Predictor variables, where n_samples is the number of samples and n_features is the number of predictors.

    Y : array-like, shape = [n_samples, 1]
        Response variables, where n_samples is the number of samples.

    param_dict : dict
        List of attributes to calculate and return bootstrap confidence intervals.

    folds: : a positive integer, (default 10)
        The number of folds used in the computation.

    bootnum : a positive integer, (default 100)
        The number of bootstrap samples used in the computation for the plot.

    Methods
    -------
    Run: Runs all necessary methods prior to plot.

    Plot: Creates a R2/Q2 plot.
    """

    def __init__(self, model, X, Y, param_dict, folds=10, bootnum=100):
        super().__init__(model=model, X=X, Y=Y, param_dict=param_dict, folds=folds, bootnum=bootnum)
        self.crossval_idx = StratifiedKFold(n_splits=folds)

    def calc_ypred(self):
        """Calculates ypred full and ypred cv."""
        self.ypred_full = []
        self.ypred_cv = []
        for params in self.param_list:
            # Set hyper-parameters
            params_i = params
            model_i = self.model(**params_i)
            # Full
            model_i.train(self.X, self.Y)
            ypred_full_i = model_i.test(self.X)
            self.ypred_full.append(ypred_full_i)
            # CV (for each fold)
            ypred_cv_i = self._calc_cv_ypred(model_i, self.X, self.Y)
            self.ypred_cv.append(ypred_cv_i)

    def calc_stats(self):
        """Calculates binary statistics from ypred full and ypred cv."""
        stats_list = []
        for i in range(len(self.param_list)):
            # Create dictionaries with binary_metrics
            stats_full_i = binary_metrics(self.Y, self.ypred_full[i])
            stats_cv_i = binary_metrics(self.Y, self.ypred_cv[i])
            # Rename columns
            stats_full_i = {k + "full": v for k, v in stats_full_i.items()}
            stats_cv_i = {k + "cv": v for k, v in stats_cv_i.items()}
            stats_cv_i["R²"] = stats_full_i.pop("R²full")
            stats_cv_i["Q²"] = stats_cv_i.pop("R²cv")
            # Combine and append
            stats_combined = {**stats_full_i, **stats_cv_i}
            stats_list.append(stats_combined)
        self.table = self._format_table(stats_list)  # Transpose, Add headers
        return self.table

    def run(self):
        """Runs all functions prior to plot."""
        self.calc_ypred()
        self.calc_stats()
        if self.bootnum > 1:
            self.calc_ypred_boot()
            self.calc_stats_boot()

    def calc_ypred_boot(self):
        """Calculates ypred full and ypred cv for each bootstrap resample."""
        self.ytrue_boot = []
        self.ypred_full_boot = []
        self.ypred_cv_boot = []
        for i in tqdm(range(self.bootnum), desc="Kfold"):
            bootidx_i = np.random.choice(len(self.Y), len(self.Y))
            newX = self.X[bootidx_i, :]
            newY = self.Y[bootidx_i]
            ypred_full_nboot_i = []
            ypred_cv_nboot_i = []
            for params in self.param_list:
                # Set hyper-parameters
                model_i = self.model(**params)
                # Full
                model_i.train(newX, newY)
                ypred_full_i = model_i.test(newX)
                ypred_full_nboot_i.append(ypred_full_i)
                # cv
                ypred_cv_i = self._calc_cv_ypred(model_i, newX, newY)
                ypred_cv_nboot_i.append(ypred_cv_i)
            self.ytrue_boot.append(newY)
            self.ypred_full_boot.append(ypred_full_nboot_i)
            self.ypred_cv_boot.append(ypred_cv_nboot_i)

    def calc_stats_boot(self):
        """Calculates binary statistics from ypred full and ypred cv for each bootstrap resample."""
        self.full_boot_metrics = []
        self.cv_boot_metrics = []
        for i in range(len(self.param_list)):
            stats_full_i = []
            stats_cv_i = []
            for j in range(self.bootnum):
                stats_full = binary_metrics(self.ytrue_boot[j], self.ypred_full_boot[j][i])
                stats_full_i.append(stats_full)
                stats_cv = binary_metrics(self.ytrue_boot[j], self.ypred_cv_boot[j][i])
                stats_cv_i.append(stats_cv)
            self.full_boot_metrics.append(stats_full_i)
            self.cv_boot_metrics.append(stats_cv_i)

    def _calc_cv_ypred(self, model_i, X, Y):
        """Method used to calculate ypred cv."""
        ypred_cv_i = [None] * len(Y)
        for train, test in self.crossval_idx.split(self.X, self.Y):
            X_train = X[train, :]
            Y_train = Y[train]
            X_test = X[test, :]
            model_i.train(X_train, Y_train)
            ypred_cv_i_j = model_i.test(X_test)
            # Return value to y_pred_cv in the correct position # Better way to do this
            for (idx, val) in zip(test, ypred_cv_i_j):
                ypred_cv_i[idx] = val.tolist()
        return ypred_cv_i

    def _format_table(self, stats_list):
        """Make stats pretty (pandas table -> proper names in columns)."""
        table = pd.DataFrame(stats_list).T
        param_list_string = []
        for i in range(len(self.param_list)):
            param_list_string.append(str(self.param_list[i]))
        table.columns = param_list_string
        return table

    def plot(self, metric="r2q2"):
        """Create a full/cv plot using based on metric selected.

        Parameters
        ----------
        metric : string, (default "r2q2")
            metric has to be either "r2q2", "auc", "acc", "f1score", "prec", "sens", or "spec".
        """

        # Choose metric to plot
        metric_title = np.array(["ACCURACY", "AUC", "F1-SCORE", "PRECISION", "R²", "SENSITIVITY", "SPECIFICITY"])
        metric_list = np.array(["acc", "auc", "f1score", "prec", "r2q2", "sens", "spec"])
        metric_idx = np.where(metric_list == metric)[0][0]

        # get full, cv, and diff
        full = self.table.iloc[2 * metric_idx + 1]
        cv = self.table.iloc[2 * metric_idx]
        diff = abs(full - cv)
        full_text = self.table.iloc[2 * metric_idx + 1].name
        cv_text = self.table.iloc[2 * metric_idx].name
        diff_text = "DIFFERENCE " + "(" + full_text + " - " + cv_text + ")"

        # round full, cv, and diff for hovertool
        full_hover = []
        cv_hover = []
        diff_hover = []
        for j in range(len(full)):
            full_hover.append("%.2f" % round(full[j], 2))
            cv_hover.append("%.2f" % round(cv[j], 2))
            diff_hover.append("%.2f" % round(diff[j], 2))

        # get key, values (as string) from param_dict (key -> title, values -> x axis values)
        for k, v in self.param_dict.items():
            key = k
            values = v
        values_string = [str(i) for i in values]

        # store data in ColumnDataSource for Bokeh
        data = dict(full=full, cv=cv, diff=diff, full_hover=full_hover, cv_hover=cv_hover, diff_hover=diff_hover, values_string=values_string)
        source = ColumnDataSource(data=data)

        fig1_yrange = (min(diff) - max(0.1 * (min(diff)), 0.1), max(diff) + max(0.1 * (max(diff)), 0.1))
        fig1_xrange = (min(cv) - max(0.1 * (min(cv)), 0.1), max(cv) + max(0.1 * (max(cv)), 0.1))
        fig1_title = diff_text + " vs " + cv_text

        # Figure 1 (DIFFERENCE (R2 - Q2) vs. Q2)
        fig1 = figure(x_axis_label=cv_text, y_axis_label=diff_text, title=fig1_title, tools="tap,pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select", y_range=fig1_yrange, x_range=fig1_xrange, plot_width=485, plot_height=405)

        # Figure 1: Add a line
        fig1_line = fig1.line(cv, diff, line_width=2, line_color="black", line_alpha=0.25)

        # Figure 1: Add circles (interactive click)
        fig1_circ = fig1.circle("cv", "diff", size=17, alpha=0.7, color="green", source=source)
        fig1_circ.selection_glyph = Circle(fill_color="green", line_width=2, line_color="black", fill_alpha=0.6)
        fig1_circ.nonselection_glyph.fill_color = "green"
        fig1_circ.nonselection_glyph.fill_alpha = 0.4
        fig1_circ.nonselection_glyph.line_color = "white"
        fig1_text = fig1.text(x="cv", y="diff", text="values_string", source=source, text_font_size="10pt", text_color="white", x_offset=-3.5, y_offset=7)
        fig1_text.nonselection_glyph.text_color = "white"
        fig1_text.nonselection_glyph.text_alpha = 0.6

        # Figure 1: Add hovertool
        fig1.add_tools(HoverTool(renderers=[fig1_circ], tooltips=[(full_text, "@full_hover"), (cv_text, "@cv_hover"), ("Diff", "@diff_hover")]))

        # Figure 1: Extra formating
        fig1.axis.major_label_text_font_size = "8pt"
        if metric is "r2q2" or metric is "auc":
            fig1.title.text_font_size = "12pt"
            fig1.xaxis.axis_label_text_font_size = "10pt"
            fig1.yaxis.axis_label_text_font_size = "10pt"
        else:
            fig1.title.text_font_size = "10pt"
            fig1.xaxis.axis_label_text_font_size = "9pt"
            fig1.yaxis.axis_label_text_font_size = "9pt"

        # Figure 2: full/cv
        fig2_title = full_text + " & " + cv_text + " vs no. of components"
        fig2 = figure(x_axis_label="components", y_axis_label="Value", title=fig2_title, plot_width=485, plot_height=405, x_range=pd.unique(values_string), y_range=(0, 1.1), tools="pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select")

        # Figure 2: add confidence intervals if bootnum > 1
        if self.bootnum > 1:
            lower_ci_full = []
            upper_ci_full = []
            lower_ci_cv = []
            upper_ci_cv = []
            # Get all upper, lower 95% CI (full/cv) for each specific n_component and append
            for m in range(len(self.full_boot_metrics)):
                full_boot = []
                cv_boot = []
                for k in range(len(self.full_boot_metrics[0])):
                    full_boot.append(self.full_boot_metrics[m][k][metric_title[metric_idx]])
                    cv_boot.append(self.cv_boot_metrics[m][k][metric_title[metric_idx]])
                # Calculated percentile 95% CI and append
                full_bias = np.mean(full_boot) - full[m]
                cv_bias = np.mean(cv_boot) - cv[m]
                lower_ci_full.append(np.percentile(full_boot, 2.5) - full_bias)
                upper_ci_full.append(np.percentile(full_boot, 97.5) - full_bias)
                lower_ci_cv.append(np.percentile(cv_boot, 2.5) - cv_bias)
                upper_ci_cv.append(np.percentile(cv_boot, 97.5) - cv_bias)

            # Plot as a patch
            x_patch = np.hstack((values_string, values_string[::-1]))
            y_patch_r2 = np.hstack((lower_ci_full, upper_ci_full[::-1]))
            fig2.patch(x_patch, y_patch_r2, alpha=0.10, color="red")
            y_patch_q2 = np.hstack((lower_ci_cv, upper_ci_cv[::-1]))
            fig2.patch(x_patch, y_patch_q2, alpha=0.10, color="blue")

        # Figure 2: add full
        fig2_line_full = fig2.line(values_string, full, line_color="red", line_width=2)
        fig2_circ_full = fig2.circle("values_string", "full", line_color="red", fill_color="white", fill_alpha=1, size=8, source=source, legend=full_text)
        fig2_circ_full.selection_glyph = Circle(line_color="red", fill_color="white", line_width=2)
        fig2_circ_full.nonselection_glyph.line_color = "red"
        fig2_circ_full.nonselection_glyph.fill_color = "white"
        fig2_circ_full.nonselection_glyph.line_alpha = 0.4

        # Figure 2: add cv
        fig2_line_cv = fig2.line(values_string, cv, line_color="blue", line_width=2)
        fig2_circ_cv = fig2.circle("values_string", "cv", line_color="blue", fill_color="white", fill_alpha=1, size=8, source=source, legend=cv_text)
        fig2_circ_cv.selection_glyph = Circle(line_color="blue", fill_color="white", line_width=2)
        fig2_circ_cv.nonselection_glyph.line_color = "blue"
        fig2_circ_cv.nonselection_glyph.fill_color = "white"
        fig2_circ_cv.nonselection_glyph.line_alpha = 0.4

        # Add hovertool and taptool
        fig2.add_tools(HoverTool(renderers=[fig2_circ_full], tooltips=[(full_text, "@full_hover")], mode="vline"))
        fig2.add_tools(HoverTool(renderers=[fig2_circ_cv], tooltips=[(cv_text, "@cv_hover")], mode="vline"))
        fig2.add_tools(TapTool(renderers=[fig2_circ_full, fig2_circ_cv]))

        # Figure 2: Extra formating
        fig2.axis.major_label_text_font_size = "8pt"
        if metric is "r2q2" or metric is "auc":
            fig2.title.text_font_size = "12pt"
            fig2.xaxis.axis_label_text_font_size = "10pt"
            fig2.yaxis.axis_label_text_font_size = "10pt"
        else:
            fig2.title.text_font_size = "10pt"
            fig2.xaxis.axis_label_text_font_size = "9pt"
            fig2.yaxis.axis_label_text_font_size = "9pt"

        # Figure 2: legend
        if metric is "r2q2":
            fig2.legend.location = "top_left"
        else:
            fig2.legend.location = "bottom_right"

        # Create a grid and output figures
        grid = np.full((1, 2), None)
        grid[0, 0] = fig1
        grid[0, 1] = fig2
        fig = gridplot(grid.tolist(), merge_tools=True)
        output_notebook()
        show(fig)
