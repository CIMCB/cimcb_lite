from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd
import scipy
from bokeh.layouts import widgetbox, gridplot, column, row, layout
from bokeh.models import HoverTool, Band
from bokeh.models.widgets import DataTable, Div, TableColumn
from bokeh.models.annotations import Title
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from scipy import interp
from sklearn import metrics
from sklearn.utils import resample
from ..bootstrap import Perc, BC, BCA
from ..plot import scatter, scatterCI, boxplot, distribution, permutation_test, roc_calculate, roc_plot
from ..utils import binary_metrics


class BaseModel(ABC):
    """Base class for models: PLS_SIMPLS."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        """Trains the model."""
        pass

    @abstractmethod
    def test(self):
        """Tests the model."""
        pass

    @abstractproperty
    def bootlist(self):
        """A list of attributes for bootstrap resampling."""
        pass

    def evaluate(self, testset=None, specificity=False, cutoffscore=False, bootnum=1000):
        """Plots a figure containing a Violin plot, Distribution plot, ROC plot and Binary Metrics statistics.

        Parameters
        ----------
        testset : array-like, shape = [n_samples, 2] or None, (default None)
            If testset is None, use train Y and train Y predicted for evaluate. Alternatively, testset is used to evaluate model in the format [Ytest, Ypred].

        specificity : number or False, (default False)
            Use the specificity to draw error bar. When False, use the cutoff score of 0.5.

        cutoffscore : number or False, (default False)
            Use the cutoff score to draw error bar. When False, use the specificity selected.

        bootnum : a positive integer, (default 1000)
            The number of bootstrap samples used in the computation.
        """
        Ytrue_train = self.Y
        Yscore_train = self.Y_pred.flatten()

        # Get Ytrue_test, Yscore_test from testset
        if testset is not None:
            Ytrue_test = np.array(testset[0])
            Yscore_test = np.array(testset[1])

            # Error checking
            if len(Ytrue_test) != len(Yscore_test):
                raise ValueError("evaluate can't be used as length of Ytrue does not match length of Yscore in test set.")
            if len(np.unique(Ytrue_test)) != 2:
                raise ValueError("Ytrue_test needs to have 2 groups. There is {}".format(len(np.unique(Y))))
            if np.sort(np.unique(Ytrue_test))[0] != 0:
                raise ValueError("Ytrue_test should only contain 0s and 1s.")
            if np.sort(np.unique(Ytrue_test))[1] != 1:
                raise ValueError("Ytrue_test should only contain 0s and 1s.")

            # Get Yscore_combined and Ytrue_combined_name (Labeled Ytrue)
            Yscore_combined = np.concatenate([Yscore_train, Yscore_test])
            Ytrue_combined = np.concatenate([Ytrue_train, Ytrue_test + 2])  # Each Ytrue per group is unique
            Ytrue_combined_name = Ytrue_combined.astype(np.str)
            Ytrue_combined_name[Ytrue_combined == 0] = "Train (0)"
            Ytrue_combined_name[Ytrue_combined == 1] = "Train (1)"
            Ytrue_combined_name[Ytrue_combined == 2] = "Test (0)"
            Ytrue_combined_name[Ytrue_combined == 3] = "Test (1)"

        # Expliclity states which metric and value is used for the error_bar
        if specificity is not False:
            metric = "specificity"
            val = specificity
        elif cutoffscore is not False:
            metric = "cutoffscore"
            val = cutoffscore
        else:
            metric = "specificity"
            val = 0.8

        # ROC plot
        tpr, fpr, tpr_ci, stats, stats_bootci = roc_calculate(Ytrue_train, Yscore_train, bootnum=100, metric=metric, val=val)
        roc_title = "Specificity: {}".format(np.round(stats["val_specificity"], 2))
        roc_bokeh = roc_plot(tpr, fpr, tpr_ci, width=320, height=315, title=roc_title, errorbar=stats["val_specificity"])
        if testset is not None:
            fpr_test, tpr_test, threshold_test = metrics.roc_curve(Ytrue_test, Yscore_test, pos_label=1, drop_intermediate=False)
            fpr_test = np.insert(fpr_test, 0, 0)
            tpr_test = np.insert(tpr_test, 0, 0)
            roc_bokeh.line(fpr_test, tpr_test, color="red", line_width=3.5, alpha=0.6, legend="ROC Curve (Test)")  # Add ROC Curve(Test) to roc_bokeh

        # Violin plot
        violin_title = "Cut-off: {}".format(np.round(stats["val_cutoffscore"], 2))
        if testset is None:
            violin_bokeh = boxplot(Yscore_train, Ytrue_train, xlabel="Class", ylabel="Predicted Score", violin=True, color=["#FFCCCC", "#CCE5FF"], width=320, height=315, title=violin_title, font_size="11pt")
        else:
            violin_bokeh = boxplot(Yscore_combined, Ytrue_combined_name, xlabel="Class", ylabel="Predicted Score", violin=True, color=["#fcaeae", "#aed3f9", "#FFCCCC", "#CCE5FF"], width=320, height=315, group_name=["Train (0)", "Test (0)", "Train (1)", "Test (1)"], group_name_sort=["Test (0)", "Test (1)", "Train (0)", "Train (1)"], title=violin_title, font_size="11pt")
        violin_bokeh.multi_line([[-100, 100]], [[stats["val_cutoffscore"], stats["val_cutoffscore"]]], line_color="black", line_width=2, line_alpha=1.0, line_dash="dashed")

        # Distribution plot
        if testset is None:
            dist_bokeh = distribution(Yscore_train, group=Ytrue_train, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315)
        else:
            dist_bokeh = distribution(Yscore_combined, group=Ytrue_combined_name, kde=True, title="", xlabel="Predicted Score", ylabel="p.d.f.", width=320, height=315)
        dist_bokeh.multi_line([[stats["val_cutoffscore"], stats["val_cutoffscore"]]], [[-100, 100]], line_color="black", line_width=2, line_alpha=1.0, line_dash="dashed")

        # Man-Whitney U for Table (round and use scienitic notation if p-value > 0.001)
        manw_pval = scipy.stats.mannwhitneyu(Yscore_train[Ytrue_train == 0], Yscore_train[Ytrue_train == 1], alternative="two-sided")[1]
        if manw_pval > 0.001:
            manw_pval_round = "%0.2f" % manw_pval
        else:
            manw_pval_round = "%0.2e" % manw_pval
        if testset is not None:
            testmanw_pval = scipy.stats.mannwhitneyu(Yscore_test[Ytrue_test == 0], Yscore_test[Ytrue_test == 1], alternative="two-sided")[1]
            if testmanw_pval > 0.001:
                testmanw_pval_round = "%0.2f" % testmanw_pval
            else:
                testmanw_pval_round = "%0.2e" % testmanw_pval

        # Create a stats table for test
        if testset is not None:
            teststats = binary_metrics(Ytrue_test, Yscore_test, cut_off=stats["val_cutoffscore"])
            teststats_round = {}
            for i in teststats.keys():
                teststats_round[i] = np.round(teststats[i], 2)

        # Round stats, and stats_bootci for Table
        stats_round = {}
        for i in stats.keys():
            stats_round[i] = np.round(stats[i], 2)
        bootci_round = {}
        for i in stats_bootci.keys():
            bootci_round[i] = np.round(stats_bootci[i], 2)

        # Create table
        tabledata = dict(
            evaluate=[["Train"]],
            manw_pval=[["{}".format(manw_pval_round)]],
            auc=[["{} ({}, {})".format(stats_round["AUC"], bootci_round["AUC"][0], bootci_round["AUC"][1])]],
            accuracy=[["{} ({}, {})".format(stats_round["ACCURACY"], bootci_round["ACCURACY"][0], bootci_round["ACCURACY"][1])]],
            precision=[["{} ({}, {})".format(stats_round["PRECISION"], bootci_round["PRECISION"][0], bootci_round["PRECISION"][1])]],
            sensitivity=[["{} ({}, {})".format(stats_round["SENSITIVITY"], bootci_round["SENSITIVITY"][0], bootci_round["SENSITIVITY"][1])]],
            specificity=[["{} ({}, {})".format(stats_round["SPECIFICITY"], bootci_round["SPECIFICITY"][0], bootci_round["SPECIFICITY"][1])]],
            F1score=[["{} ({}, {})".format(stats_round["F1-SCORE"], bootci_round["F1-SCORE"][0], bootci_round["F1-SCORE"][1])]],
            R2=[["{} ({}, {})".format(stats_round["R²"], bootci_round["R²"][0], bootci_round["R²"][1])]],
        )

        # Append test data
        if testset is not None:
            tabledata["evaluate"].append(["Test"])
            tabledata["manw_pval"].append([testmanw_pval_round])
            tabledata["auc"].append([teststats_round["AUC"]])
            tabledata["accuracy"].append([teststats_round["ACCURACY"]])
            tabledata["precision"].append([teststats_round["PRECISION"]])
            tabledata["sensitivity"].append([teststats_round["SENSITIVITY"]])
            tabledata["specificity"].append([teststats_round["SPECIFICITY"]])
            tabledata["F1score"].append([teststats_round["F1-SCORE"]])
            tabledata["R2"].append([teststats_round["R²"]])

        # Plot table
        source = ColumnDataSource(data=tabledata)
        columns = [TableColumn(field="evaluate", title="Evaluate"), TableColumn(field="manw_pval", title="MW-U Pvalue"), TableColumn(field="R2", title="R2"), TableColumn(field="auc", title="AUC"), TableColumn(field="accuracy", title="Accuracy"), TableColumn(field="precision", title="Precision"), TableColumn(field="sensitivity", title="Sensitivity"), TableColumn(field="F1score", title="F1score")]
        table_bokeh = widgetbox(DataTable(source=source, columns=columns, width=950, height=90), width=950, height=80)

        # Title
        if specificity is not False:
            title = "Specificity fixed to: {}".format(np.round(val, 2))
        elif cutoffscore is not False:
            title = "Score cut-off fixed to: {}".format(np.round(val, 2))
        else:
            title = "Specificity fixed to: {}".format(np.round(val, 2))
        title_bokeh = "<h3>{}</h3>".format(title)

        # Combine table, violin plot and roc plot into one figure
        fig = layout([[violin_bokeh, dist_bokeh, roc_bokeh], [table_bokeh]], toolbar_location="right")
        output_notebook()
        show(column(Div(text=title_bokeh, width=900, height=50), fig))

    def calc_bootci(self, bootnum=100, type="bca"):
        """Calculates bootstrap confidence intervals based on bootlist.

        Parameters
        ----------
        bootnum : a positive integer, (default 100)
            The number of bootstrap samples used in the computation.

        type : 'bc', 'bca', 'perc', (default 'bca')
            Methods for bootstrap confidence intervals. 'bc' is bias-corrected bootstrap confidence intervals. 'bca' is bias-corrected and accelerated bootstrap confidence intervals. 'perc' is percentile confidence intervals.
        """
        bootlist = self.bootlist
        if type is "bca":
            boot = BCA(self, self.X, self.Y, self.bootlist, bootnum=bootnum)
        if type is "bc":
            boot = BC(self, self.X, self.Y, self.bootlist, bootnum=bootnum)
        if type is "perc":
            boot = Perc(self, self.X, self.Y, self.bootlist, bootnum=bootnum)
        self.bootci = boot.run()

    def plot_featureimportance(self, PeakTable, peaklist=None, ylabel="Label", sort=True):
        """Plots feature importance metrics.

        Parameters
        ----------
        PeakTable : DataFrame
            Peak sheet with the required columns.

        peaklist : list or None, (default None)
            Peaks to include in plot (the default is to include all samples).

        ylabel : string, (default "Label")
            Name of column in PeakTable to use as the ylabel.

        sort : boolean, (default True)
            Whether to sort plots in absolute descending order.

        Returns
        -------
        Peaksheet : DataFrame
            New PeakTable with added "Coef" and "VIP" columns (+ "Coef-95CI" and  "VIP-95CI" if calc_bootci is used prior to plot_featureimportance).
        """
        if not hasattr(self, "bootci"):
            print("Use method calc_bootci prior to plot_featureimportance to add 95% confidence intervals to plots.")
            ci_coef = None
            ci_vip = None
        else:
            ci_coef = self.bootci["model.coef_"]
            ci_vip = self.bootci["model.vip_"]

        # Remove rows from PeakTable if not in peaklist
        if peaklist is not None:
            PeakTable = PeakTable[PeakTable["Name"].isin(peaklist)]
        peaklabel = PeakTable[ylabel]

        # Plot
        fig_1 = scatterCI(self.model.coef_, ci=ci_coef, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=0, col_hline=True, title="Coefficient Plot", sort_abs=sort)
        fig_2 = scatterCI(self.model.vip_, ci=ci_vip, label=peaklabel, hoverlabel=PeakTable[["Idx", "Name", "Label"]], hline=1, col_hline=False, title="Variable Importance in Projection (VIP)", sort_abs=sort)
        fig = layout([[fig_1], [fig_2]])
        output_notebook()
        show(fig)

        # Return table with: Idx, Name, Label, Coefficient, 95CI, VIP, 95CI
        if not hasattr(self, "bootci"):
            coef = pd.DataFrame([self.model.coef_]).T
            coef.rename(columns={0: "Coef"}, inplace=True)
            vip = pd.DataFrame([self.model.vip_]).T
            vip.rename(columns={0: "VIP"}, inplace=True)
        else:
            coef = pd.DataFrame([self.model.coef_, self.bootci["model.coef_"]]).T
            coef.rename(columns={0: "Coef", 1: "Coef-95CI"}, inplace=True)
            vip = pd.DataFrame([self.model.vip_, self.bootci["model.vip_"]]).T
            vip.rename(columns={0: "VIP", 1: "VIP-95CI"}, inplace=True)

        Peaksheet = PeakTable.copy()
        Peaksheet["Coef"] = coef["Coef"].values
        Peaksheet["VIP"] = vip["VIP"].values
        if hasattr(self, "bootci"):
            Peaksheet["Coef-95CI"] = coef["Coef-95CI"].values
            Peaksheet["VIP-95CI"] = vip["VIP-95CI"].values
        return Peaksheet

    def permutation_test(self, nperm=100):
        """Plots permutation test figures.

        Parameters
        ----------
        nperm : positive integer, (default 100)
            Number of permutations.
        """
        fig = permutation_test(self, self.X, self.Y, nperm=nperm)
        output_notebook()
        show(fig)
