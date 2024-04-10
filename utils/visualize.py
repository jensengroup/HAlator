import numpy as np
import matplotlib as mpl

# mpl.use('Agg')
# mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

fm.fontManager.addfont("/groups/kemi/borup/HAlator/utils/Helvetica.ttf")
fm.findfont("Helvetica", fontext="ttf", rebuild_if_missing=False)

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scipy.stats import (
    spearmanr,
    pearsonr,
)


def plot_confusion_matrix(cm, fig_name, save_fig=False):
    font = {
        "family": "sans-serif",
        "sans-serif": "Helvetica",
        "weight": "normal",
        "size": 20,
    }
    plt.rc("font", **font)
    plt.rc("text", usetex=True)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("axes", labelsize=20)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt="g", annot_kws={"fontsize": 20})
    # annot=True to annotate cells

    # labels, title and ticks

    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["0", "1"])
    ax.yaxis.set_ticklabels(["0", "1"])
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name, format="svg", dpi=1200)

    return plt.show()


def plot_linear_fit(ax, x, y):
    fit = np.polyfit(x, y, 1)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = fit[0] * x_fit + fit[1]
    ax.plot(x_fit, y_fit, linestyle="--", color="green")


def plot_multiple_subplots_delta_g(
    n,
    x_list,
    y_test,
    titles_list,
    save_fig=False,
    fig_name="fig.png",
    lst_textstr=None,
    outliers=False,
    residual=5,
):
    font = {
        "family": "sans-serif",
        "sans-serif": "Helvetica",
        "weight": "normal",
        "size": 20,
    }
    plt.rc("font", **font)
    # plt.rc("text", usetex=True)
    mpl.rcParams["mathtext.rm"] = "Helvetica"
    # mpl.rcParams['mathtext.it'] = 'Helvetica:italic'
    mpl.rcParams["axes.unicode_minus"] = False
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("axes", labelsize=20)
    single_plot_width = 6  # Width of a single subplot
    single_plot_height = 6  # Height of a single subplot

    f, ax0 = plt.subplots(1, n, figsize=(single_plot_width * n, single_plot_height))
    if n == 1:
        ax0 = [ax0]

    f.set_facecolor("white")

    for i in range(n):
        if lst_textstr is not None:
            textstr = f"{lst_textstr[i]}"
        else:
            textstr = ""

        ax0[i].scatter(x_list[i], y_test[i], alpha=0.5, s=12, color="black")

        x_list[i] = np.array(x_list[i]).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
        print(f"shape of x_list: {x_list[i].shape}")
        print(f"shape of y_test: {y_test.shape}")
        # x = np.array(x_list[i]).reshape(-1, 1)
        # y = np.array(y_test[i]).reshape(-1, 1)
        x = x_list[i]
        y = y_test
        model = LinearRegression().fit(x, y)

        y_pred = model.predict(x)
        residuals = y_test - y_pred[i]

        # Highlight residuals greater than 10
        indices_gt = np.where(np.abs(residuals) >= residual)[0]
        # print(np.where(np.abs(residuals) >= residual))
        # print(f"outliers: {list(indices_gt)}")

        if outliers:
            ax0[i].scatter(
                x_list[i][indices_gt], y_test[indices_gt], alpha=0.5, s=12, color="red"
            )

        if model.intercept_ > 0:
            reg = f"{float(model.coef_[0]):.2f}x+{float(model.intercept_):.2f}"
        else:
            reg = f"{float(model.coef_[0]):.2f}x{float(model.intercept_):.2f}"

        r_sq = model.score(x, y)
        rmse = mean_squared_error(y, model.predict(x), squared=False)
        mae = mean_absolute_error(y, model.predict(x))
        spearman, _ = spearmanr(y, model.predict(x))
        pearson, _ = pearsonr(y.flatten(), model.predict(x).flatten())
        # textstr = textstr+f"\n{reg}\nr\u00b2 = {r_sq:.2f}\nr = {pearson:.2f}\n"+r"$\rho$"+f" = {spearman:.2f}"+f"\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
        textstr = (
            textstr
            + f"\n{reg}\nr = {pearson:.2f}\n"
            + "\u03C1"
            + f" = {spearman:.2f}"
            + f"\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
        )

        print(f"model coef: {model.coef_}")
        print(f"model intercept: {model.intercept_}")
        print(f"shape x : {x.shape}")
        plot_linear_fit(
            ax0[i],
            x.reshape(
                -1,
            ),
            y_test,
        )

        ax0[i].grid(False)
        ax0[i].text(
            0.02,
            0.98,
            textstr,
            transform=ax0[i].transAxes,
            fontsize=20,
            verticalalignment="top",
        )

        # ax0[i].set_ylim([-5, 40])
        # ax0[i].set_xticks(np.arange(260, 340, 20))
        ax0[i].set_yticks(np.arange(-5, 40, 5))

        # Set minor ticks
        ax0[i].xaxis.set_minor_locator(MultipleLocator(10))
        ax0[i].yaxis.set_minor_locator(MultipleLocator(5))

        ax0[i].set_ylabel(
            "Experimental $\mathrm{pK_{a}}$"
        )  # Experimental pKa $\mathrm{pK_{a}}$"
        ax0[i].set_xlabel("QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ [kcal/mol]")
        ax0[i].set_title(titles_list[i], fontsize=16)

    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name, format="svg", dpi=1200)


def plot_multiple_subplots_delta_g_halator(
    n,
    x_list,
    y_test,
    titles_list,
    save_fig=False,
    fig_name="fig.png",
    lst_textstr=None,
    outliers=False,
    residual=5,
):
    font = {
        "family": "sans-serif",
        "sans-serif": "Helvetica",
        "weight": "normal",
        "size": 20,
    }
    plt.rc("font", **font)
    # plt.rc("text", usetex=True)
    mpl.rcParams["mathtext.rm"] = "Helvetica"
    # mpl.rcParams['mathtext.it'] = 'Helvetica:italic'
    mpl.rcParams["axes.unicode_minus"] = False
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("axes", labelsize=20)
    single_plot_width = 6  # Width of a single subplot
    single_plot_height = 6  # Height of a single subplot

    f, ax0 = plt.subplots(1, n, figsize=(single_plot_width * n, single_plot_height))
    if n == 1:
        ax0 = [ax0]

    f.set_facecolor("white")

    for i in range(n):
        if lst_textstr is not None:
            textstr = f"{lst_textstr[i]}"
        else:
            textstr = ""

        ax0[i].scatter(x_list[i], y_test[i], alpha=0.5, s=12, color="black")

        x_list[i] = np.array(x_list[i]).reshape(-1, 1)
        y_test[i] = np.array(y_test[i]).reshape(-1, 1)
        print(f"shape of x_list: {x_list[i].shape}")
        print(f"shape of y_test: {y_test[i].shape}")
        # x = np.array(x_list[i]).reshape(-1, 1)
        # y = np.array(y_test[i]).reshape(-1, 1)
        x = x_list[i]
        y = y_test[i]
        model = LinearRegression().fit(x, y)

        y_pred = model.predict(x)
        residuals = y_test[i] - y_pred[i]

        # Highlight residuals greater than 10
        indices_gt = np.where(np.abs(residuals) >= residual)[0]
        # print(np.where(np.abs(residuals) >= residual))
        # print(f"outliers: {list(indices_gt)}")

        if outliers:
            ax0[i].scatter(
                x_list[i][indices_gt],
                y_test[i][indices_gt],
                alpha=0.5,
                s=12,
                color="red",
            )

        if model.intercept_ > 0:
            reg = f"{float(model.coef_[0]):.2f}x+{float(model.intercept_):.2f}"
        else:
            reg = f"{float(model.coef_[0]):.2f}x{float(model.intercept_):.2f}"

        r_sq = model.score(x, y)
        rmse = mean_squared_error(y, model.predict(x), squared=False)
        mae = mean_absolute_error(y, model.predict(x))
        spearman, _ = spearmanr(y, model.predict(x))
        pearson, _ = pearsonr(y.flatten(), model.predict(x).flatten())
        # textstr = textstr+f"\n{reg}\nr\u00b2 = {r_sq:.2f}\nr = {pearson:.2f}\n"+r"$\rho$"+f" = {spearman:.2f}"+f"\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
        textstr = (
            textstr
            + f"\n{reg}\nr = {pearson:.2f}\n"
            + "\u03C1"
            + f" = {spearman:.2f}"
            + f"\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
        )

        print(f"model coef: {model.coef_}")
        print(f"model intercept: {model.intercept_}")
        print(f"shape x : {x.shape}")
        plot_linear_fit(
            ax0[i],
            x.reshape(
                -1,
            ),
            y_test[i],
        )

        ax0[i].grid(False)
        ax0[i].text(
            0.02,
            0.98,
            textstr,
            transform=ax0[i].transAxes,
            fontsize=20,
            verticalalignment="top",
        )

        # ax0[i].set_ylim([-5, 40])
        # ax0[i].set_xticks(np.arange(260, 340, 20))
        # ax0[i].set_yticks(np.arange(-5, 40, 5))

        # Set minor ticks
        ax0[i].xaxis.set_minor_locator(MultipleLocator(10))
        ax0[i].yaxis.set_minor_locator(MultipleLocator(5))

        ax0[i].set_ylabel("Experimental HA")  # Experimental pKa $\mathrm{pK_{a}}$"
        ax0[i].set_xlabel("QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ [kcal/mol]")
        ax0[i].set_title(titles_list[i], fontsize=16)

    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name, format="svg", dpi=1200)


def plot_multiple_subplots_calc_pred(
    n,
    y_preds_list,
    y_test,
    titles_list,
    save_fig=False,
    fig_name="fig.png",
    lst_textstr=None,
    outliers=False,
    residual=5,
):
    font = {
        "family": "sans-serif",
        "sans-serif": "Helvetica",
        "weight": "normal",
        "size": 20,
    }

    plt.rc("font", **font)
    # Set 'Helvetica' as the default sans-serif font
    # Use the sans-serif font also for math text rendered by LaTeX
    # mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    # plt.rcParams['text.usetex'] = True
    # plt.rcParams['text.latex.preamble'] = [
    # r'\usepackage{sansmath}',
    # r'\usepackage{helvet}',
    # r'\sansmath']
    # mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams["mathtext.rm"] = "Helvetica"
    # mpl.rcParams['mathtext.it'] = 'Helvetica:italic'
    mpl.rcParams["axes.unicode_minus"] = False
    # plt.rc("text", usetex=True)
    # plt.rc("mathtext.regular")
    # plt.rc('mathtext', rm='Helvetica')
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("axes", labelsize=20)
    single_plot_width = 6  # Width of a single subplot
    single_plot_height = 6  # Height of a single subplot

    f, ax0 = plt.subplots(1, n, figsize=(single_plot_width * n, single_plot_height))
    if n == 1:
        ax0 = [ax0]
    # sharey="row"

    # f.set_figwidth(12)
    # f.set_figheight(4)
    f.set_facecolor("white")
    # f.set_figaspect('equal')
    for i in range(n):
        if lst_textstr is not None:
            textstr = f"({lst_textstr[i]})"
        else:
            textstr = ""
        # textstr= f"({string.ascii_lowercase[i]})"
        # For ax0
        ax0[i].scatter(y_preds_list[i], y_test, alpha=0.5, s=12, color="black")

        residuals = y_test - y_preds_list[i]

        # Highlight residuals greater than 10
        indices_gt = np.where(np.abs(residuals) >= residual)[0]
        print(f"outliers: {list(indices_gt)}")
        if outliers:
            ax0[i].scatter(
                y_preds_list[i][indices_gt],
                y_test[indices_gt],
                alpha=0.5,
                s=12,
                color="red",
            )

        y_preds_list[i] = np.array(y_preds_list[i]).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)
        print(f"shape of x_list: {y_preds_list[i].shape}")
        print(f"shape of y_test: {y_test.shape}")
        # x = np.array(x_list[i]).reshape(-1, 1)
        # y = np.array(y_test[i]).reshape(-1, 1)
        x = y_preds_list[i]
        y = y_test
        model = LinearRegression().fit(x, y)

        if model.intercept_ > 0:
            reg = f"{float(model.coef_[0]):.2f}x+{float(model.intercept_):.2f}"
        else:
            reg = f"{float(model.coef_[0]):.2f}x{float(model.intercept_):.2f}"

        r_sq = model.score(x, y)
        rmse = mean_squared_error(y, model.predict(x), squared=False)
        mae = mean_absolute_error(y, model.predict(x))
        spearman, _ = spearmanr(y, model.predict(x))
        pearson, _ = pearsonr(y.flatten(), model.predict(x).flatten())
        # pearson = r_sq**0.5
        # textstr = textstr+f"\n{reg}\nr = {pearson:.2f}\n"+r"$\mathbf{\rho}$"+f" = {spearman:.2f}"+f"\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
        textstr = (
            textstr
            + f"\n{reg}\nr = {pearson:.2f}\n"
            + "\u03C1"
            + f" = {spearman:.2f}"
            + f"\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
        )

        print(f"model coef: {model.coef_}")
        print(f"model intercept: {model.intercept_}")
        print(f"shape x : {x.shape}")
        plot_linear_fit(
            ax0[i],
            x.reshape(
                -1,
            ),
            y_test,
        )

        # plot_linear_fit(ax0[i], y_preds_list[i], y_test)
        ax0[i].plot(y, y, linestyle="--", color="black")

        # Adding scores to legends (replace 'compute_score' with your actual function)
        #     for name, score in compute_scores_regression(y_test, y_preds_list[i]).items():
        #    # for name, score in compute_scores_regression(y_test, y_preds_list[i]).items():
        #         ax0[i].plot([], [], " ", label=f"{name}={score:.2f}")

        ax0[i].legend(
            loc="upper left", bbox_to_anchor=(-0.18, 1.0), frameon=False, fontsize=20
        )  # bbox_to_anchor=(-0.15, 0.95)
        ax0[i].grid(False)
        ax0[i].text(
            0.02,
            0.98,
            textstr,
            transform=ax0[i].transAxes,
            fontsize=20,
            verticalalignment="top",
        )
        ax0[i].set_xlim([-5, 60])
        ax0[i].set_ylim([-5, 60])
        ax0[i].set_xticks(np.arange(-5, 60, 10))
        ax0[i].set_yticks(np.arange(-5, 60, 10))
        # Set minor ticks
        ax0[i].xaxis.set_minor_locator(MultipleLocator(5))
        ax0[i].yaxis.set_minor_locator(MultipleLocator(5))

        # mpl.rcParams['mathtext.bf'] = 'Helvetica:bold'
        # mpl.rcParams['text.usetex'] = True
        # ax0[i].set_ylabel("QM computed $\mathrm{pK_{a}}$")  # Experimental pKa #Calculated pKa $\mathrm{pK_{a}}$
        # ax0[i].set_xlabel("ML predicted $\mathrm{pK_{a}}$")  # ${a_b}_c$ Calculated pKa #Predicted pKa $\mahrm{pK_{a}}$
        ax0[i].set_ylabel(
            "QM computed pK$\mathrm{_a}$"
        )  # Experimental pKa #Calculated pKa $\mathrm{pK_{a}}$
        ax0[i].set_xlabel(
            "ML predicted pK$\mathrm{_a}$"
        )  # ${a_b}_c$ Calculated pKa #Predicted pKa $\mahrm{pK_{a}}$
        # x_pos = 0.95  # This is an example position for the 'a'; adjust it as needed
        # y_pos = -0.05  # This is an example position for the 'a'; adjust it as needed
        # ax0[i].text(x_pos, y_pos, 'a', transform=ax0[i].xaxis.label.get_transform(), ha='right', va='top', fontsize=20)

        ax0[i].set_title(titles_list[i])
        ax0[i].tick_params(axis="both", which="both", length=5)

    # f.suptitle("bordwellCH pKa values. 20 conformers", y=1.05)
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name, format="svg", dpi=1200)


def plot_single_subplot_delta_g_halator(
    x,
    y,
    title="",
    y_label="Experimental HA",
    x_label="QM computed $\mathrm{\Delta G ^{\circ} _{min}}$ [kcal/mol]",
    save_fig=False,
    save_format="svg",
    fig_name="single_fig.svg",
    textstr=None,
    outliers=False,
    residual=5,
    fig=None,  # Make sure to define fig as an optional argument
):
    # Apply font settings globally
    font = {
        "family": "sans-serif",
        "sans-serif": "Helvetica",
        "weight": "normal",
        "size": 20,
    }
    plt.rc("font", **font)
    plt.rcParams["mathtext.rm"] = "Helvetica"
    plt.rcParams["axes.unicode_minus"] = False

    # Check if a figure object is passed, otherwise create a new figure
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)  # Add a subplot to the existing/new figure

    # Your plotting logic remains the same
    ax.scatter(x, y, alpha=0.5, s=12, color="black")
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    residuals = y - y_pred
    print(f"mean abs residual {np.mean(np.abs(residuals)):.1f}")
    print(f"max abs residual {max(np.abs(residuals))[0]:.1f}")

    ax.plot(x, y_pred, linestyle="--", color="green")
    # ax.plot(((min(x)), max(x)), (min(y), max(y)), linestyle="-", color="k")

    if outliers:
        indices_gt = np.where(np.abs(residuals) >= residual)[0]
        ax.scatter(x[indices_gt], y[indices_gt], alpha=0.5, s=12, color="red")
        print(f"index of outliers: {indices_gt}")
        for outlier_idx, delta_g_min, exp in zip(
            indices_gt, x[indices_gt], y[indices_gt]
        ):
            print(
                f"outlier idx: {outlier_idx}, delta_g_min: {delta_g_min[0]:.2f}, exp: {exp[0]:.2f}"
            )

    reg = f"{float(model.coef_[0]):.2f}x" + ("+" if model.intercept_ >= 0 else "")
    reg += f"{float(model.intercept_):.2f}"

    pearson, _ = pearsonr(y.flatten(), model.predict(x).flatten())
    spearman, _ = spearmanr(y, model.predict(x))
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    # textstr = (
    #     (textstr or "")
    #     + f"\n{reg}\nr = {pearson:.2f}\n\u03C1 = {spearman:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
    # )

    if not textstr:
        textstr = f"{reg}\nr = {pearson:.2f}\n\u03C1 = {spearman:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
    else:
        textstr = (
            textstr
            + f"\n{reg}\nr = {pearson:.2f}\n\u03C1 = {spearman:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
        )

    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="top",
    )

    # To add the identity line:
    # lims = [
    #     np.min([x, y]),  # min of both axes
    #     np.max([x, y]),  # max of both axes
    # ]
    # ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
    # ax.set_aspect("equal")
    # ax.set_xlim(lims)
    # ax.set_ylim(lims)

    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.set_ylabel(f"{y_label}")
    ax.set_xlabel(f"{x_label}")
    ax.set_title(title, fontsize=20)

    # Adjust layout
    fig.tight_layout()

    # Save figure if required
    if save_fig:
        if save_format == "svg":
            plt.savefig(f"{fig_name}.{save_format}", format="svg", dpi=1200)
        if save_format == "png":
            plt.savefig(f"{fig_name}.{save_format}", format="png", dpi=1200)
        else:
            plt.savefig(f"{fig_name}.{save_format}", format="pdf", dpi=1200)

    return fig, model.coef_[0][0], model.intercept_[0]  # Return the figure object


def plot_single_subplot_calc_pred_halator(
    x,
    y,
    title="",
    y_label="Experimental HA",
    x_label="QM computed HA",
    save_fig=False,
    save_format="svg",
    fig_name="single_fig.svg",
    textstr=None,
    outliers=False,
    residual=5,
    fig=None,  # Make sure to define fig as an optional argument
):
    # Apply font settings globally
    font = {
        "family": "sans-serif",
        "sans-serif": "Helvetica",
        "weight": "normal",
        "size": 20,
    }
    plt.rc("font", **font)
    plt.rcParams["mathtext.rm"] = "Helvetica"
    plt.rcParams["axes.unicode_minus"] = False

    # Check if a figure object is passed, otherwise create a new figure
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)  # Add a subplot to the existing/new figure

    # Your plotting logic remains the same
    ax.scatter(x, y, alpha=0.5, s=12, color="black")
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    model = LinearRegression().fit(x, y)
    y_pred = model.predict(x)
    residuals = y - y_pred
    print(f"mean abs residual {np.mean(np.abs(residuals)):.1f}")
    print(f"max abs residual {max(np.abs(residuals))[0]:.1f}")

    ax.plot(x, y_pred, linestyle="--", color="green")
    # ax.plot(((min(x)), max(x)), (min(y), max(y)), linestyle="-", color="k")

    if outliers:
        indices_gt = np.where(np.abs(residuals) >= residual)[0]
        ax.scatter(x[indices_gt], y[indices_gt], alpha=0.5, s=12, color="red")

        for delta_g_min, exp in zip(x[indices_gt], y[indices_gt]):
            print(f"delta_g_min: {delta_g_min[0]:.2f}, exp: {exp[0]:.2f}")

    reg = f"{float(model.coef_[0]):.2f}x" + ("+" if model.intercept_ >= 0 else "")
    reg += f"{float(model.intercept_):.2f}"

    pearson, _ = pearsonr(y.flatten(), model.predict(x).flatten())
    spearman, _ = spearmanr(y, model.predict(x))
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)

    if not textstr:
        textstr = f"{reg}\nr = {pearson:.2f}\n\u03C1 = {spearman:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
    else:
        textstr = (
            textstr
            + f"\n{reg}\nr = {pearson:.2f}\n\u03C1 = {spearman:.2f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}\n"
        )

    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="top",
    )

    # To add the identity line:
    # lims = [
    #     np.min([x, y]),  # min of both axes
    #     np.max([x, y]),  # max of both axes
    # ]
    # ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
    # ax.set_aspect("equal")
    # ax.set_xlim(lims)
    # ax.set_ylim(lims)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # Plot the identity line
    ax.plot(lims, lims, "k-", alpha=0.75, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.set_ylabel(f"{y_label}")
    ax.set_xlabel(f"{x_label}")
    ax.set_title(title, fontsize=20)

    # Adjust layout
    fig.tight_layout()

    # Save figure if required
    if save_fig:
        if save_format == "svg":
            plt.savefig(f"{fig_name}.{save_format}", format="svg", dpi=1200)
        if save_format == "png":
            plt.savefig(f"{fig_name}.{save_format}", format="png", dpi=1200)
        else:
            plt.savefig(f"{fig_name}.{save_format}", format="pdf", dpi=1200)

    return fig, model.coef_[0][0], model.intercept_[0]  # Return the figure object


def create_benchmark_hist(
    lst_errors,
    save_fig=False,
    save_format="svg",
    fig_name="single_fig.svg",
):

    fig, ax = plt.subplots(figsize=(10, 6))

    font = {
        "family": "sans-serif",
        "sans-serif": "Helvetica",
        "weight": "normal",
        "size": 20,
    }
    plt.rc("font", **font)
    plt.rcParams["mathtext.rm"] = "Helvetica"
    plt.rcParams["axes.unicode_minus"] = False
    # Positions for the boxplots
    positions = [0.0, 0.6, 1.2, 1.8, 2.4]  # Adjust as needed

    # Width of the boxes
    widths = 0.3  # Adjust as needed

    # Create side-by-side boxplots with adjusted width and positions
    #  [
    #     [x for x in xtb_errors2 if not np.isnan(x)],
    #     [x for x in r2scan_sp_errors2 if not np.isnan(x)],
    #     [x for x in camb3lyp4d_sp_errors2 if not np.isnan(x)],
    #     [x for x in r2scan_optfreq_errors2 if not np.isnan(x)],
    #     [x for x in camb3lyp4d_optfreq_errors2 if not np.isnan(x)],
    # ]
    boxplot = ax.boxplot(
        lst_errors,
        vert=True,
        positions=positions,
        showmeans=False,
        meanline=False,
        meanprops={"color": "red", "linewidth": 2},
        showfliers=False,
        widths=widths,
        patch_artist=True,
        boxprops=dict(facecolor="white", linewidth=1),
    )
    for element in boxplot["medians"]:
        element.set_color("black")

    # Create a patch for each boxplot
    # patches = [
    #     mpatches.Patch(color="white", label="(1) GFN2-xTB"),
    #     mpatches.Patch(color="white", label="(2) r\u00b2SCAN-3c sp"),
    #     mpatches.Patch(color="white", label="(3) CAM-B3LYP D4 sp"),
    #     mpatches.Patch(color="white", label="(4) r\u00b2SCAN-3c opt-freq"),
    #     mpatches.Patch(color="white", label="(5) CAM-B3LYP D4 opt-freq"),
    # ]

    # Add the legend to the plot
    # ax.legend(
    #     handles=patches, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left"
    # )  # handles=patches, frameon=False, bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3

    # Customize the x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_ylabel("Absolute error [HA units]")
    ax.set_xlabel("Method")
    # ax.set_title('Absolute errors for each method', fontdict={'fontsize': 16, 'fontweight': 'bold'})
    plt.tight_layout()
    # Save figure if required
    if save_fig:
        if save_format == "svg":
            plt.savefig(f"{fig_name}.{save_format}", format="svg", dpi=1200)
        if save_format == "png":
            plt.savefig(f"{fig_name}.{save_format}", format="png", dpi=1200)
        else:
            plt.savefig(f"{fig_name}.{save_format}", format="pdf", dpi=1200)
