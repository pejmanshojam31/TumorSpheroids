import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import statsmodels.api as sm
import os
import re
from matplotlib.backends.backend_pdf import PdfPages
class Visualization:
    def __init__(self, output_dir, color_map=None, y_true=None, y_pred=None, y_proba=None):
        self.output_dir = output_dir
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.color_map = {
            'FaDu': {
                'controlled': 'steelblue',   # changed from '#1f77b4'
                'relapsed': 'lightblue'      # changed from '#d62728'
            },
            'SAS': {
                'controlled': 'steelblue',   # changed from '#2ca02c'
                'relapsed': 'lightblue'      # changed from '#ff7f0e'
            },
            'Unknown': {
                'controlled': '#7f7f7f',
                'relapsed': '#bcbd22'
            }
        }


    def _save_pdf(self, fig, filename: str):
            """
            Save the current Matplotlib figure as a PDF in self.output_dir.

            Parameters
            ----------
            fig : matplotlib.figure.Figure
                The figure object to save.
            filename : str
                Base name (without extension) for the saved PDF.
            """
            pdf_path = os.path.join(self.output_dir, f"{filename}.pdf")
            fig.savefig(pdf_path, format="pdf", bbox_inches="tight")


    def plot_confusion_matrix(self, normalize=False, title="Confusion Matrix", use_best_threshold=False):
        """
        Plot the confusion matrix as a heatmap.

        :param normalize: Whether to normalize the confusion matrix.
        :param title: Title of the plot.
        :param use_best_threshold: If True, use self.best_y_pred instead of self.y_pred.
        """
        if use_best_threshold and hasattr(self, 'best_y_pred'):
            y_pred = self.best_y_pred
        else:
            y_pred = self.y_pred

        cm = confusion_matrix(self.y_true, y_pred)
        fmt = ".2f" if normalize else "d"
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # ✅ Controlled is Positive, Relapsed is Negative
        ticklabels = ["Controlled\n (Positive)", "Relapsed\n (Negative)"]

        plt.figure(figsize=(12, 9))
        ax = sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=ticklabels,
            yticklabels=ticklabels,
            annot_kws={"size": 26, "weight": "bold"},
            cbar=True
        )

        # Colorbar styling
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=24)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight('bold')
        cbar.set_label('Normalized Frequency' if normalize else 'Count',
                    size=34, weight='bold', labelpad=30)

        suffix = " (Optimal Threshold)" if use_best_threshold else ""
        plt.xlabel("Predicted Label", fontsize=34, fontweight="bold", labelpad=30)
        plt.ylabel("True Label", fontsize=34, fontweight="bold", labelpad=30)
        plt.xticks(fontsize=24, fontweight="bold", rotation=0)
        plt.yticks(fontsize=24, fontweight="bold", rotation=0)

        plt.savefig(r"E:\Project_HTWD\revised 8\Figure 5A.pdf",
                    format="pdf", bbox_inches="tight", dpi=300)
        plt.show()


    def plot_roc_curve(self, title="ROC Curve"):
        """
        Plot the ROC curve, calculate AUC, and store the best threshold and predictions.
        """
        if self.y_proba is None:
            print("ROC Curve requires prediction probabilities.")
            return

        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_proba)
        roc_auc = auc(fpr, tpr)

        # Find best threshold using Youden's J statistic
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_fpr = fpr[best_idx]
        best_tpr = tpr[best_idx]
        best_threshold = thresholds[best_idx]

        # Store best threshold and predictions using it
        self.best_threshold = best_threshold
        self.best_y_pred = (self.y_proba >= best_threshold).astype(int)

        # Plot ROC
        plt.figure(figsize=(12,8))
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2.5, linestyle='--', label='Random Guess')
        plt.axvline(x=best_fpr, linestyle='--', color='grey', lw=2)
        plt.axhline(y=best_tpr, linestyle='--', color='grey', lw=2)
        # Make the dot for the threshold bigger (s=180)
        plt.scatter([best_fpr], [best_tpr], color='red', s=180, zorder=5, label=f'Best Threshold = {best_threshold:.2f}')

        plt.xlabel('False Positive Rate', fontsize=31, fontweight="bold", labelpad=30)
        plt.ylabel('True Positive Rate', fontsize=31, fontweight="bold", labelpad=30)
        plt.xticks(fontsize=22, fontweight="bold")
        plt.yticks(fontsize=22, fontweight="bold")
        plt.legend(loc="lower right", prop={'size': 22, 'weight': 'bold'},frameon=False)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.savefig(r"E:\Project_HTWD\revised 8\Figure 7B.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.show()

    
        
    '''def plot_precision_recall_curve(self, title="Precision-Recall Curve"):
        """
        Plot the precision-recall curve.
        
        :param title: Title of the plot.
        """
        if self.y_proba is None:
            print("Precision-Recall Curve requires prediction probabilities.")
            return

        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label="Precision-Recall Curve")
        plt.title(title, fontsize=20, fontweight="bold")
        plt.xlabel('Recall', fontsize=16, fontweight="bold")
        plt.ylabel('Precision', fontsize=16, fontweight="bold")
        plt.xticks(fontsize=14, fontweight="bold")
        plt.yticks(fontsize=14, fontweight="bold")
        plt.legend(loc="lower left", fontsize=12)
        plt.savefig("Precision-Recall Curve_300dpi.pdf", format="pdf", bbox_inches="tight", dpi=300)

        plt.show()
'''
    def plot_prediction_distribution(self, title="Prediction Probability Distribution"):
        """
        Plot the distribution of prediction probabilities.
        
        :param title: Title of the plot.
        """
        if self.y_proba is None:
            print("Prediction Probability Distribution requires prediction probabilities.")
            return

        plt.figure(figsize=(10, 8))
        sns.histplot(self.y_proba[self.y_true == 0], color="blue", label="Controlled (True Class 0)", kde=True)
        sns.histplot(self.y_proba[self.y_true == 1], color="red", label="Relapsed (True Class 1)", kde=True)
        plt.title(title, fontsize=20, fontweight="bold")
        plt.xlabel("Predicted Probability", fontsize=16, fontweight="bold")
        plt.ylabel("Density", fontsize=16, fontweight="bold")
        plt.xticks(fontsize=14, fontweight="bold")
        plt.yticks(fontsize=14, fontweight="bold")
        plt.legend(fontsize=12)
        plt.savefig("Prediction Probability Distribution_300dpi.pdf", format="pdf", bbox_inches="tight", dpi=300)
        plt.show()

    def plot_feature_importances(self, model, feature_names=None, top_n=20, title="Feature Importances"):
        """
        Plot feature importances if available.
        
        :param model: Trained machine learning model (must support feature_importances_ or coef_).
        :param feature_names: List of feature names.
        :param top_n: Number of top features to display.
        :param title: Title of the plot.
        """
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
            else:
                print("The model does not support feature importances or coefficients.")
                return

            indices = np.argsort(importances)[-top_n:][::-1]
            feature_importances = importances[indices]
            feature_labels = feature_names[indices] if feature_names is not None else [f"Feature {i}" for i in indices]

            plt.figure(figsize=(12, 8))
            sns.barplot(x=feature_importances, y=feature_labels, palette="viridis")
            plt.title(title, fontsize=20, fontweight="bold")
            plt.xlabel("Importance Score", fontsize=16, fontweight="bold")
            plt.ylabel("Feature", fontsize=16, fontweight="bold")
            plt.xticks(fontsize=14, fontweight="bold")
            plt.yticks(fontsize=14, fontweight="bold")
            plt.savefig("feature_importance_300dpi.pdf", format="pdf", bbox_inches="tight", dpi=300)
            plt.show()

        except Exception as e:
            print(f"Error plotting feature importances: {e}")

    def plot_accuracy_by_DoR(
        self,
        all_days,
        accuracy_values,
        total_samples,
        label,
        dod_map,
        interval=3,
        # ── layout / canvas controls ───────────────────────────────────────
        fig_size=(14, 8),          # make taller to get more canvas
        use_constrained=False,     # True = matplotlib’s constrained layout
        bottom_margin=0.60,        # used when use_constrained=False
        apply_tight_layout=False,  # if True, uses tight_layout(rect=...) instead of subplots_adjust
        save_bbox_tight=True,      # let savefig include outer labels/text
        save_pad_inches=0.5,       # extra border when bbox_inches='tight'
        # ── label text & font sizes ───────────────────────────────────────
        x_label='Day of Relapse (+Time Gain by Prediction)',
        y_label='Accuracy',
        y2_label='Number of Samples',
        x_label_fontsize=16,
        y_label_fontsize=16,
        y2_label_fontsize=14,
        xtick_fontsize=12,
        ytick_fontsize=12,
        y2tick_fontsize=16,
        dod_dor_fontsize=13,       # the small DoD/DoR text drawn below bars
        legend_fontsize=16,
        x_label_pad=30,
        y_label_pad=30,
        y2_label_pad=30,
        xtick_rotation=90,
        # ── styling ───────────────────────────────────────────────────────
        bar_color='steelblue',
        fit_color='blue',
        sample_line_color='black',
        sample_line_width=3,
        fit_line_style='--',
        bar_alpha=0.8,
        capsize=5,
    ):
        """
        Bar plot of accuracy vs. Day-of-Relapse (DoR) with Day-of-Diagnosis (DoD) under each bar.
        Fits log(accuracy) = β0 + β1·DoR (WLS), prints τ = 1/|β1| and t1/2 = ln(2)·τ.
        Only DoR ≤ 36 is shown.
        """
        # ── filter by DoR ≤ 36 ────────────────────────────────────────────
        mask            = np.asarray(all_days) <= 36
        all_days        = np.asarray(all_days)[mask]
        accuracy_values = np.asarray(accuracy_values)[mask]
        total_samples   = np.asarray(total_samples)[mask]
        se              = np.sqrt((accuracy_values * (1 - accuracy_values)) / total_samples)

        x_pos = all_days.astype(float)

        # dynamic bar width (80% of smallest spacing)
        if len(x_pos) > 1:
            bar_width = 0.8 * np.diff(np.sort(x_pos)).min()
        else:
            bar_width = 0.8

        # ── figure / axes ─────────────────────────────────────────────────
        if use_constrained:
            fig, ax1 = plt.subplots(figsize=fig_size, layout="constrained")
        else:
            fig, ax1 = plt.subplots(figsize=fig_size)

        # ── bars with binomial SE ─────────────────────────────────────────
        ax1.bar(
            x_pos,
            accuracy_values,
            width=bar_width,
            color=bar_color,
            yerr=se,
            ecolor=bar_color,
            capsize=capsize,
            alpha=bar_alpha,
            label='_nolegend_',
            align='center',
        )

        # labels/limits on primary axis
        ax1.set_xlabel("")
        # place both pieces centered under the axis
        ax1.text(0.45, -0.4, "Day of Relapse ", transform=ax1.transAxes,
                ha='right', va='center', fontsize=x_label_fontsize, fontweight='bold', color='black')
        ax1.text(0.45, -0.4, "(+Time Gain by Prediction)", transform=ax1.transAxes,
                ha='left',  va='center', fontsize=x_label_fontsize, fontweight='bold', color='red')
        #ax1.xaxis.set_label_coords(0.5, -0.4)  # push the x-label further down
        ax1.set_ylabel(y_label, fontsize=y_label_fontsize, fontweight='bold', color=bar_color, labelpad=y_label_pad)
        ax1.set_ylim(0, 1.05)
        ax1.set_yticks(np.linspace(0, 1.0, 6))
        ax1.set_yticklabels([f'{y:.1f}' for y in np.linspace(0, 1.0, 6)],
                            fontsize=ytick_fontsize, fontweight='bold', color=bar_color)
        ax1.tick_params(axis='y', colors=bar_color, width=2, labelsize=ytick_fontsize)
        ax1.grid(False)

        # ── manual DoD + DoR labels under the axis ───────────────────────
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([''] * len(x_pos), fontsize=xtick_fontsize)
        for xpos, dor in zip(x_pos, all_days):
            dod = dod_map.get(int(dor), np.nan)
            dor_str = f'{int(dor)}'
            dod_str = (f'+{int(dod)}' if (not np.isnan(dod) and dod > 0)
                    else (f'{int(dod)}' if not np.isnan(dod) else ''))
            # DoD (red)
            ax1.text(xpos, -0.03, f'({dod_str})',
                    transform=ax1.get_xaxis_transform(),
                    ha='center', va='top', rotation=xtick_rotation,
                    fontsize=dod_dor_fontsize, fontweight='bold', color='red')
            # DoR (black)
            ax1.text(xpos, -0.18, dor_str,
                    transform=ax1.get_xaxis_transform(),
                    ha='center', va='top', rotation=xtick_rotation,
                    fontsize=dod_dor_fontsize, fontweight='bold', color='black')

        # ── secondary axis: sample counts ─────────────────────────────────
        ax2 = ax1.twinx()
        ax2.plot(x_pos, total_samples, color=sample_line_color, marker='o',
                linewidth=sample_line_width, label='_nolegend_')
        ax2.set_ylabel(y2_label, fontsize=y2_label_fontsize,
                    fontweight='bold', color=sample_line_color, labelpad=y2_label_pad)
        ax2.set_ylim(0, max(total_samples) * 1.2 if len(total_samples) else 1)
        ax2.grid(False)
        ax2.tick_params(axis='y', colors=sample_line_color, width=3, labelsize=y2tick_fontsize)
        for tick in ax2.get_yticklabels():
            tick.set_fontweight('bold')

        # ── weighted exponential fit on (0,1) accuracies ─────────────────
        mask_fit = (accuracy_values > 0) & (accuracy_values < 1) & (~np.isnan(se))
        if mask_fit.sum() >= 2:
            Xd = sm.add_constant(x_pos[mask_fit])
            model = sm.WLS(np.log(accuracy_values[mask_fit]), Xd,
                        weights=1 / (se[mask_fit] ** 2))
            fit_result = model.fit()
            y_pred = np.exp(fit_result.predict(sm.add_constant(x_pos)))

            slope     = fit_result.params[1]
            e_folding = 1 / abs(slope) if slope != 0 else np.inf
            half_life = np.log(2) * e_folding
            print(f"📉 {label.capitalize()} range: "
                f"Slope (β₁) = {slope:.4f} [1/d], "
                f"E-folding time (τ) = {e_folding:.2f} d, "
                f"Half-life (t₁/₂) = {half_life:.2f} d")

            fit_line, = ax1.plot(x_pos, y_pred, fit_line_style, linewidth=2.5,
                                color=fit_color, label='Accuracy Fit')
            ax1.legend(handles=[fit_line], labels=['Accuracy Fit'],
                    loc='upper center', bbox_to_anchor=(0.5, 1.02),
                    frameon=False, fontsize=legend_fontsize)
        else:
            print(f"ℹ️ {label.capitalize()}: not enough points for a stable exponential fit.")

        # ── limits and spacing ───────────────────────────────────────────
        ax1.set_xlim(x_pos.min() - bar_width, x_pos.max() + bar_width)

        # Choose one spacing strategy:
        if use_constrained:
            pass  # constrained layout manages spacing automatically
        elif apply_tight_layout:
            # Reserve bottom slice so your DoR/DoD texts aren’t clipped.
            plt.tight_layout(rect=[0, bottom_margin, 1, 1])
        else:
            # Manual control: safest to guarantee space for the annotations below.
            plt.subplots_adjust(bottom=bottom_margin)

        # ── save ─────────────────────────────────────────────────────────
        save_kwargs = {}
        if save_bbox_tight:
            save_kwargs.update(dict(bbox_inches="tight", pad_inches=save_pad_inches))
        plt.savefig(os.path.join(self.output_dir, f'{label}_accuracy_DoR_plot.png'), dpi=300, **save_kwargs)
        plt.close()
        
        
    def plot_combined_accuracy_by_SCP(
        self,
        df: pd.DataFrame,
        title: str,
        filename: str,
        exp_name: str = "",
        show_legend: bool = True,
        annotate: bool = False,
        use_labels: bool = False,  # kept for API compatibility
        xtick_fontsize: int = 12,
        xtick_rotation: int = 0,   # currently not used (manual labels)
        bar_width: float = 0.6,
        axis_label_fontsize: int = 28,
        tick_label_fontsize: int = 22,
        annotation_fontsize: int = 20,
        sort_cols: list[str] | None = None,
        style: dict | None = None,          # NEW
        legend_kwargs: dict | None = None,  # NEW (optional)
    ):
        """Draws the *combined* SCP plot (all data points)."""
        if sort_cols is None:
            sort_cols = ["SCP", "dose", "temperature", "time"]
        sort_cols = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

        for cell_line in df["cell_line"].unique():
            df_cl = df[df["cell_line"] == cell_line].reset_index(drop=True)
            x = np.arange(len(df_cl))
            fig, ax1 = plt.subplots(figsize=(26, 8))
            ax2 = ax1.twinx()
            fname_full = f"{cell_line.lower()}_{exp_name.lower()}_{filename}" if exp_name else f"{cell_line.lower()}_{filename}"

            for i, row in df_cl.iterrows():
                self._draw_bars(ax1, x[i], row, bar_width)
                if annotate:
                    ax1.text(
                        x[i], 1.04,
                        f"{row['controlled_accuracy']:.2f}\n{row['relapsed_accuracy']:.2f}",
                        ha="center", va="bottom",
                        fontsize=annotation_fontsize, fontweight="bold"
                    )

            # shared styling with per-plot overrides
            self._style_axes(
                ax1, ax2, df_cl, x,
                xtick_fontsize, tick_label_fontsize,
                y_axis_label_fontsize=axis_label_fontsize,   # both axes share this size
                x_axis_label_fontsize=axis_label_fontsize,
                filename=filename,
                **(style or {})
            )

            if show_legend:
                self._add_legend(ax1, cell_line, legend_kwargs=legend_kwargs)

            #ax1.set_title(title, fontsize=axis_label_fontsize, fontweight="bold", pad=16)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(os.path.join(self.output_dir, fname_full), dpi=300, bbox_inches="tight")
            plt.close()

    # ------------------------------------------------------------- #
    # 2.  MIDDLE-SCP PLOTTER                                        #
    # ------------------------------------------------------------- #
    def plot_middle_accuracy_by_SCP(
        self,
        df: pd.DataFrame,
        title: str,
        filename: str,
        exp_name: str = "",
        show_legend: bool = True,
        annotate: bool = True,
        use_labels: bool = True,  # kept for API compatibility
        xtick_fontsize: int = 28,
        xtick_rotation: int = 0,  # currently not used (manual labels)
        bar_width: float = 0.3,
        y_label_fontsize: int = 28,      # separate control
        x_label_fontsize: int = 30,      # separate control
        tick_label_fontsize: int = 28,
        annotation_fontsize: int = 28,
        sort_cols: list[str] | None = None,
        style: dict | None = None,          # NEW
        legend_kwargs: dict | None = None,  # NEW (optional)
    ):
        """Draws the *middle-SCP* subset plot (0 < SCP < 1)."""
        df = df.query("0 < SCP < 1")
        if sort_cols is None:
            sort_cols = ["SCP", "dose", "temperature", "time"]
        sort_cols = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

        for cell_line in df["cell_line"].unique():
            df_cl = df[df["cell_line"] == cell_line].reset_index(drop=True)
            x = np.arange(len(df_cl))
            fig, ax1 = plt.subplots(figsize=(20, 12))
            ax2 = ax1.twinx()
            fname_full = f"{cell_line.lower()}_{exp_name.lower()}_{filename}" if exp_name else f"{cell_line.lower()}_{filename}"

            for i, row in df_cl.iterrows():
                self._draw_bars(ax1, x[i], row, bar_width)
                if annotate:
                    ax1.text(
                        x[i], 1.04,
                        f"{row['controlled_accuracy']:.2f}\n{row['relapsed_accuracy']:.2f}",
                        ha="center", va="bottom",
                        fontsize=annotation_fontsize, fontweight="bold"
                    )

            # shared styling with per-plot overrides
            self._style_axes(
                ax1, ax2, df_cl, x,
                xtick_fontsize, tick_label_fontsize,
                y_axis_label_fontsize=y_label_fontsize,
                x_axis_label_fontsize=x_label_fontsize,
                filename=filename,
                **(style or {})
            )

            if show_legend:
                self._add_legend(ax1, cell_line, legend_kwargs=legend_kwargs)

            #x1.set_title(title, fontsize=max(x_label_fontsize, y_label_fontsize), fontweight="bold", pad=16)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(os.path.join(self.output_dir, fname_full), dpi=300, bbox_inches="tight")
            plt.close()

    # ------------------------------------------------------------- #
    # 3.  SHARED DRAW + STYLE HELPERS                               #
    # ------------------------------------------------------------- #
    def _draw_bars(self, ax, xpos, row, width: float):
        cell_line = row["cell_line"]
        ctrl_color = self.color_map.get(cell_line, {}).get("controlled", "steelblue")
        rela_color = self.color_map.get(cell_line, {}).get("relapsed", "lightblue")

        cp, ca = float(row["SCP"]), float(row["controlled_accuracy"])
        rp, ra = 1 - cp, float(row["relapsed_accuracy"])

        # controlled stack
        ax.bar(xpos, cp, width=width, facecolor="none", edgecolor="black", linewidth=2)
        ax.bar(xpos, cp * ca, width=width, color=ctrl_color, edgecolor="none")
        ax.bar(
            xpos, cp * (1 - ca), width=width, facecolor="white", edgecolor="none",
            hatch="//", bottom=cp * ca
        )

        # relapsed stack (inverted)
        ax.bar(
            xpos, rp, width=width, facecolor="none", edgecolor="black",
            linewidth=2, bottom=1 - rp
        )
        ax.bar(
            xpos, rp * ra, width=width, color=rela_color, edgecolor="none",
            bottom=1 - rp * ra
        )
        ax.bar(
            xpos, rp * (1 - ra), width=width, facecolor="white", edgecolor="none",
            hatch="//", bottom=1 - rp
        )

        # SCP marker
        ax.scatter(xpos, cp, color="black", s=40, zorder=5)

    def _style_axes(
        self,
        ax1, ax2,
        df_cl: pd.DataFrame, x_positions: np.ndarray,
        xtick_fontsize: int, tick_label_fontsize: int,
        y_axis_label_fontsize: int,
        x_axis_label_fontsize: int,
        filename: str,
        *,
        # --- existing knobs ---
        bottom_margin: float = 0.30,             # space for multi-row x labels
        x_label_text: str = "Treatment Arm",
        x_label_y: float = -0.35,                # vertical position of x-axis label
        row_offsets: tuple[float, float, float] = (-0.08, -0.16, -0.24),  # dose, temp, time
        header_rt_pos: tuple[float, float] = (-0.09, -0.09),              # ("RT", (Gy)) anchor x,y
        header_ht_pos: tuple[float, float] = (-0.09, -0.21),              # ("HT", (°C)) anchor x,y
        unit_rt_pos:   tuple[float, float] = (-0.02, -0.09),              # (Gy) x,y
        unit_ht_pos:   tuple[float, float] = (-0.02, -0.17),              # (°C) x,y
        unit_time_pos: tuple[float, float] = (-0.02, -0.25),              # (min) x,y
        left_ylabel: str  = "Proportion of Cases",
        right_ylabel: str = "Inverted Proportion of Cases",
        left_labelpad: int = 30,
        right_labelpad: int = 30,
        invert_right_axis: bool = True,

        # --- NEW: per-plot control of decorative lines ---
        draw_decor: bool = True,
        hline_y: float = -0.13,
        hline_xmin: float = -1,
        hline_xmax: float = 1,
        hline_linewidth: float = 1.0,
        hline_color: str = "black",

        vline_x: float = -0.08,
        vline_ymin: float = -0.27,
        vline_ymax: float = -0.06,
        vline_linewidth: float = 1.2,
        vline_color: str = "black",
    ):
        """Axis formatting shared by both plotters, with overrideable layout knobs."""
        ax1.set_ylim(0, 1.0)
        ax1.set_xlim(-1, len(df_cl))
        ax1.axhline(0, color="black", linewidth=1.5)

        # left & right y-axis labels
        ax1.set_ylabel(
            left_ylabel,
            fontsize=y_axis_label_fontsize, fontweight="bold", labelpad=left_labelpad
        )
        ax2.set_ylabel(
            right_ylabel,
            fontsize=y_axis_label_fontsize, fontweight="bold", labelpad=right_labelpad
        )

        ax1.tick_params(axis="y", labelsize=tick_label_fontsize)
        ax2.tick_params(axis="y", labelsize=tick_label_fontsize)
        for lbl in ax1.get_yticklabels() + ax2.get_yticklabels():
            lbl.set_fontweight("bold")

        # manual x-axis row labels
        dose_y, temp_y, time_y = row_offsets
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels([])
        ax1.figure.subplots_adjust(bottom=bottom_margin)

        for i, d in enumerate(df_cl.get("dose", pd.Series(index=df_cl.index, dtype=float))):
            ax1.text(
                i, dose_y, f"{d:.0f}", transform=ax1.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=xtick_fontsize, fontweight="bold"
            )
        for i, t in enumerate(df_cl.get("temperature", pd.Series(index=df_cl.index, dtype=float))):
            ax1.text(
                i, temp_y, f"{t:.0f}", transform=ax1.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=xtick_fontsize, fontweight="bold"
            )
        for i, T in enumerate(df_cl.get("time", pd.Series(index=df_cl.index, dtype=float))):
            ax1.text(
                i, time_y, f"{T:.0f}", transform=ax1.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=xtick_fontsize, fontweight="bold"
            )

        # axis headers
        ax1.text(unit_rt_pos[0],   unit_rt_pos[1],   "(Gy)", transform=ax1.transAxes,
                ha="right", va="center", fontsize=xtick_fontsize, fontweight="bold")
        ax1.text(header_rt_pos[0], header_rt_pos[1], "RT",   transform=ax1.transAxes,
                ha="right", va="center", fontsize=xtick_fontsize, fontweight="bold")
        ax1.text(unit_ht_pos[0],   unit_ht_pos[1],   "(°C)", transform=ax1.transAxes,
                ha="right", va="center", fontsize=xtick_fontsize, fontweight="bold")
        ax1.text(header_ht_pos[0], header_ht_pos[1], "HT",   transform=ax1.transAxes,
                ha="right", va="center", fontsize=xtick_fontsize, fontweight="bold")
        ax1.text(unit_time_pos[0], unit_time_pos[1], "(min)", transform=ax1.transAxes,
                ha="right", va="center", fontsize=xtick_fontsize, fontweight="bold")

        # X-axis label (separate location & size control)
        ax1.text(0.5, x_label_y, x_label_text, transform=ax1.transAxes,
                ha="center", va="center",
                fontsize=x_axis_label_fontsize, fontweight="bold")

        # small annotation for middle-scp
        if "middle_scp" in filename:
            ax1.text(-0.02, 1.04, "Acc: C\n        R", transform=ax1.transAxes,
                    ha="center", va="bottom",
                    fontsize=26, fontweight="bold")

        # --- NEW block: decorative lines ---
        if draw_decor and hline_y is not None:
            ax1.hlines(
                y=hline_y, xmin=hline_xmin, xmax=hline_xmax,
                transform=ax1.transAxes, colors=hline_color,
                linewidth=hline_linewidth, clip_on=False
            )

        if draw_decor and vline_x is not None:
            ax1.vlines(
                x=vline_x, ymin=vline_ymin, ymax=vline_ymax,
                transform=ax1.transAxes, colors=vline_color,
                linewidth=vline_linewidth, clip_on=False
            )

        # right (inverted) y-axis ticks
        if invert_right_axis:
            ax2.set_ylim(1.0, 0)
        else:
            ax2.set_ylim(0, 1.0)
        ax2.set_yticks(np.round(np.linspace(ax2.get_ylim()[0], ax2.get_ylim()[1], 6), 2))

    def _add_legend(self, ax, cell_line: str, legend_kwargs: dict | None = None):
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        ctrl_color = self.color_map.get(cell_line, {}).get("controlled", "steelblue")
        rela_color = self.color_map.get(cell_line, {}).get("relapsed", "lightblue")

        handles = [
            Line2D([0], [0], marker="o", color="black", label="SCP", markersize=10, linestyle="None"),
            mpatches.Patch(facecolor=rela_color, edgecolor="none", label="Proportion of True Relapsed Cases"),
            mpatches.Rectangle((0, 0), 1.5, 1, facecolor="white", edgecolor="black",
                               label="Proportion of Falsely Classified Cases"),
            mpatches.Patch(facecolor=ctrl_color, edgecolor="none", label="Proportion of True Controlled Cases"),
            Line2D([], [], linestyle="None", label="Acc: C/R [Accuracy of Controlled vs Relapsed]"),
        ]

        # defaults, can be overridden by legend_kwargs
        kwargs = dict(loc="upper left", bbox_to_anchor=(1.5, 1.0), frameon=False, fontsize=28)
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        ax.legend(handles=handles, **kwargs)

    # ------------------------------------------------------------- #
    # 4.  MERGE / AGGREGATE / PLOT                                  #
    # ------------------------------------------------------------- #
    def _aggregate_by_arm(
        self,
        df: pd.DataFrame,
        group_keys: tuple[str, ...] = ("cell_line", "dose", "temperature", "time"),
        weight_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Collapse rows that share the same treatment arm (dose, time, temperature)
        within the same cell_line. Averages SCP, controlled_accuracy and
        relapsed_accuracy across plates.

        If `weight_col` is provided (e.g., number of cases per plate), a weighted
        average is used. Otherwise, simple arithmetic means are used.

        Keeps helper columns:
          - n_plates : number of plates merged into that arm
          - plates   : comma-separated list of plate ids merged
        Rows missing any of the group_keys are left as-is.
        """
        # keep only keys that exist
        group_keys = tuple(k for k in group_keys if k in df.columns)
        if len(set(("cell_line", "dose", "temperature", "time")).intersection(group_keys)) < 4:
            # meta not available -> nothing to aggregate
            return df

        # only aggregate rows with complete meta
        mask_complete = df[list(group_keys)].notna().all(axis=1)
        df_g = df.loc[mask_complete].copy()
        df_rest = df.loc[~mask_complete].copy()

        # ensure numeric for metrics
        for col in ["SCP", "controlled_accuracy", "relapsed_accuracy"]:
            if col in df_g.columns:
                df_g[col] = pd.to_numeric(df_g[col], errors="coerce")

        def agg_group(g: pd.DataFrame) -> pd.Series:
            w = None
            if weight_col and (weight_col in g.columns):
                w = pd.to_numeric(g[weight_col], errors="coerce")

            def wavg(series_name: str):
                s = pd.to_numeric(g[series_name], errors="coerce")
                if w is None or w.isna().all() or float(w.fillna(0).sum()) == 0:
                    return s.mean()
                valid = s.notna() & w.notna()
                if not valid.any():
                    return np.nan
                return np.average(s.loc[valid], weights=w.loc[valid])

            plates = ",".join(sorted(map(str, g.get("plate", pd.Series([])).dropna().astype(str).unique())))
            return pd.Series({
                "SCP": wavg("SCP"),
                "controlled_accuracy": wavg("controlled_accuracy"),
                "relapsed_accuracy": wavg("relapsed_accuracy"),
                "n_plates": g["plate"].nunique() if "plate" in g.columns else len(g),
                "plates": plates,
                "plate": plates,  # keep a 'plate' column for downstream compatibility
            })

        agg = (
            df_g.groupby(list(group_keys), dropna=False)
                .apply(agg_group)
                .reset_index()
        )

        out = pd.concat([agg, df_rest], ignore_index=True, sort=False)

        # Final NA handling (plotting expects numbers)
        for col in ["SCP", "controlled_accuracy", "relapsed_accuracy"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out[["SCP", "controlled_accuracy", "relapsed_accuracy"]] = out[
            ["SCP", "controlled_accuracy", "relapsed_accuracy"]
        ].fillna(0)

        if "plate" not in out.columns:
            out["plate"] = out.get("plates", "")
        return out

    # ------------------------------------------------------------------ #
    # 5.  MASTER: merge inputs -> (optional) aggregate -> plot           #
    # ------------------------------------------------------------------ #
    def safe_merge_and_plot(
        self,
        df_ctrl,                      # DataFrame **or** list/tuple of DataFrames
        df_rela,                      #   "
        scp_df,                       #   "
        label: str,                   # e.g. "fadu"
        tag: str,                     # e.g. "short_exp1", "short_exp2", or just "short"
        show_legend: bool = True,
        combined_kwargs: dict | None = None,
        middle_kwargs: dict | None = None,
        merge_duplicate_arms: bool = True,   # NEW
        weight_col: str | None = None,       # NEW (e.g., "n_cases")
    ):
        """
        Merges data from one **or several** experiments, optionally collapses
        duplicate treatment arms (same dose, time, temperature per cell_line),
        and produces both plots.

        Parameters
        ----------
        df_ctrl, df_rela, scp_df : (list of) pd.DataFrame
            Pass either a single DataFrame (legacy behaviour) **or**
            an iterable of DataFrames – one per experiment – which will be
            concatenated automatically.
        label : str
            Cell-line or dataset label (e.g. 'fadu').
        tag : str
            Treatment-arm or grouping tag (e.g. 'short_exp1').
            Any trailing “_expN” (where N is an integer) is stripped
            before the file names are generated.
        """

        # 1) Accept both single-DF and list-of-DF inputs
        if isinstance(df_ctrl, (list, tuple)):
            df_ctrl = pd.concat(df_ctrl, ignore_index=True)
        if isinstance(df_rela, (list, tuple)):
            df_rela = pd.concat(df_rela, ignore_index=True)
        if isinstance(scp_df, (list, tuple)):
            scp_df = pd.concat(scp_df, ignore_index=True)

        # 2) Normalise the tag: remove any “_exp\d+” suffix
        tag_clean = re.sub(r"_?exp\d+$", "", str(tag).lower())

        combined_kwargs = combined_kwargs or {}
        middle_kwargs = middle_kwargs or {}

        # 3) Merge the three sources
        df = (
            df_ctrl[["plate", "cell_line", "controlled_accuracy"]]
            .merge(
                df_rela[["plate", "cell_line", "relapsed_accuracy"]],
                on=["plate", "cell_line"], how="outer"
            )
            .merge(scp_df, on=["plate", "cell_line"], how="left")
        )

        # Optional: bring in meta-data if available
        pred_file = os.path.join(self.output_dir, f"{label}_all_predictions.csv")
        if os.path.exists(pred_file):
            meta_cols = ["plate", "cell_line", "dose", "time", "temperature"]
            df_meta = pd.read_csv(pred_file)[meta_cols].drop_duplicates()
            df = df.merge(df_meta, on=["plate", "cell_line"], how="left")

        # Aggregate by (cell_line, dose, temperature, time) if requested
        if merge_duplicate_arms:
            df = self._aggregate_by_arm(df, weight_col=weight_col)

        mask_ctrl_pure = df["SCP"].notna() & np.isclose(df["SCP"], 1.0)
        mask_rela_pure = df["SCP"].notna() & np.isclose(df["SCP"], 0.0)

        def _mean_safe(series: pd.Series) -> float | float:
            s = pd.to_numeric(series, errors="coerce").dropna()
            return float(s.mean()) if len(s) else float("nan")

        avg_ctrl_at_1 = _mean_safe(df.loc[mask_ctrl_pure, "controlled_accuracy"])
        avg_rela_at_0 = _mean_safe(df.loc[mask_rela_pure, "relapsed_accuracy"])

        n_ctrl_1 = int(mask_ctrl_pure.sum())
        n_rela_0 = int(mask_rela_pure.sum())

        def _fmt(x):  # robust printing if NaN
            return "NaN" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"

        print(f"[{label}/{tag_clean}] Avg Controlled Accuracy @ SCP=1: {_fmt(avg_ctrl_at_1)} (n={n_ctrl_1})")
        print(f"[{label}/{tag_clean}] Avg Relapsed  Accuracy @ SCP=0: {_fmt(avg_rela_at_0)} (n={n_rela_0})")
        # --------------------------------------------------------------
        if "cell_line" in df.columns:
            print(f"[{label}/{tag_clean}] --- Per Cell-Line ---")

            # prefer an available weight column; fall back to n_plates; else unweighted
            use_w = weight_col if (weight_col and weight_col in df.columns) else (
                "n_plates" if "n_plates" in df.columns else None
            )

            def _wavg(col_name: str, mask: pd.Series) -> float:
                s = pd.to_numeric(df.loc[mask, col_name], errors="coerce")
                if use_w:
                    w = pd.to_numeric(df.loc[mask, use_w], errors="coerce")
                    valid = s.notna() & w.notna()
                    if valid.any() and float(w.loc[valid].sum()) > 0:
                        return float(np.average(s.loc[valid], weights=w.loc[valid]))
                s = s.dropna()
                return float(s.mean()) if len(s) else float("nan")

            for cl in sorted(df["cell_line"].dropna().unique()):
                m_cl = df["cell_line"] == cl
                m1   = mask_ctrl_pure & m_cl     # SCP == 1
                m0   = mask_rela_pure & m_cl     # SCP == 0

                cl_ctrl = _wavg("controlled_accuracy", m1)
                cl_rela = _wavg("relapsed_accuracy", m0)
                n1 = int(m1.sum())
                n0 = int(m0.sum())

                print(f"  {cl}: Ctrl@SCP=1 {_fmt(cl_ctrl)} (n={n1}) | "
                      f"Rela@SCP=0 {_fmt(cl_rela)} (n={n0})")

        # Fill missing values AFTER aggregation
        for col in ["controlled_accuracy", "relapsed_accuracy", "SCP"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 4) Build file names (clean)
        combined_fn = f"{label}_combine_accuracy.png"
        middle_fn = f"{label}_middle_scp_accuracy.png"

        # 5) Produce the plots
        self.plot_combined_accuracy_by_SCP(
            df,
            f"{label.upper()} - Combined Accuracy ({tag_clean})",
            combined_fn,
            show_legend=show_legend,
            annotate=False,
            use_labels=False,
            **(combined_kwargs or {})
        )

        self.plot_middle_accuracy_by_SCP(
            df,
            f"{label.upper()} {tag_clean} (Middle SCP)",
            middle_fn,
            show_legend=show_legend,
            annotate=True,
            use_labels=True,
            **(middle_kwargs or {})
        )