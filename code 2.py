#!/usr/bin/env python3
"""
Professional Volcano Plot Generator - Custom Implementation
Matches bioinfokit functionality with additional customization options
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


class VolcanoPlot:
    """Create publication-ready volcano plots with full customization"""

    def __init__(self, csv_file, lfc='log2FC', pv='p-value', geneid='GeneNames'):
        """
        Initialize the VolcanoPlot object

        Parameters:
        -----------
        csv_file : str
            Path to the CSV file containing gene expression data
        lfc : str
            Column name for log2 fold change
        pv : str
            Column name for p-value
        geneid : str
            Column name for gene identifiers
        """
        self.df = pd.read_csv(csv_file)
        self.lfc_col = lfc
        self.pv_col = pv
        self.geneid_col = geneid

        # Calculate -log10(p-value) for plotting
        self.df['-log10(pv)'] = -np.log10(self.df[self.pv_col])

        print(f"Loaded {len(self.df)} genes from {csv_file}")

    def plot(self,
             genenames=None,
             gstyle=2,
             lfc_thr=(1, 2),
             pv_thr=(0.05, 0.01),
             color=("#00239CFF", "grey", "#E10600FF"),
             valpha=0.5,
             markerdot='*',
             dotsize=20,
             sign_line=True,
             xlm=(-6, 6, 1),
             ylm=(0, 61, 5),
             axtickfontsize=10,
             axtickfontname='Verdana',
             figsize=(12, 8),
             dpi=300,
             figtype='png',
             plotlegend=True,
             legendpos='upper right',
             legendanchor=(1.46, 1),
             save_path=None):
        """
        Create a volcano plot with bioinfokit-style customization

        Parameters:
        -----------
        genenames : tuple or dict, optional
            Gene names to highlight as text or with boxes
            - tuple: ("GENE1", "GENE2") for text labels
            - dict: {"GENE1": "Label1", "GENE2": "Label2"} for custom labels
        gstyle : int
            Style for gene labels:
            - 0: No labels
            - 1: Text style (direct label on point)
            - 2: Box style (box with arrow to point)
        lfc_thr : tuple
            (primary_threshold, secondary_threshold) for log2FC
            Primary is main significance line, secondary is more stringent
        pv_thr : tuple
            (primary_threshold, secondary_threshold) for p-value
        color : tuple
            (down_regulated_color, not_significant_color, up_regulated_color)
        valpha : float
            Transparency/alpha value (0-1)
        markerdot : str
            Marker style: 'o' (circle), '*' (star), '^' (triangle), 's' (square), etc.
        dotsize : int
            Size of the marker points
        sign_line : bool
            Whether to show threshold lines
        xlm : tuple
            (min, max, step) for x-axis limits
        ylm : tuple
            (min, max, step) for y-axis limits
        axtickfontsize : int
            Font size for axis tick labels
        axtickfontname : str
            Font name for axis tick labels
        figsize : tuple
            Figure size in inches (width, height)
        dpi : int
            Resolution in dots per inch
        figtype : str
            File format: 'png', 'jpg', 'pdf', 'svg'
        plotlegend : bool
            Whether to display legend
        legendpos : str
            Legend position: 'upper right', 'upper left', 'lower right', etc.
        legendanchor : tuple
            (x, y) anchor position for legend outside plot
        save_path : str, optional
            Custom path to save the figure

        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Extract thresholds
        lfc_thr1, lfc_thr2 = lfc_thr
        pv_thr1, pv_thr2 = pv_thr
        pv_thr1_log = -np.log10(pv_thr1)
        pv_thr2_log = -np.log10(pv_thr2)

        # Classify genes based on thresholds
        down_reg = (self.df[self.lfc_col] < -lfc_thr1) & (self.df[self.pv_col] < pv_thr1)
        up_reg = (self.df[self.lfc_col] > lfc_thr1) & (self.df[self.pv_col] < pv_thr1)
        not_sig = ~(down_reg | up_reg)

        color_down, color_not_sig, color_up = color

        # Plot not significant genes first (background)
        ax.scatter(self.df[not_sig][self.lfc_col],
                   self.df[not_sig]['-log10(pv)'],
                   c=color_not_sig, alpha=valpha, s=dotsize,
                   marker=markerdot, label='Not Significant',
                   edgecolors='none', zorder=1)

        # Plot down-regulated genes
        ax.scatter(self.df[down_reg][self.lfc_col],
                   self.df[down_reg]['-log10(pv)'],
                   c=color_down, alpha=valpha, s=dotsize,
                   marker=markerdot, label='Down-regulated',
                   edgecolors='none', zorder=2)

        # Plot up-regulated genes
        ax.scatter(self.df[up_reg][self.lfc_col],
                   self.df[up_reg]['-log10(pv)'],
                   c=color_up, alpha=valpha, s=dotsize,
                   marker=markerdot, label='Up-regulated',
                   edgecolors='none', zorder=2)

        # Add threshold lines
        if sign_line:
            # Log2FC thresholds
            ax.axvline(-lfc_thr1, color='grey', linestyle='--', linewidth=1.5,
                       alpha=0.7, zorder=0)
            ax.axvline(lfc_thr1, color='grey', linestyle='--', linewidth=1.5,
                       alpha=0.7, zorder=0)

            # P-value threshold
            ax.axhline(pv_thr1_log, color='grey', linestyle='--', linewidth=1.5,
                       alpha=0.7, zorder=0)

            # Secondary thresholds (lighter)
            if lfc_thr2 is not None:
                ax.axvline(-lfc_thr2, color='grey', linestyle=':', linewidth=1,
                           alpha=0.4, zorder=0)
                ax.axvline(lfc_thr2, color='grey', linestyle=':', linewidth=1,
                           alpha=0.4, zorder=0)

            if pv_thr2 is not None:
                ax.axhline(pv_thr2_log, color='grey', linestyle=':', linewidth=1,
                           alpha=0.4, zorder=0)

        # Add gene labels
        if genenames is not None and gstyle > 0:
            self._add_gene_labels(ax, genenames, gstyle, lfc_thr1, pv_thr1)

        # Set axis limits and ticks
        ax.set_xlim(xlm[0], xlm[1])
        ax.set_ylim(ylm[0], ylm[1])
        ax.set_xticks(np.arange(xlm[0], xlm[1] + xlm[2], xlm[2]))
        ax.set_yticks(np.arange(ylm[0], ylm[1] + ylm[2], ylm[2]))

        # Labels and formatting
        ax.set_xlabel('log₂(Fold Change)', fontsize=13, fontweight='bold',
                      fontname=axtickfontname)
        ax.set_ylabel('-log₁₀(p-value)', fontsize=13, fontweight='bold',
                      fontname=axtickfontname)
        ax.set_title('Volcano Plot', fontsize=16, fontweight='bold', pad=20)

        # Tick formatting
        ax.tick_params(labelsize=axtickfontsize, width=1.5, length=6)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(axtickfontname)
            label.set_fontsize(axtickfontsize)

        # Spine styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Legend
        if plotlegend:
            ax.legend(loc=legendpos, bbox_to_anchor=legendanchor,
                      fontsize=11, framealpha=0.95, edgecolor='black',
                      fancybox=True, shadow=True)

        # Grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

        plt.tight_layout()

        # Save figure
        if save_path is None:
            save_path = f'volcano_plot.{figtype}'

        plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', format=figtype)
        print(f"Plot saved to {save_path}")

        return fig, ax

    def _add_gene_labels(self, ax, genenames, gstyle, lfc_thr, pv_thr):
        """
        Add gene labels to the plot

        Parameters:
        -----------
        ax : matplotlib axis
            The plot axis
        genenames : tuple or dict
            Gene names to label
        gstyle : int
            Label style (1=text, 2=box)
        lfc_thr : float
            Log2FC threshold for significance classification
        pv_thr : float
            P-value threshold for significance classification
        """

        # Convert tuple to dict if needed
        if isinstance(genenames, (list, tuple)):
            genenames = {gene: gene for gene in genenames}

        # Get colors for genes
        for gene_id, label in genenames.items():
            gene_data = self.df[self.df[self.geneid_col] == gene_id]

            if len(gene_data) == 0:
                print(f"Warning: Gene '{gene_id}' not found in dataset")
                continue

            x = gene_data[self.lfc_col].values[0]
            y = gene_data['-log10(pv)'].values[0]

            # Determine if gene is significant
            is_down = (x < -lfc_thr) & (gene_data[self.pv_col].values[0] < pv_thr)
            is_up = (x > lfc_thr) & (gene_data[self.pv_col].values[0] < pv_thr)

            if is_down:
                color = "#00239CFF"
            elif is_up:
                color = "#E10600FF"
            else:
                color = "grey"

            if gstyle == 1:
                # Text style - direct label
                ax.annotate(label, xy=(x, y), xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white', alpha=0.7,
                                      edgecolor='none'),
                            zorder=5)

            elif gstyle == 2:
                # Box style - box with arrow
                ax.annotate(label, xy=(x, y), xytext=(20, 20),
                            textcoords='offset points',
                            fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.5',
                                      facecolor=color, alpha=0.8,
                                      edgecolor='black', linewidth=1.5),
                            arrowprops=dict(arrowstyle='->',
                                            connectionstyle='arc3,rad=0.3',
                                            color='black', lw=1.5),
                            color='white' if gstyle == 2 else 'black',
                            zorder=5)

    def get_gene_info(self, gene_name):
        """Get information about a specific gene"""
        gene = self.df[self.df[self.geneid_col] == gene_name]
        if len(gene) > 0:
            return gene.iloc[0]
        else:
            print(f"Gene '{gene_name}' not found")
            return None

    def list_significant_genes(self, lfc_thr=1, pv_thr=0.05, direction='all', top_n=20):
        """
        List significant genes

        Parameters:
        -----------
        lfc_thr : float
            Log2FC threshold
        pv_thr : float
            P-value threshold
        direction : str
            'all', 'up', or 'down'
        top_n : int
            Number of top genes to return
        """
        down_reg = (self.df[self.lfc_col] < -lfc_thr) & (self.df[self.pv_col] < pv_thr)
        up_reg = (self.df[self.lfc_col] > lfc_thr) & (self.df[self.pv_col] < pv_thr)

        if direction == 'all':
            result = self.df[down_reg | up_reg]
        elif direction == 'up':
            result = self.df[up_reg]
        elif direction == 'down':
            result = self.df[down_reg]
        else:
            result = self.df

        return result.sort_values(self.pv_col).head(top_n)


# ============================================================================
# EXAMPLE USAGE - Matches your bioinfokit code
# ============================================================================

if __name__ == "__main__":

    # Initialize with your data
    df = pd.read_csv('volcano.csv')

    volcano = VolcanoPlot('volcano.csv', lfc='log2FC', pv='p-value', geneid='GeneNames')

    # Create volcano plot matching your bioinfokit style
    fig, ax = volcano.plot(
        genenames=("LOC_Os06g40940.3", "LOC_Os03g03720.1"),  # Gene labels (text style)
        # genenames={"LOC_Os06g40940.3": "AB", "LOC_Os03g03720.1": "CD"},  # Custom labels
        gstyle=2,  # Box style with arrows
        lfc_thr=(1, 2),  # Primary and secondary log2FC thresholds
        pv_thr=(0.05, 0.01),  # Primary and secondary p-value thresholds
        color=("#00239CFF", "grey", "#E10600FF"),  # Down, not sig, up colors
        valpha=0.5,  # Transparency
        markerdot='*',  # Marker shape (change to 'o' for circles)
        dotsize=20,  # Marker size
        sign_line=True,  # Show threshold lines
        xlm=(-6, 6, 1),  # X-axis: min, max, step
        ylm=(0, 61, 5),  # Y-axis: min, max, step
        axtickfontsize=10,
        axtickfontname='Verdana',
        figtype='png',  # Output format
        plotlegend=True,
        legendpos='upper right',
        legendanchor=(1.46, 1),
        save_path=' volcano_plot.png'
    )

    # Display top significant genes
    print("\n=== Top 10 Significant Genes ===")
    top_genes = volcano.list_significant_genes(lfc_thr=1, pv_thr=0.05, top_n=10)
    print(top_genes[['GeneNames', 'log2FC', 'p-value']])

    # Get info for a specific gene
    print("\n=== Gene Information ===")
    gene_info = volcano.get_gene_info('LOC_Os06g40940.3')
    if gene_info is not None:
        print(f"Gene: LOC_Os06g40940.3")
        print(f"log2FC: {gene_info['log2FC']:.4f}")
        print(f"p-value: {gene_info['p-value']:.2e}")

    plt.show()


# !/usr/bin/env python3
"""
Interactive Volcano Plot Generator with Hover Gene Names
Uses Plotly for interactive visualizations with gene name hover tooltips
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


class InteractiveVolcanoPlot:
    """Create interactive volcano plots with gene name hover tooltips"""

    def __init__(self, csv_file, lfc='log2FC', pv='p-value', geneid='GeneNames'):
        """
        Initialize the InteractiveVolcanoPlot object

        Parameters:
        -----------
        csv_file : str
            Path to the CSV file containing gene expression data
        lfc : str
            Column name for log2 fold change
        pv : str
            Column name for p-value
        geneid : str
            Column name for gene identifiers
        """
        self.df = pd.read_csv(csv_file)
        self.lfc_col = lfc
        self.pv_col = pv
        self.geneid_col = geneid

        # Calculate -log10(p-value) for plotting
        self.df['-log10(pv)'] = -np.log10(self.df[self.pv_col])

        print(f"Loaded {len(self.df)} genes from {csv_file}")

    def plot(self,
             selected_genes=None,
             lfc_thr=(1, 2),
             pv_thr=(0.05, 0.01),
             color=("#00239CFF", "grey", "#E10600FF"),
             valpha=0.6,
             markerdot='circle',
             dotsize=8,
             sign_line=True,
             xlm=(-6, 6),
             ylm=(0, 60),
             axtickfontsize=12,
             axtickfontname='Verdana',
             figsize=(1000, 700),
             title='Volcano Plot',
             save_path=None,
             html_path=None):
        """
        Create an interactive volcano plot with hover gene names

        Parameters:
        -----------
        selected_genes : list or tuple, optional
            Specific genes to highlight with larger markers and boxes
        lfc_thr : tuple
            (primary_threshold, secondary_threshold) for log2FC
        pv_thr : tuple
            (primary_threshold, secondary_threshold) for p-value
        color : tuple
            (down_regulated_color, not_significant_color, up_regulated_color)
        valpha : float
            Transparency/alpha value (0-1)
        markerdot : str
            Marker style: 'circle', 'star', 'diamond', 'square', 'triangle', etc.
        dotsize : int
            Size of the marker points
        sign_line : bool
            Whether to show threshold lines
        xlm : tuple
            (min, max) for x-axis limits
        ylm : tuple
            (min, max) for y-axis limits
        axtickfontsize : int
            Font size for axis tick labels
        axtickfontname : str
            Font name for axis tick labels
        figsize : tuple
            Figure size in pixels (width, height)
        title : str
            Plot title
        save_path : str, optional
            Path to save as static image (requires kaleido)
        html_path : str, optional
            Path to save interactive HTML file

        Returns:
        --------
        fig : plotly figure object
        """

        # Extract thresholds
        lfc_thr1, lfc_thr2 = lfc_thr
        pv_thr1, pv_thr2 = pv_thr
        pv_thr1_log = -np.log10(pv_thr1)
        pv_thr2_log = -np.log10(pv_thr2)

        # Classify genes based on thresholds
        down_reg = (self.df[self.lfc_col] < -lfc_thr1) & (self.df[self.pv_col] < pv_thr1)
        up_reg = (self.df[self.lfc_col] > lfc_thr1) & (self.df[self.pv_col] < pv_thr1)
        not_sig = ~(down_reg | up_reg)

        color_down, color_not_sig, color_up = color

        # Create figure
        fig = go.Figure()

        # Convert alpha to rgba
        alpha_int = int(valpha * 255)

        # Helper function to convert hex to rgba
        def hex_to_rgba(hex_color, alpha):
            # Handle named colors
            color_map = {
                'grey': '#808080',
                'gray': '#808080',
                'red': '#FF0000',
                'blue': '#0000FF',
                'green': '#008000',
                'black': '#000000',
                'white': '#FFFFFF'
            }

            if hex_color.lower() in color_map:
                hex_color = color_map[hex_color.lower()]

            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 8:  # RRGGBBAA format
                hex_color = hex_color[:6]

            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f'rgba({r},{g},{b},{alpha})'
            except ValueError:
                # Fallback to grey if color parsing fails
                return f'rgba(128,128,128,{alpha})'

        # Plot not significant genes
        not_sig_data = self.df[not_sig]
        fig.add_trace(go.Scatter(
            x=not_sig_data[self.lfc_col],
            y=not_sig_data['-log10(pv)'],
            mode='markers',
            marker=dict(
                size=dotsize,
                color=hex_to_rgba(color_not_sig, valpha),
                symbol=markerdot,
                line=dict(width=0)
            ),
            text=not_sig_data[self.geneid_col],
            hovertemplate='<b>%{text}</b><br>' +
                          f'{self.lfc_col}: %{{x:.4f}}<br>' +
                          f'{self.pv_col}: %{{customdata}}<br>' +
                          '<extra></extra>',
            customdata=not_sig_data[self.pv_col],
            name='Not Significant',
            showlegend=True
        ))

        # Plot down-regulated genes
        down_reg_data = self.df[down_reg]
        fig.add_trace(go.Scatter(
            x=down_reg_data[self.lfc_col],
            y=down_reg_data['-log10(pv)'],
            mode='markers',
            marker=dict(
                size=dotsize,
                color=hex_to_rgba(color_down, valpha),
                symbol=markerdot,
                line=dict(width=0)
            ),
            text=down_reg_data[self.geneid_col],
            hovertemplate='<b>%{text}</b><br>' +
                          f'{self.lfc_col}: %{{x:.4f}}<br>' +
                          f'{self.pv_col}: %{{customdata}}<br>' +
                          '<extra></extra>',
            customdata=down_reg_data[self.pv_col],
            name='Down-regulated',
            showlegend=True
        ))

        # Plot up-regulated genes
        up_reg_data = self.df[up_reg]
        fig.add_trace(go.Scatter(
            x=up_reg_data[self.lfc_col],
            y=up_reg_data['-log10(pv)'],
            mode='markers',
            marker=dict(
                size=dotsize,
                color=hex_to_rgba(color_up, valpha),
                symbol=markerdot,
                line=dict(width=0)
            ),
            text=up_reg_data[self.geneid_col],
            hovertemplate='<b>%{text}</b><br>' +
                          f'{self.lfc_col}: %{{x:.4f}}<br>' +
                          f'{self.pv_col}: %{{customdata}}<br>' +
                          '<extra></extra>',
            customdata=up_reg_data[self.pv_col],
            name='Up-regulated',
            showlegend=True
        ))

        # Add threshold lines
        if sign_line:
            # Primary log2FC threshold lines
            fig.add_vline(x=-lfc_thr1, line_dash="dash", line_color="grey",
                          line_width=2, opacity=0.7, annotation_text="")
            fig.add_vline(x=lfc_thr1, line_dash="dash", line_color="grey",
                          line_width=2, opacity=0.7, annotation_text="")

            # Primary p-value threshold line
            fig.add_hline(y=pv_thr1_log, line_dash="dash", line_color="grey",
                          line_width=2, opacity=0.7, annotation_text="")

            # Secondary thresholds (lighter)
            if lfc_thr2 is not None:
                fig.add_vline(x=-lfc_thr2, line_dash="dot", line_color="grey",
                              line_width=1, opacity=0.4)
                fig.add_vline(x=lfc_thr2, line_dash="dot", line_color="grey",
                              line_width=1, opacity=0.4)

            if pv_thr2 is not None:
                fig.add_hline(y=pv_thr2_log, line_dash="dot", line_color="grey",
                              line_width=1, opacity=0.4)

        # Highlight selected genes with annotations (arrows and names)
        if selected_genes is not None:
            if isinstance(selected_genes, str):
                selected_genes = [selected_genes]

            for gene in selected_genes:
                gene_data = self.df[self.df[self.geneid_col] == gene]

                if len(gene_data) == 0:
                    print(f"Warning: Gene '{gene}' not found in dataset")
                    continue

                x = gene_data[self.lfc_col].values[0]
                y = gene_data['-log10(pv)'].values[0]

                # Add annotation with arrow pointing to gene
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=gene,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='black',
                    ax=40,  # Horizontal distance for arrow
                    ay=-40,  # Vertical distance for arrow
                    bgcolor='rgba(243, 156, 18, 0.9)',
                    bordercolor='black',
                    borderwidth=1.5,
                    borderpad=5,
                    font=dict(size=11, color='white', family=axtickfontname, weight='bold')
                )

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, family=axtickfontname)),
            xaxis=dict(
                title=f'{self.lfc_col}',
                title_font=dict(size=14, family=axtickfontname),
                tickfont=dict(size=axtickfontsize, family=axtickfontname),
                range=xlm,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.3)'
            ),
            yaxis=dict(
                title=f'-log₁₀({self.pv_col})',
                title_font=dict(size=14, family=axtickfontname),
                tickfont=dict(size=axtickfontsize, family=axtickfontname),
                range=ylm,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.3)'
            ),
            width=figsize[0],
            height=figsize[1],
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=11, family=axtickfontname)
            ),
            margin=dict(l=80, r=150, t=80, b=80)
        )

        # Save as HTML (interactive)
        if html_path is None:
            html_path = 'volcano_plot_interactive.html'

        fig.write_html(html_path)
        print(f"Interactive plot saved to {html_path}")

        # Save as static image if path provided
        if save_path is not None:
            try:
                fig.write_image(save_path, width=figsize[0], height=figsize[1], scale=2)
                print(f"Static image saved to {save_path}")
            except Exception as e:
                print(f"Note: To save as PNG/PDF, install kaleido: pip install kaleido")
                print(f"Error: {e}")

        return fig

    def get_gene_info(self, gene_name):
        """Get information about a specific gene"""
        gene = self.df[self.df[self.geneid_col] == gene_name]
        if len(gene) > 0:
            return gene.iloc[0]
        else:
            print(f"Gene '{gene_name}' not found")
            return None

    def list_significant_genes(self, lfc_thr=1, pv_thr=0.05, direction='all', top_n=20):
        """
        List significant genes

        Parameters:
        -----------
        lfc_thr : float
            Log2FC threshold
        pv_thr : float
            P-value threshold
        direction : str
            'all', 'up', or 'down'
        top_n : int
            Number of top genes to return
        """
        down_reg = (self.df[self.lfc_col] < -lfc_thr) & (self.df[self.pv_col] < pv_thr)
        up_reg = (self.df[self.lfc_col] > lfc_thr) & (self.df[self.pv_col] < pv_thr)

        if direction == 'all':
            result = self.df[down_reg | up_reg]
        elif direction == 'up':
            result = self.df[up_reg]
        elif direction == 'down':
            result = self.df[down_reg]
        else:
            result = self.df

        return result.sort_values(self.pv_col).head(top_n)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize with your data
    volcano = InteractiveVolcanoPlot(
        'volcano.csv',
        lfc='log2FC',
        pv='p-value',
        geneid='GeneNames'
    )

    # Display top significant genes
    print("\n=== Top 10 Significant Genes ===")
    top_genes = volcano.list_significant_genes(lfc_thr=1, pv_thr=0.05, top_n=10)
    print(top_genes[['GeneNames', 'log2FC', 'p-value']])

    # Create interactive volcano plot with hover
    print("\n=== Creating Interactive Volcano Plot ===")
    fig = volcano.plot(
        selected_genes=['LOC_Os12g42876.1'],  # Highlight these genes
        lfc_thr=(1, 2),
        pv_thr=(0.05, 0.01),
        color=("#00239CFF", "grey", "#E10600FF"),
        valpha=0.6,
        markerdot='circle',
        dotsize=8,
        sign_line=True,
        xlm=(-6, 6),
        ylm=(0, 60),
        axtickfontsize=12,
        axtickfontname='Verdana',
        figsize=(1000, 700),
        title='Interactive Volcano Plot',
        html_path=' volcano_plot_interactive.html'
    )

    # Show in browser (if running in Jupyter or interactive environment)
    fig.show()

    print("\n✅ Open the HTML file in your browser to interact with the plot!")
    print("   Features:")
    print("   - Hover over points to see gene names, log2FC, and p-value")
    print("   - Zoom by clicking and dragging")
    print("   - Pan by shift-clicking and dragging")
    print("   - Click legend items to toggle categories")
    print("   - Download as PNG using camera icon in top right")
