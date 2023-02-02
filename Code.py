from bioinfokit import analys, visuz
import pandas as pd
df = pd.read_csv('volcano.csv')
visuz.GeneExpression.volcano(df, lfc='log2FC', pv='p-value',
                       geneid="GeneNames",
    genenames=("LOC_Os06g40940.3", "LOC_Os03g03720.1"), #Add gene labels (text style) to the points
    # genenames=({"LOC_Os06g40940.3":'AB', "LOC_Os03g03720.1":'CD'}), #Add gene name (text style) to the points
                       gstyle=2, #Add gene labels or gene name (box style) to the points
                       lfc_thr=(1, 2), pv_thr=(0.05, 0.01), # Change log fold change and p value threshold
                       color=("#00239CFF", "grey", "#E10600FF"), # Change color of volcano plot
                       valpha=0.5,  #Change transparency of volcano plot
                       markerdot='*', #Change the shape of the points
                       dotsize=20, #Change the size of the points
                       sign_line=True, #Add threshold lines
                       xlm=(-6,6,1), ylm=(0,61,5), axtickfontsize=10, #Change X and Y range ticks, font size and name for tick labels
                           axtickfontname='Verdana',
                       figtype='jpg', #file format
                       plotlegend=True, legendpos='upper right',
    legendanchor=(1.46,1))

#See WD to find the result
