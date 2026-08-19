import numpy as np
from scipy.stats import pearsonr, spearmanr, sem
from scipy.signal import argrelmax
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import Span, ColumnDataSource
from bokeh.transform import linear_cmap
from bokeh.layouts import layout


def fragment_distribution(length_dist, title=None, display=True, save_fn=None):
    """
    Plot the fragment distribution
    """
    
    # Determine total distribution
    max_length = max(len(length_dist[chrom]) for chrom in length_dist)
    fragment_lengths = np.zeros(max_length, dtype=int)
    for chrom in length_dist:
        fragment_lengths[:len(length_dist[chrom])] = length_dist[chrom]

    # Creat Axes object if not given
    TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    p = figure(tools=TOOLS, title=title, x_axis_label="Read length", y_axis_label="Density")
    
    # Plot fragment lengths
    p.line(np.arange(len(fragment_lengths)), fragment_lengths, line_width=2)

    # Plot local maxima
    for maxima in argrelmax(fragment_lengths, order=25)[0]:
        vline = Span(location=maxima, dimension='height', line_color='black', line_width=1)
        p.renderers.extend([vline])
    
    # Save of display plot
    if save_fn is not None:
        output_file(save_fn)
        save(p)
    if display:
        show(p)

    return p


def cnv_plot(cnv_object, title=None, display=True, save_fn=None):
    """
    """

    # Check genome is present
    if cnv_object.genome is None:
        raise AttributeError("Genome file not given")
        
    # Create figure
    TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    p = figure(tools=TOOLS, title=title, )

    # Create color map
    palette = ['#660220', '#b01b2f', '#d46151', '#f2a585',
               '#fcdbc8', '#f7f7f7', '#d2e5ef', '#94c5dd',
               '#4794c1', '#2668aa', '#083160'][::-1]
    mapper = linear_cmap(field_name='y', palette=palette, low=-0.2, high=0.2)

    # Plot bins
    for chrom in cnv_object.segs:
        chrom_ind = cnv_object.bins.loc[:,"seqnames"].values == chrom
        source = ColumnDataSource(dict(x=cnv_object.bins.loc[chrom_ind,"start"].values + cnv_object.chrom_shift[chrom],
                                        y=cnv_object.bins.loc[chrom_ind,"ratios"].values))

        p.circle(x='x', y='y', line_color=mapper, color=mapper, fill_alpha=1, size=1, source=source)

    # Plot segments
    for chrom in cnv_object.segs:
        # Check that chrom in in genome
        try:
            shift = cnv_object.chrom_shift[chrom]
        except KeyError:
            continue

        # Positive
        for i in cnv_object.segs[chrom]["pos"]:
            p.line([i.start+shift, i.end+shift],
                   [i.value, i.value],
                   color="black")

        # Negative
        for i in cnv_object.segs[chrom]["neg"]:
            p.line([i.start+shift, i.end+shift],
                   [i.value, i.value],
                   color="black")
                    
    # Plot zero lines
    hline = Span(location=0, dimension='width', line_color='darkgrey', line_width=1)
    p.renderers.extend([hline])
    
    # Plot chromosome lines
    last_chrom = list(cnv_object.genome.keys())[-1]
    for chrom in cnv_object.genome:
        vline = Span(location=cnv_object.chrom_shift[chrom], dimension='height', line_color='black', line_width=1)
        p.renderers.extend([vline])
    vline = Span(location=cnv_object.genome[last_chrom] + cnv_object.chrom_shift[last_chrom], dimension='height', line_color='black', line_width=1)
    p.renderers.extend([vline])
                
    # Plot centro lines
    if cnv_object.centro is not None:
        for chrom in cnv_object.genome:
            vline = Span(location=cnv_object.centro[chrom] + cnv_object.chrom_shift[chrom], dimension='height', line_color='grey', line_width=0.75, line_dash="4 4")
            p.renderers.extend([vline])
    
    # Plot chromosome names
    text_x = []
    for chrom in cnv_object.genome:
        text_x.append(int(cnv_object.chrom_shift[chrom] + (cnv_object.genome[chrom] / 2)))
    p.xaxis.ticker = text_x
    p.xaxis.major_label_overrides = {text_x[i]:c for i, c in enumerate(list(cnv_object.genome.keys()))}
    p.xaxis.major_label_orientation = np.pi/2
    p.yaxis.major_label_orientation = "vertical"
    
    # Set attributes
    p.xaxis.bounds = (0, cnv_object.genome[last_chrom] + cnv_object.chrom_shift[last_chrom])
    
    # Save of display plot
    if save_fn != None:
        output_file(save_fn)
        save(p)
    if display:
        show(p)


def plot_profile(frag_profile, genome, chrom_shift, centro, title=None, display=True, save_fn=None):
    """
    """

    # Create figure
    TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
    p = figure(tools=TOOLS, title=title)

    # Plot fragment lengths
    for chrom in frag_profile:
        # Check that chrom in in genome
        try:
            shift = chrom_shift[chrom]
        except KeyError:
            continue

        p.line(frag_profile[chrom].index.values+shift, frag_profile[chrom].values, line_width=2)

    # Plot zero lines
    hline = Span(location=0, dimension='width', line_color='darkgrey', line_width=1)
    p.renderers.extend([hline])
    
    # Plot chromosome lines
    last_chrom = list(genome.keys())[-1]
    for chrom in genome:
        vline = Span(location=chrom_shift[chrom], dimension='height', line_color='black', line_width=1)
        p.renderers.extend([vline])
    vline = Span(location=genome[last_chrom] + chrom_shift[last_chrom], dimension='height', line_color='black', line_width=1)
    p.renderers.extend([vline])
                
    # Plot centro lines
    if centro is not None:
        for chrom in genome:
            vline = Span(location=centro[chrom] + chrom_shift[chrom], dimension='height', line_color='grey', line_width=0.75, line_dash="4 4")
            p.renderers.extend([vline])
    
    # Plot chromosome names
    text_x = []
    for chrom in genome:
        text_x.append(int(chrom_shift[chrom] + (genome[chrom] / 2)))
    p.xaxis.ticker = text_x
    p.xaxis.major_label_overrides = {text_x[i]:c for i, c in enumerate(list(genome.keys()))}
    p.xaxis.major_label_orientation = np.pi/2
    p.yaxis.major_label_orientation = "vertical"
    
    # Set attributes
    p.xaxis.bounds = (0, genome[last_chrom] + chrom_shift[last_chrom])
    
    # Save of display plot
    if save_fn is not None:
        output_file(save_fn)
        save(p)
    if display:
        show(p)

    return p


def plot_dashboard(cnv_object1, cnv_object2, cnv_object3, length_dist, frag_profile, display=True, save_fn=None):
    """
    """

    # Plot length dist
    p1 = fragment_distribution(length_dist, title=None, display=False)

    # Plot cnvs
    p2 = cnv_plot(cnv_object1, title=None, display=False)
    p3 = cnv_plot(cnv_object2, title=None, display=False)
    p4 = cnv_plot(cnv_object3, title=None, display=False)

    # Plot fragment profile
    p5 = plot_profile(frag_profile, cnv_object1.genome, cnv_object1.chrom_shift,
                      cnv_object1.centro, title=None, display=False)

    # Create grid
    l = layout([
                [p1],
                [p2],
                [p3],
                [p4],
                [p5],
               ], sizing_mode="stretch_both")

    # Save of display plot
    if save_fn is not None:
        output_file(save_fn)
        save(l)
    if display:
        show(l)
