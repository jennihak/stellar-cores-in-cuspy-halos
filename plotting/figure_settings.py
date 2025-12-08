import os
import matplotlib as mpl
import seaborn as sns

this_dir = os.path.dirname(os.path.realpath(__file__))

# Define global figure settings
mpl.rcdefaults()
mpl.rc_file(os.path.join(this_dir, "matplotlibrc"))

# Define global colormaps
cmap_times = sns.color_palette('crest', 8)          # for times
cmap_masses = sns.color_palette('flare', 7)         # for mass ratios
cmap_fiducials = sns.cubehelix_palette(10)          # for fiducials
color_orange = sns.color_palette('colorblind')[1]   # for average / hierarchical
cmap_orange = [mpl.colors.to_rgba(color_orange, alpha=0.4), mpl.colors.to_rgba(color_orange, alpha=1)]      # for contour plots


