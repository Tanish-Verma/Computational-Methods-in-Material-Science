# --1. MATPLOTLIB OVERVIEW--
# Matplotlib is built on a few core components:
# Figure: The overall window or page that holds the plot.
# Axes: The area where data is plotted (can have multiple axes in a figure).
# Axis: The individual x and y-axis with tick marks and labels.
# Artist: Everything you see on the figure (lines, text, etc.).

#--IMPORTING MATPLOTLIB--
import matplotlib.pyplot as plt  # Pyplot is the interface used to create plots
import numpy as np  # Commonly used for creating data

#--2. MAIN PLOT TYPES--
#--2.1 LINE PLOTS--
# A line plot is the most basic plot in Matplotlib. It connects a series of points with lines.
x = np.linspace(0, 10, 100)  # Generate 100 points between 0 and 10
y = np.sin(x)  # Sine function

plt.plot(x, y, color='blue', linestyle='-', linewidth=2, marker='o', markersize=6, label='Sine Wave')

# Adding labels and title
plt.xlabel('X-axis (radians)')
plt.ylabel('Y-axis (sin(x))')
plt.title('Basic Line Plot')
plt.legend()  # Displays the label inside the plot
plt.grid(True)  # Adds gridlines
plt.show()

# Attributes of plt.plot()
# x, y: Coordinates of data points.
# color: Sets the color of the line (e.g., 'blue', 'red', 'green').
# linestyle: Defines the style of the line ('-', '--', '-.', ':').
# linewidth (lw): Width of the line (default 1.0).
# marker: Marker style for data points ('o', 's', '^', 'D').
# markersize (ms): Size of the marker.
# label: Adds a label for the legend.
# alpha: Transparency level of the plot.

# 2.2 --SCATTER PLOT-- (plt.scatter)
# A scatter plot displays individual data points without connecting lines.
x = np.random.rand(50)  # Generate 50 random x values
y = np.random.rand(50)  # Generate 50 random y values
sizes = 100 * np.random.rand(50)  # Random size for scatter points

plt.scatter(x, y, s=sizes, color='red', alpha=0.5, edgecolor='black')

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Random Data')
plt.show()

# Attributes of plt.scatter()
# x, y: Coordinates of data points.
# s: Size of each marker (can be scalar or array-like).
# color (c): Color of the markers (can also be array for colormap).
# alpha: Transparency level (between 0 and 1).
# marker: Style of markers ('o', 's', '^', etc.).
# edgecolor: Color of marker edges.

# 2.3 --BAR PLOT-- (plt.bar)
# A bar plot shows rectangular bars for each data point, often used for categorical data.
categories = ['A', 'B', 'C', 'D']
values = [5, 7, 8, 4]

plt.bar(categories, values, color='purple', edgecolor='black', linewidth=1.5)

# Adding labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot Example')
plt.grid(axis='y')  # Add gridlines to y-axis only
plt.show()

# Attributes of plt.bar()
# x: Categories or positions of bars.
# height: Heights of the bars.
# width: Width of the bars (default 0.8).
# color: Fill color of the bars.
# edgecolor: Color of the edges of the bars.
# linewidth: Width of the bar edges.
# align: Aligns the bars ('center', 'edge').

# 2.4 --HISTOGRAM-- (plt.hist)
# A histogram displays the distribution of a dataset by showing the frequency of data points in specified bins.
data = np.random.randn(1000)  # Generate 1000 random points from normal distribution

plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

# Adding labels and title
plt.xlabel('Data values')
plt.ylabel('Frequency')
plt.title('Histogram Example')

plt.show()

# Attributes of plt.hist()
# x: Data points.
# bins: Number of bins (intervals) in the histogram.
# range: Tuple specifying the range (e.g., (min, max)).
# color: Color of the bars.
# edgecolor: Color of bar edges.
# density: If True, the histogram is normalized (probability distribution).
# alpha: Transparency of the bars.

# 2.5 --PIE CHART-- (plt.pie)
# A pie chart represents data as slices of a pie.
sizes = [25, 35, 20, 20]
labels = ['Category A', 'Category B', 'Category C', 'Category D']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])

# Adding title
plt.title('Pie Chart Example')

plt.show()

# Attributes of plt.pie()
# x: Data points or sizes of the slices.
# labels: Labels for each slice.
# autopct: String to format percentages on slices ('%1.1f%%').
# startangle: Starting angle of the chart (default is 0, but commonly starts at 90).
# colors: List of colors for each slice.
# explode: Array to "explode" slices outward.

# 3. --PLOT CUSTOMIZATION--
# Matplotlib provides numerous ways to customize the appearance of plots. Here are the key functions:

# 3.1 --LABELS AND TITLES--
plt.xlabel('X-axis label', fontsize=12, color='blue', labelpad=10)  # Adds an x-axis label
plt.ylabel('Y-axis label', fontsize=12, color='blue', labelpad=10)  # Adds a y-axis label
plt.title('Plot Title', fontsize=16, color='red', pad=20)  # Adds a title to the plot

# Attributes for plt.xlabel(), plt.ylabel(), plt.title()
# fontsize: Size of the font.
# color: Color of the text.
# labelpad: Distance between the label and the axis.
# pad: Distance between the title and the plot.

# 3.2 --LEGENDS--
plt.legend(loc='upper right', fontsize=12, title='Legend Title', shadow=True)

# Attributes for plt.legend()
# loc: Location of the legend ('upper left', 'lower right', etc.).
# fontsize: Size of the font in the legend.
# title: Title for the legend.
# shadow: If True, adds a shadow behind the legend.

# 3.3 --GRIDS--
plt.grid(color='gray', linestyle='--', linewidth=0.5)

# Attributes for plt.grid()
# color: Color of the grid lines.
# linestyle: Style of the grid lines ('-', '--', ':', '-.').
# linewidth: Thickness of the grid lines.

# 3.4 --TICKS--
plt.xticks([1, 2, 3, 4],['A', 'B', 'C', 'D'],rotation=45,fontsize=12,color='blue') # Setting the tick positions and labels for the x-axis
plt.yticks([10, 20, 25, 30],['Low', 'Medium', 'High', 'Very High'],fontsize=10,color='green') # Setting the tick positions and labels for the y-axis

# Attributes for xticks and yticks:
# positions: List of numerical values where the ticks should appear along the axis.
# labels: List of strings that will be displayed at the specified positions.
# rotation: Angle in degrees to rotate the labels (useful for slanted or vertical text).
# fontsize: Size of the font used for the labels.
# color: Color of the tick labels.
# fontweight: Sets the weight of the font (e.g., 'bold', 'normal').
# fontname: Specifies the font family (e.g., 'Arial', 'Times New Roman').

# 3.5 --FIGURE--

plt.figure(num=1,figsize=(8, 6),dpi=120,facecolor='lightgrey',edgecolor='blue',frameon=True,clear=True,layout='constrained')

# Attributes of plt.figure()
# num: Identifier (int or string) for the figure, useful for referencing or updating a specific figure.
# figsize: Sets width and height of the figure in inches; controls the physical size of the figure.
# dpi: Sets resolution in dots per inch; higher DPI provides finer detail but larger file sizes.
# facecolor: Background color of the figure, affecting the whole canvas area.
# edgecolor: Border color of the figure's outer edge.
# frameon: Boolean controlling visibility of the figure's frame; True shows the frame.
# clear: Clears an existing figure with the same num identifier if it exists.
# layout: Automatically adjusts subplot layout to avoid overlaps; 'constrained' is commonly used.

# 4. --SAVING FIGURES-- (plt.savefig())
# You can save plots in various formats.
plt.savefig('plot.png', dpi=300, bbox_inches='tight', format='png', transparent=True)

# Attributes of plt.savefig()
# dpi: Dots per inch (resolution).
# bbox_inches: 'tight' trims extra whitespace.
# format: Format of the file ('png', 'jpg', 'pdf', etc.).
# transparent: Makes the background transparent.
