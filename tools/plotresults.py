
#%%
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np 

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15
# font = {'family' : 'Times New Roman', 'size': 10}
# matplotlib.rc('font', **font)
x = [2.021, 3.582, 3.173, 3.235, 3.235, 4.784, 1.697, 2.637, 2.647, 3.061]
# x = [53, 61, 62, 64 ,90]
y = [5.258, 5.238, 3.492, 3.884, 3.330, 6.096, 5.491, 4.162, 6.274, 4.544]
s = [446, 361, 2478, 518, 526, 441, 437, 2646, 650, 644]
n = ["Spiral", "COMA", "LSA-Conv", "SDConv* (ours)", "HSDConv (ours)", "FeaStNet", "Spiral++", "VCMeshConv", "VCMeshConv (B=4)", "LSA-small (B=8)"]
colors = ['orange', 'green', 'blue', 'deeppink', 'red', 'purple', 'goldenrod', "darkcyan", "lightseagreen", "cornflowerblue"]
scatter = plt.scatter(x, y, s=s, color=colors)
# Setting x and y axes limits
plt.xlim(1.25, 6.5)  # Set the start and end points for the x-axis
plt.ylim(3, 7.0)  # Set the start and end points for the y-axis

# produce a legend with a cross section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.25)
# Example sizes for the legend (representative of your dataset)
legend_sizes = [250, 500, 1000, 2000]  # Sizes of markers you want to show in the legend
legend_labels = ['250', '500', '1000', '2000']  # Corresponding labels
# Create legend handles manually
legend_handles = [Line2D([0], [0], marker='o', color='w', label=str(lbl),
                          markerfacecolor='gray', markersize=np.sqrt(size), linestyle='None', alpha=0.5)
                  for size, lbl in zip(legend_sizes, legend_labels)]

# Use these handles to create the legend
legend2 = plt.legend(handles=legend_handles, title="Parameter # (K)", fontsize='small', loc="right",
           handlelength=1.5, handletextpad=0.5, markerscale=1.0)

# legend2 = plt.legend(handles, labels, loc="lower right", title="Parameter # (K)", fontsize='small', 
#                      handlelength=1.5, handletextpad=0.5, markerscale=0.55)

plt.annotate(n[0], # this is the text
                (x[0],y[0]), # this is the point to label
                color=colors[0],
                textcoords="offset points", # how to position the text
                xytext=(10,16), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[1], # this is the text
                (x[1],y[1]), # this is the point to label
                color=colors[1],
                textcoords="offset points", # how to position the text
                xytext=(0, 15), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[2], # this is the text
                (x[2],y[2]), # this is the point to label
                color=colors[2],
                textcoords="offset points", # how to position the text
                xytext=(64,0), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[3], # this is the text
                (x[3],y[3]), # this is the point to label
                fontweight='bold',
                color=colors[3],
                textcoords="offset points", # how to position the text
                xytext=(68,0), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[4], # this is the text
                (x[4],y[4]), # this is the point to label
                fontweight='bold',
                color=colors[4],
                textcoords="offset points", # how to position the text
                xytext=(75,-5), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[5], # this is the text
                (x[5],y[5]), # this is the point to label
                color=colors[5],
                textcoords="offset points", # how to position the text
                xytext=(0,16), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[6], # this is the text
                (x[6],y[6]), # this is the point to label
                color=colors[6],
                textcoords="offset points", # how to position the text
                xytext=(0,15), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[7], # this is the text
                (x[7],y[7]), # this is the point to label
                color=colors[7],
                textcoords="offset points", # how to position the text
                xytext=(-28,30), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[8], # this is the text
                (x[8],y[8]), # this is the point to label
                color=colors[7],
                textcoords="offset points", # how to position the text
                xytext=(0,18), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[9], # this is the text
                (x[9],y[9]), # this is the point to label
                color=colors[9],
                textcoords="offset points", # how to position the text
                xytext=(10,18), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.xlabel("Time of inferring test set (s)")
plt.ylabel("L2 errors (mm)")

plt.savefig('../images/complexity1.png', dpi=300, bbox_inches='tight', transparent=False)
plt.savefig('../images/complexity.pdf', format='pdf', bbox_inches='tight')
# plt.show()


print(x)

# %%