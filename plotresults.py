
#%%
import matplotlib
import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15
# font = {'family' : 'Times New Roman', 'size': 10}
# matplotlib.rc('font', **font)
x = [2.021, 3.582, 3.173, 3.235, 3.235, 4.784, 1.697]
# x = [53, 61, 62, 64 ,90]
y = [5.258, 5.238, 3.492, 3.884, 3.330, 6.096, 5.491]
s = [446, 361, 2478, 518, 526, 441, 437]
n = ["Spiral", "COMA", "LSA-Conv", "SDConv* (ours)", "HSDConv (ours)", "FeaStNet", "Spiral++"]
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']
scatter = plt.scatter(x, y, s=s, color=colors)
# Setting x and y axes limits
plt.xlim(1.25, 6.5)  # Set the start and end points for the x-axis
plt.ylim(3, 6.5)  # Set the start and end points for the y-axis

# produce a legend with a cross section of sizes from the scatter
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.25)
legend2 = plt.legend(handles, labels, loc="lower right", title="Parameter # (K)", fontsize='small', 
                     handlelength=1.5, handletextpad=0.5, markerscale=0.65)

plt.annotate(n[0], # this is the text
                (x[0],y[0]), # this is the point to label
                color=colors[0],
                textcoords="offset points", # how to position the text
                xytext=(0,15), # distance from text to points (x,y)
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
                xytext=(0,-24), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.annotate(n[6], # this is the text
                (x[6],y[6]), # this is the point to label
                color=colors[6],
                textcoords="offset points", # how to position the text
                xytext=(0,15), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center
plt.xlabel("Time of inferring test set (s)")
plt.ylabel("L2 errors (mm)")

plt.savefig('complexity.pdf', format='pdf', bbox_inches='tight')
# plt.show()


print(x)

# %%