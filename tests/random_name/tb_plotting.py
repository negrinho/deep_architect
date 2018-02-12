### plotting 
import os
if "DISPLAY" not in os.environ:# or os.environ["DISPLAY"] == ':0.0':
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# TODO: edit with retrieve_vars
class LinePlot:
    def __init__(self, title=None, xlabel=None, ylabel=None):
        self.data = []
        self.cfg = {'title' : title,
                    'xlabel' : xlabel,
                    'ylabel' : ylabel}
    
    def add_line(self, xs, ys, label=None, err=None):
        d = {"xs" : xs, 
             "ys" : ys, 
             "label" : label,
             "err" : err}
        self.data.append(d)

    def plot(self, show=True, fpath=None):
        f = plt.figure()
        for d in self.data:
             plt.errorbar(d['xs'], d['ys'], yerr=d['err'], label=d['label'])
            
        plt.title(self.cfg['title'])
        plt.xlabel(self.cfg['xlabel'])
        plt.ylabel(self.cfg['ylabel'])
        plt.legend(loc='best')

        if fpath != None:
            f.savefig(fpath, bbox_inches='tight')
        if show:
            f.show()
        return f

# TODO: I guess that they should be single figures.
class GridPlot:
    def __init__(self):
        pass
    
    def add_plot(self):
        pass
    
    def new_row(self):
        pass
    
    def plot(self, show=True, fpath=None):
        pass


# what is the difference between axis and figures. plot, axis, figures

### TODO: 
class BarPlot:
    def __init__(self):
        pass


## TODO: perhaps make it easy to do references.
class GridPlot:
    def __init__(self):
        self.xs = []


# FFMpegFileWriter

class PlotAnimation:
    pass

# class 


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig, ax = plt.subplots()

# x = np.arange(0, 2*np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))


# def animate(i):
#     line.set_ydata(np.sin(x + i/10.0))  # update the data
#     return line,


# # Init only required for blitting to give a clean slate.
# def init():
#     line.set_ydata(np.ma.array(x, mask=True))
#     return line,

# ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
#                               interval=25, blit=True)
# plt.show()



class Animation:
    def __init__(self):
        self.xs = []
    
    def add_frame(self, x):
        self.xs.append(x)
    
    def plot(self, fps, show=True, fpath=None):




# NOTE: for example, you can keep the labels and do something with the 
# the rest of the model. 
# you can do a lot of thing. 


# another type of graph that is common.
# which is.
    
# TODO: log plots vs non log plots. more properties to change.
# TODO: updatable figure.

# TODO: add histogram plo, and stuff like that.


### data to latex

# perhaps add hlines if needed lines and stuff like that. this can be done.
# NOTE: perhaps can have some structured representation for the model.
# TODO: add the latex header. make it multi column too.
def generate_latex_table(mat, num_places, row_labels=None, column_labels=None,
        bold_type=None, fpath=None, show=True):
    assert bold_type == None or (bold_type[0] in {'smallest', 'largest'} and 
        bold_type[1] in {'in_row', 'in_col', 'all'})
    assert row_labels == None or len(row_labels) == mat.shape[0] and ( 
        column_labels == None or len(column_labels) == mat.shape[1])
    
    # effective number of latex rows and cols
    num_rows = mat.shape[0]
    num_cols = mat.shape[1]
    if row_labels != None:
        num_rows += 1
    if column_labels != None:
        num_cols += 1

    # round data
    proc_mat = np.round(mat, num_places)
    
    # determine the bolded entries:
    bold_mat = np.zeros_like(mat, dtype='bool')
    if bold_type != None:
        if bold_type[0] == 'largest':
            aux_fn = np.argmax
        else:
            aux_fn = np.argmin
        
        # determine the bolded elements; if many conflict, bold them all.
        if bold_type[1] == 'in_row':
            idxs = aux_fn(proc_mat, axis=1)
            for i in xrange(mat.shape[0]):
                mval = proc_mat[i, idxs[i]]
                bold_mat[i, :] = (proc_mat[i, :] == mval)

        elif bold_type[1] == 'in_col':
            idxs = aux_fn(proc_mat, axis=0)
            for j in xrange(mat.shape[1]):
                mval = proc_mat[idxs[j], j]
                bold_mat[:, j] = (proc_mat[:, j] == mval)
        else:
            idx = aux_fn(proc_mat)
            for j in xrange(mat.shape[1]):
                mval = proc_mat[:][idx]
                bold_mat[:, j] = (proc_mat[:, j] == mval)

    # construct the strings for the data.
    data = np.zeros_like(mat, dtype=object)
    for (i, j) in iter_product(range(mat.shape[0]), range(mat.shape[1])):
        s = "%s" % proc_mat[i, j]
        if bold_mat[i, j]:
            s = "\\textbf{%s}" % s
        data[i, j] = s
    
    header = ''
    if column_labels != None:
        header = " & ".join(column_labels) + " \\\\ \n"
        if row_labels != None:
            header = "&" + header
    
    body = [" & ".join(data_row) + " \\\\ \n" for data_row in data]
    if row_labels != None:
        body = [lab + " & " + body_row for (lab, body_row) in zip(row_labels, body)]

    table = header + "".join(body)
    if fpath != None:
        with open(fpath, 'w') as f:
            f.write(table)
    if show:
        print table


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# fig = plt.figure()


# def f(x, y):
#     return np.sin(x) + np.cos(y)

# x = np.linspace(0, 2 * np.pi, 120)
# y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# # ims is a list of lists, each row is a list of artists to draw in the
# # current frame; here we are just animating one artist, the image, in
# # each frame
# ims = []
# for i in range(60):
#     x += np.pi / 15.
#     y += np.pi / 20.
#     print f(x, y)
#     im = plt.imshow(f(x, y), animated=True)
#     ims.append([im])

# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                 repeat_delay=1000)

# # ani.save('dynamic_images.mp4')

# plt.show()

# # TODO: also have some way of adding 

# subplots with animation. that would be nice. multiple animations side by side.
# rather than classes, it may be worth


# TODO: also have a cheatsheet for matplotlib.