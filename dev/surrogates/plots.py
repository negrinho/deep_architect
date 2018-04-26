import darch.search_logging as sl
import darch.visualization as vi
import numpy as np
import seaborn as sns; sns.set()

# checking these across time.
log_lst = sl.read_search_folder('./logs/cifar10_medium/run-0')
xkey = 'epoch_number'
ykey = 'validation_accuracy'
num_lines = 8
time_plotter = vi.LinePlot(xlabel='time_in_minutes', ylabel=ykey)
epoch_plotter = vi.LinePlot(xlabel=xkey, ylabel=ykey)
for lg in sorted(log_lst, key=lambda x: x['results']['sequences'][ykey][-1], reverse=True)[:num_lines]:
    r = lg['results']['sequences']
    time_plotter.add_line(np.linspace(0.0, 120.0, len(r[xkey]) + 1)[1:], r[ykey])
    epoch_plotter.add_line(r[xkey], r[ykey])
time_plotter.plot()
epoch_plotter.plot()
