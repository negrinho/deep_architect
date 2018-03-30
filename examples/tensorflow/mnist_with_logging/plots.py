
import darch.search_logging as sl
import darch.visualization as vi
import numpy as np
import random

if __name__ == '__main__':
    log_lst = sl.read_search_folder('logs/test')
    random.shuffle(log_lst)
    plotter = vi.LinePlot()
    eval_ids = xrange(1, len(log_lst) + 1)
    val_accs = vi.running_max([name_to_log['results']['validation_accuracy'] for name_to_log in log_lst])
    plotter.add_line(eval_ids, val_accs)
    plotter.plot()