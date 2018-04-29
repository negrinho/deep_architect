import darch.visualization as vi
import darch.search_logging as sl
import numpy as np

def map_dict(d, fn):
    return {k : fn(k, v) for (k, v) in d.iteritems()}

def zip_toggle(xs):
    """[[x1, ...], [x2, ...], [x3, ...]] --> [(x1, x2, .., xn) ...];
        [(x1, x2, .., xn) ...] --> [[x1, ...], [x2, ...], [x3, ...]]"""
    assert isinstance(xs, list)
    return zip(*xs)

def argsort(xs, fns, increasing=True):
    """The functions in fns are used to compute a key which are then used to
    construct a tuple which is then used to sort. The earlier keys are more
    important than the later ones.
    """
    def key_fn(x):
        return tuple([f(x) for f in fns])

    idxs, _ = zip_toggle(
        sorted(enumerate(xs),
            key=lambda x: key_fn(x[1]),
            reverse=not increasing))
    return idxs

def sort(xs, fns, increasing=True):
    idxs = argsort(xs, fns, increasing)
    return apply_permutation(xs, idxs)

def apply_permutation(xs, idxs):
    assert len(set(idxs).intersection(range(len(xs)))) == len(xs)
    return [xs[i] for i in idxs]

def keep_top_k(ds, fn, k, maximizing=True):
    return sort(ds, [fn], increasing=not maximizing)[:k]

def generate_indices(num_items, max_num_indices=32, increase_factor=2, multiplicative_increases=True):
    assert not (multiplicative_increases and increase_factor == 1)
    idxs = []
    v = 0
    for i in xrange(max_num_indices):
        idxs.append(v)
        if multiplicative_increases:
            v = (v + 1) * increase_factor - 1
        else:
            v += increase_factor
        if v >= num_items:
            break
    return idxs

def sort_sequences(sequence_lst_lst, value_lst, maximizing):
    """Sorts a list of lists of sequences according to value_lst.

    All lists of sequences in the list of sequences must have the same length.
    """
    assert all(len(seq) == len(value_lst) for seq in sequence_lst_lst)
    idxs = argsort(value_lst, [lambda x: x], increasing=not maximizing)
    sorted_sequence_lst = [apply_permutation(seq, idxs) for seq in sequence_lst_lst]
    return sorted_sequence_lst

def callibration_plot(time_sequence_lst, value_sequence_lst, maximizing=True,
        max_num_plots=8, increase_factor=2, multiplicative_increases=True,
        time_axis_label=None, value_axis_label=None, show=True, plot_filepath=None):
    """Useful functionality to determine how much computation is reasoanble to
    expend to do a good job identifying a good model.
    """
    assert len(time_sequence_lst) == len(value_sequence_lst)

    if maximizing:
        value_lst = [np.max(seq) for seq in value_sequence_lst]
    else:
        value_lst = [np.min(seq) for seq in value_sequence_lst]

    (sorted_time_sequence_lst, sorted_value_sequence_lst) = sort_sequences(
        [time_sequence_lst, value_sequence_lst], value_lst, maximizing)

    num_sequences = len(time_sequence_lst)
    indices = generate_indices(num_sequences, max_num_plots,
        increase_factor, multiplicative_increases)

    plotter = vi.LinePlot(title='Total number of sequences: %d' % num_sequences,
        xlabel=time_axis_label, ylabel=value_axis_label)
    for idx in indices:
        plotter.add_line(sorted_time_sequence_lst[idx],
            sorted_value_sequence_lst[idx], str(idx))
    plotter.plot(show=show, fpath=plot_filepath)

def get_value_at_time(query_time, time_sequence, value_sequence, maximizing=True):
    """Gets the value at a specific time step.

    It is assume that `time_sequence` is ordered in increasing order of time.
    """
    assert len(time_sequence) == len(value_sequence)
    if query_time < time_sequence[0]:
        v = - np.inf if maximizing else np.inf
    elif query_time == time_sequence[0]:
        v = value_sequence[0]
    elif query_time >= time_sequence[-1]:
        v = value_sequence[-1]
    else:
        left_idx = 0
        right_idx = len(value_sequence) - 1
        while left_idx <= right_idx:
            idx = (left_idx + right_idx) / 2
            t = time_sequence[idx]
            t_next = time_sequence[idx + 1]
            if t <= query_time and t_next > query_time:
                v = value_sequence[idx]
                break
            else:
                if t < query_time:
                    left_idx = idx + 1
                elif t > query_time:
                    right_idx = idx - 1
                else:
                    # should never happen.
                    assert False
    return v

def callibration_table(time_sequence_lst, value_sequence_lst, maximizing=True,
        max_num_ranks=8, rank_increase_factor=2, rank_multiplicative_increases=True,
        start_time=1e-3, num_time_instants=16, time_increase_factor=2, time_multiplicative_increases=True,
        time_label=None, value_label=None, show=True, table_filepath=None):
    """Gets the rank of different models at different time steps.

    The true rank of a model is determined by the best performance that it
    achieves in the sequence. A good amount of computation is suffient to
    order models close to their true ordering. Entries in the table are
    true rank of the model.
    This functionality is useful to inform what amount of computation is
    sufficient to perform architecture search on this problem.
    """
    assert len(time_sequence_lst) == len(value_sequence_lst)

    if maximizing:
        value_lst = [np.max(seq) for seq in value_sequence_lst]
    else:
        value_lst = [np.min(seq) for seq in value_sequence_lst]

    (sorted_time_sequence_lst, sorted_value_sequence_lst) = sort_sequences(
        [time_sequence_lst, value_sequence_lst], value_lst, maximizing)

    num_sequences = len(time_sequence_lst)
    indices = generate_indices(num_sequences, max_num_ranks,
        rank_increase_factor, rank_multiplicative_increases)

    if time_multiplicative_increases:
        time_instants = [start_time * (time_increase_factor ** i)
            for i in xrange(num_time_instants)]
    else:
        time_instants = [start_time + i * time_increase_factor
            for i in xrange(num_time_instants)]

    rows = []
    for t in time_instants:
        values_at_t = []
        for i in xrange(num_sequences):
            v = get_value_at_time(t,
                    sorted_time_sequence_lst[i], sorted_value_sequence_lst[i],
                    maximizing=maximizing)
            values_at_t.append(v)

        ranks = argsort(values_at_t, [lambda x: x], increasing=not maximizing)
        row = [ranks[idx] for idx in indices]
        rows.append(row)

    # constructing the table.
    table_lines = ["Total number of sequences: %d" % num_sequences]
    if time_label is not None or value_label is not None:
        lst = []
        if time_label is not None:
            lst.append('time=%s' % time_label)
        if value_label is not None:
            lst.append('value=%s' % value_label)
        line = ", ".join(lst)
        table_lines.append(line)

    line = " ".join(['time / rank\t'] + ["%5d" % idx for idx in indices])
    table_lines.append(line)
    for i, t in enumerate(time_instants):
        line = " ".join(["%2.2e\t" % t] + ["%5d" % rank for rank in rows[i]])
        table_lines.append(line)

    if table_filepath is not None:
        sl.write_textfile(table_filepath, table_lines)
    if show:
        print "\n".join(table_lines)
