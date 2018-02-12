def get_eval_fn(start_fn, train_fn, is_over_fn, end_fn):
    def eval_fn(d):
        start_fn(d)
        while not is_over_fn(d):
            train_fn(d)
        end_fn(d)
    return eval_fn