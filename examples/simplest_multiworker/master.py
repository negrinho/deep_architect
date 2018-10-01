import deep_architect.search_logging as sl
import deep_architect.visualization as vi
import deep_architect.utils as ut
from deep_architect.searchers.random import RandomSearcher
# TODO: change this relative import.
import search_space as ss


def main():
    # Loading the config file.
    cfg = ut.get_config()
    # Creating the searcher.
    searcher = RandomSearcher(ss.search_space_fn)
    # Creating the search folder for logging information.
    sl.create_search_folderpath(
        cfg['folderpath'],
        cfg['search_name'],
        abort_if_exists=cfg["abort_if_exists"],
        delete_if_exists=cfg['delete_if_exists'],
        create_parent_folders=True)
    # Search loop.
    for evaluation_id in xrange(cfg['num_samples']):
        logger = sl.EvaluationLogger(cfg["folderpath"], cfg["search_name"],
                                     evaluation_id)
        if not logger.config_exists():
            inputs, outputs, hyperp_value_lst, searcher_eval_token = searcher.sample(
            )
            # Logging results (including graph).
            logger.log_config(hyperp_value_lst, searcher_eval_token)
            logger.log_features(inputs, outputs)
            vi.draw_graph(
                outputs.values(),
                True,
                True,
                print_to_screen=False,
                out_folderpath=logger.get_evaluation_data_folderpath())


if __name__ == '__main__':
    main()