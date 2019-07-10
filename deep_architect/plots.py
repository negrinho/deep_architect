import deep_architect.extract as ex
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def generate_histograms(accs):
    # ptl = ex.get_logs()
    # rand_test_accuracies = [log['test_accuracy'] for log in ptl['framework_random']]
    # evol_test_accuracies = [log['test_accuracy'] for log in ptl ['framework_evolution']]

    hist_range = (min([min(accs[acc]) for acc in accs]),
                    max([max(accs[acc]) for acc in accs]))

    # hist_range = min(min(rand_test_accuracies), min(evol_test_accuracies)), \
    #     max(max(rand_test_accuracies), max(evol_test_accuracies))
    for acc in accs:
        sns.distplot(accs[acc], bins=100, kde=False, hist_kws={'range':hist_range}, axlabel='Test Accuracy on CIFAR-10', label=acc)
    # sns.distplot(evol_test_accuracies, bins=100, kde=False, hist_kws={'range':hist_range}, axlabel='Test Accuracy on Fashion MNIST', label='Evolution')
    plt.legend(loc=0)
    plt.ylabel('Number of Models')
    plt.show()

def generate_line_plot(accs, smoothing=.9):
    # ptl = ex.get_logs()
    # rand_test_accuracies = [log['test_accuracy'] for log in ptl['framework_random']]
    # evol_test_accuracies = [log['test_accuracy'] for log in ptl['framework_evolution']]
    # sliding_rand_accs = [np.mean(rand_test_accuracies[i:i+window]) for i in range(len(rand_test_accuracies) - window)]
    # sliding_evol_accs = [np.mean(evol_test_accuracies[i:i+window]) for i in range(len(evol_test_accuracies) - window)]
    for acc in accs:
        smoothed_acc = smooth(accs[acc], smoothing)
        sns.lineplot(x=range(len(smoothed_acc)), y=smoothed_acc, legend='full', label=acc)

    # sliding_rand_accs = smooth(rand_test_accuracies, smoothing)
    # sliding_evol_accs = smooth(evol_test_accuracies, smoothing)
    # sns.lineplot(x=range(len(sliding_rand_accs)), y=sliding_rand_accs, legend='full', label='Random')
    # sns.lineplot(x=range(len(sliding_evol_accs)), y=sliding_evol_accs, legend='full', label='Evolution')
    plt.legend(loc=0)
    plt.xlabel('Number of Models Sampled')
    plt.ylabel('Smoothed Test Accuracy')
    plt.show()

def generate_plots():
    ptl = ex.get_logs()
    accs = {k: [log['test_accuracy'] for log in ptl[k][:256]] for k in ptl}
    running_max = {k: [max(accs[k][:i+1]) for i in range(len(accs[k]))] for k in accs}
    generate_line_plot(accs)
    generate_line_plot(running_max, smoothing=0)
    generate_histograms(accs)

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed