(Simple) Multiworker Example
===============================
This tutorial demonstrates a non-MPI way to parallelize your architecture search experiments with DeepArchitect. If you are looking to use a MPI-based approach, see this tutorial (link) instead.  

Why Multiworker  
----------------
Parallelism is an important aspect of architecture search, thus we want to demonstrate some amount of parallelism and efficient resource allocation to reduce experiment time. This example is the simplest approach to parallelism, where we sample architecture from a master node and evaluate the architectures parallel in worker node. For more complex resource allocation methods, check out the tutorials on how to use Successive Halving and Hyperband (link) with DeepArchitect. 

Introduction and Workflow 
---------------------------
In this example, we will demonstrate parallelism for architecture search with mnist (link). The search space (link) and evaluator (link) will be keras-based. In addition, we will assume our tutorials were running on Pittsburgh Supercomputing Center Bridges server and customize to the specificity of this server. The code for the example can be found here (link)
We briefly describes our main modules and their purposes as follows: 
    - Master: This module submits a job to sample the architectures, then submit jobs to evaluate the architecture, and finally retrieves all results and get the best architecture. This will be run on a master node 
    - Search Space: Define the search space of our experiment 
    - Searcher/Sampler: This module executes the job of sampling the architecture. This will be run once on a worker node. 
    - Evaluator/Worker: This module executes the evaluation of the architecture and returns the result.
The below diagram summarizes our approach: 

                Master 

Searcher/Sampler     Evaluator/Worker 

                Search Space 

A lot of communications will be done via logging functionalities (link)

Master 
-------
The master node executes 3 important steps: 
1. Sample the architectures: the master communicates with the searcher to sample architectures, via function ```create_sample```. The searcher returns the information on the architectures sampled, such as the config file path and the output folderpath. These information will be forwarded to the evaluator to evaluate the architectures. 
2. Submit the sampled architectures to workers: for each architecture, the master node forwards the information of the sampled architecture to the evaluator to evaluate the architecture, via function ```create_and_submit_job```. The evaluator returns a dictionary of results, including the validation accuracy.  
3. Wait for the workers' results and compare: the master waits for the job to complete (via function ```done```), and get the best architecture.   

```python
def main():

    num_classes = 10
    num_samples = 2 # number of architecture to sample 
    exp_name = 'mnist_multiworker'
    best_val_acc, best_architecture = 0., -1
    
    # create and submit batch job to sample architectures and get information back  
    architectures = create_sample(num_classes, num_samples, exp_name)
    
    # create and submit batch job to evaluate each architecture
    jobs = []
    for i in architectures:
        (config_fp, evaluation_fp) = architectures[i]["config_filepath"], architectures[i]["evaluation_filepath"]
        (job_id, result_fp) = create_and_submit_job(int(i), exp_name, config_fp, evaluation_fp)
        jobs.append((int(i), job_id, result_fp))

    # extract result and remove job 
    while(len(jobs) > 0) :
        for (arch_id, result_fp, job_id) in jobs: 
            if (done(job_id) and valid_filepath(result_fp)): 
                results = ut.read_jsonfile(fp) 
                val_acc = float(results['val_accuracy'])
                print("Finished evaluating architecture %d, validation accuracy is %f" % (arch_i, val_acc))
                if val_acc > best_val_acc: 
                    best_architecture, best_val_acc = arch_id, val_acc
            jobs.remove((arch_id, eval_logger, job_id))
    
    print("Best validation accuracy is %f with architecture %d" % (best_val_acc, best_architecture)) 

```

Search Space
----------------
The search space that we use is identical to the mnist tutorial (link)

Searcher 
----------
This module connects with the search space, uses random search to sample an architecture with config and features, then logs those information and communicates back with the master. Specifically, this module returns the filepath to the config files that is crucial to the formation of an architecture. It takes in several parameters from the master: 
- num_classes: Number of classes to predict (for search space)
- num_samples: Number of architecture to sample
- exp_name: Name of Experiment (to create logging folder)
- result_fp: filepath of result (contaning config file and evaluation folderpath for each architecture sampled)

We sample architectures like in DeepArchitect, via a        ```get_search_space``` function (see mnist tutorial). Then we log important information (mainly, the config file) use the logging functionalities. The config file contains the values of the Hyperparameters that define an architecture. 

The searcher writes to a json file a dictionary ```architectures```. Each key is an architecture id (a number), and the values are the architecture's config filepath and output folderpath. Each architecture gets its own output folderpath. This file is retrieved by the master node in the ```create_sample``` function. 

```python
def main(config):

    searcher = se.RandomSearcher(get_search_space(config.num_classes)) # random searcher 
    # create a logging folder to log information (config and features)
    logger = log.SearchLogger('logs', config.exp_name, resume_if_exists=True, create_parent_folders=True)
    # return values
    architectures = dict() 
    
    for i in range(int(config.num_samples)):
        print("Sampling architecture %d" % i)
        inputs, outputs, hs, h_value_hist, searcher_eval_token = searcher.sample()
        eval_logger = logger.get_current_evaluation_logger()
        eval_logger.log_config(h_value_hist, searcher_eval_token)
        eval_logger.log_features(inputs, outputs, hs) 
        architectures[i] = {'config_filepath': eval_logger.config_filepath, 
                            'evaluation_filepath': eval_logger.get_evaluation_folderpath()}
    
    # write to a json file to communicate with master
    ut.write_jsonfile(architectures, config.result_fp)
```

Evaluator 
----------
The evaluator is essentially the "worker" node of our workflow. Here we use the familiar SimpleClassifierEvaluator in the mnist tutorial.

The evaluator takes in 2 parameters: the config file that contains the hyperparameters values for an architecture, and the output folderpath to log the results. 

The key thing is retrieving the architecture. Besides the hyperparameter information from the config file, we need the inputs and outputs object. Thus, similar to the searcher, we also have the identical ```get_search_space``` function to get the inputs and outputs from the search space. Together with the config file, we can specify our architecture using the function 

```python
specify(outputs.values(), hs, h_values["hyperp_value_lst"])
```

where outputs is the outputs obtained from get_search_space functions, hs is extra hyperparameter values (learning rate, optimizer, etc), and h_values is retrieved from the config file. Below is the key excerpt from the evaluator 

```python
def main(args): 

    num_classes = 10 

    # load and normalize data 
    (x_train, y_train),(x_test, y_test) = load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # defining evaluator
    evaluator = SimpleClassifierEvaluator((x_train, y_train),                              num_classes, max_num_training_epochs=5)
    inputs, outputs, hs = get_search_space(num_classes)() 
    h_values = ut.read_jsonfile(args.config) 
    specify(outputs.values(), hs, h_values["hyperp_value_lst"]) # hs is "extra" hyperparameters
    results = evaluator.evaluate(inputs, outputs, hs) 
    ut.write_jsonfile(results, args.result_fp)
```

Running on Pittsburgh Supercomputing Center (Bridges) 
-----------------------------------------------------
To run the above example on the bridges server: 
- Create a folder darch inside your Bridges working directory ($SCRATCH). Have at least deep_architect and the examples folder inside the darch directory. 
- Download mnist file 'mnist.npz' from https://s3.amazonaws.com/img-datasets/mnist.npz (or any other link with mnist.npz). Create folder named datasets inside darch and place 'mnist.npz' in there. 
- Start at the root directory, type in 'export PYTHONPATH=${PYTHONPATH}:/path/to/darch/'
- Run python examples/tutorials/multiworker/master.py 

In addition, you can inspect the log in 'logs/mnist_multiworker'

What's next 
-------
Below are some suggested improvements can be made to make your experiments more efficient/productive: 
    - We can use a config file for all the server information (partition, number of nodes, max walltiime, etc.)
    - More? 

Check out MPI to be more robust 
Check out Hyperband/SuccessiveHalving for resource allocation policies examples. 






