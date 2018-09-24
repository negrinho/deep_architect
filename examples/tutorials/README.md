<<<<<<< HEAD
To build a tutorial, first build in a python file following the convention and 
example in mnist_tutorial.py as follows: 

```
### ${MARKDOWN/RST/IPYNB}

# markdown/restructuredText text 
# 
# another markdown/restructuredText text 

### ${CODE} 

def dummy(): pass 

### ${MARKDOWN/RST/IPYNB}

# yet another markdown/restructuredText/ text 
```
You can write your tutorial in both markdown and rst format.

Then use the script convert.py to convert to either Markdown, rst or Ipython Notebook
Usage: python convert.py input_file.py output_file format 
where format = {markdown, rst, ipynb}

We then can also use pandoc to help converting from Markdown to RST https://pandoc.org/getting-started.html

If you want to convert md to rst, use pandoc directly 
pandoc -o text.rst text.md

Alternatively, you can also write your tutorial in markdown or rst format 


NOTE: 
Mnist tutorial main is tf.keras, with link to github code for all other frameworks (should be heavily commented for the difference)

In the future, develop to have code tabs
=======

# Getting started



In these tutorials, we cover several use-cases for using DeepArchitect in full
capacity.
These range from tutorials focusing on the user experience with the framework,
e.g.,
how to write a search space in your favorite deep learning;
how to code a simple architecture search experiment using a single GPU in DeepArchitect;
how to code a multi-worker architecture search experiment;
how to generate logging information and visualize it;
to tutorials focusing on extending the capabilities of DeepArchitect, e.g.,
what are the basic constructs that make modular architecture search possible;
how are different constructs implement


# Why use DeepArchitect

Recently, there was a stream of research in architecture search.
Unfortunately, much of this research builds their own systems from scratch,
severely reducing the possibility of building on them.
In DeepArchitect, we designed a careful API to build modular architecture search systems.
Our framework allows the researcher to identify a specific aspect of architecture
search that the researcher wants to explore and focus on it while reusing
available components from other models.
This simultaneously improves reusability and reproducibility of the researcher's
work, making it readily available to the community and allowing fairer and
easier comparisons to previous work and future work.

- Well designed APIs:
When designing DeepArchitect, we paid special attention to the design of the
different APIs to make sure that the programmer would express concepts at the
right level of abstraction. This means that compared

- Only wrapper code:
- Modular:
- Easy to define search spaces:


<!-- The tutorial series is meant to get the user up to speed with using the
framework and understanding the underlying concepts.

README can say what concepts are used in each of the different models.
I think that is important.

Single GPU.

Talk about multiple workers later in the codebase. what can I do here.

 -->

TODO: write some abbreviations that are used extensively in the codebase.
Maintaining consistency on some of these things, like they are used
very consistently.


TODOs:
* talk about the executable counterpart.
* tell about what are easy to undestand and what is required
>>>>>>> 946a8485641f72703732332d3063afb44ab90548
