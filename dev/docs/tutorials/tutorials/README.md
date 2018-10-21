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