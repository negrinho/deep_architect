"""
Script that helps converting tutorials in .py format to md, rst, ipynb format
"""
import os 
import sys 
import string 

input_file = sys.argv[1]
output_file = sys.argv[2]
target_format = sys.argv[3]
supported_format = ["markdown", "rst", "ipynb"]

md_header = "### ${MARKDOWN}" 
rst_header = "### ${RST}" 
text_headers = {md_header, rst_header} # can add more header in the future 
code_header = "### ${CODE}" 

fin = open(input_file, 'r')

if (target_format == "ipynb"):
    raise NotImplementedError()
else: 
    assert(target_format in supported_format) 
    content = ""
    code_block = ""
    mode = None # ["text" or "code"]
    for line in fin: 
        if (line.strip() in text_headers):
            header = line.strip()[6:-1].lower()
            assert(target_format == header) 
            mode = "text"
            if code_block != "": 
                content += code_block + "```\n\n" if target_format == "markdown" else code_block + "\n\n"
            continue 
        if (line.strip() == code_header): 
            mode = "code" 
            code_block = "\n```python\n" if target_format == "markdown" else "\n.. code-block:: python\n"
            continue 
        if (mode == "text"): 
            line = line[2:] # first two characters should be "# " so delete them 
            content += line
        elif (mode == "code"): 
            code_block += line if target_format == "markdown" else "\t" + line 

    # add last code block 
    if (mode == "code"): content += code_block + "```\n\n" if target_format == "markdown" else code_block + "\n\n"

fin.close()

with open(output_file, 'w') as fout: 
    fout.write(content)
    



