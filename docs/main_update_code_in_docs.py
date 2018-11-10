import deep_architect.utils as ut


def read_code_blocks(filepath):
    code_marker = '### (((CODE-BELOW))) ###'

    lines = ut.read_textfile(filepath, strip=False)
    block_lst = []
    block_lines = None
    for line in lines:
        s = line.rstrip()
        if s == code_marker:
            if block_lines is not None:
                block_lst.append(block_lines)
            block_lines = []
        else:
            block_lines.append(s)
    if block_lines is not None:
        block_lst.append(block_lines)
    return block_lst


if __name__ == "__main__":
    code_marker = '### (((CODE-BELOW))) ###'

    cmd = ut.CommandLineArgs()
    cmd.add('in_doc_filepath', 'str')
    cmd.add('in_code_filepath', 'str')
    cmd.add('out_filepath', 'str')
    d = cmd.parse()

    # wrong. this is going to be interesting. I think that reading the blocks
    # can be done easily.
    block_lst = read_code_blocks(d["in_code_filepath"])
    lines = ut.read_textfile(d["in_doc_filepath"], strip=False)
    inside_block = False
    block_idx = 0
    out_lines = []
    indent = 4
    if d["in_doc_filepath"].endswith('.rst'):
        for line in lines:
            s = line.rstrip()
            if inside_block:
                if not (s == '' or s == ' '):
                    inside_block = False
                    for b_line in block_lst[block_idx]:
                        out_lines.append((' ' * indent) + s)
                    block_idx += 1
            else:
                out_lines.append(s)

            if s == ".. code:: python":
                inside_block = True
                out_lines.append(s)

        # if it terminates with a block. likely unusual.
        if inside_block:
            for b_line in block_lst[block_idx]:
                out_lines.append((' ' * indent) + s)

    elif d["in_doc_filepath"].endswith('.md'):
        for line in lines:
            s = line.rstrip()
            if s == '```python':
                inside_block = True
                out_lines.append(s)
            elif s == '```':
                if inside_block:
                    out_lines.extend(block_lst[block_idx])
                    out_lines.append(s)
                    block_idx += 1
                    inside_block = False
            else:
                out_lines.append(s)
    else:
        raise ValueError(
            "File %s is not supported. Supported formats .md and .rst" %
            d["in_doc_filepath"])

    ut.write_textfile(d["out_filepath"], out_lines)