import deep_architect.utils as ut


def read_code_blocks(filepath):
    lines = ut.read_textfile(filepath, strip=False)
    block_lines = []
    inside_block = False
    block_lst = []
    if filepath.endswith('.rst'):
        indent = None
        for line in lines:
            s = line.rstrip()
            if inside_block:
                if s == '':
                    block_lines.append(s)
                elif s[0] == ' ':
                    if indent == None:
                        indent = 0
                        for ch in s:
                            if ch == " ":
                                indent += 1
                            else:
                                break
                    block_lines.append(s[indent:])
                else:
                    block_lst.append(block_lines)
                    block_lines = []
                    inside_block = False
                    indent = None

            if s == ".. code:: python":
                inside_block = True

    elif filepath.endswith('.md'):
        for line in lines:
            s = line.rstrip()
            if s == '```python':
                inside_block = True
            elif s == '```':
                if inside_block:
                    block_lst.append(block_lines)
                    block_lines = []
                    inside_block = False
            else:
                if inside_block:
                    block_lines.append(s)

    else:
        raise ValueError(
            "File %s is not supported. Supported formats .md and .rst" %
            filepath)
    return block_lst


if __name__ == "__main__":
    code_marker = '### (((CODE-BELOW))) ###'

    cmd = ut.CommandLineArgs()
    cmd.add('in_filepath', 'str')
    cmd.add('out_filepath', 'str')
    d = cmd.parse()

    block_lst = read_code_blocks(d["in_filepath"])
    out_lines = []
    for block_lines in block_lst:
        out_lines.append(code_marker)
        out_lines.extend(block_lines)
    ut.write_textfile(d["out_filepath"], out_lines)
