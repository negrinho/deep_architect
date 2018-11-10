import deep_architect.utils as ut

if __name__ == "__main__":
    cmd = ut.CommandLineArgs()
    cmd.add('filepath', 'str')
    d = cmd.parse()

    lines = ut.read_textfile(d["filepath"], strip=False)
    out_lines = []
    inside_block = False
    if d["filepath"].endswith('.rst'):
        indent = None
        for line in lines:
            s = line.rstrip()
            if inside_block:
                if s == '':
                    out_lines.append(s)
                elif s[0] == ' ':
                    if indent == None:
                        indent = 0
                        for ch in s:
                            if ch == " ":
                                indent += 1
                            else:
                                break
                    out_lines.append(s[indent:])
                else:
                    out_lines.append('#')
                    inside_block = False
                    indent = None

            if s == ".. code:: python":
                inside_block = True

    elif d["filepath"].endswith('.md'):
        for line in lines:
            s = line.rstrip()
            if s[:3] == '```':
                inside_block = not inside_block
            else:
                if inside_block:
                    out_lines.append(s)

    else:
        raise ValueError(
            "File %s is not supported. Supported formats .md and .rst" %
            d["filepath"])

    ut.write_textfile('extract_test.py', out_lines)