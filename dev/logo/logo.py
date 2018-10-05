from math import cos, sin, pi
import deep_architect.utils as ut
import numpy as np

tikz_lines = []

def coords(center, radius, angle_in_degrees):
    x = center[0] + radius * cos(angle_in_degrees / 180.0 * pi)
    y = center[1] + radius * sin(angle_in_degrees / 180.0 * pi)
    return (x, y)

def define_color(color_name, rgb):
    s = '\\definecolor{%s}{RGB}{%d,%d,%d}' % (color_name, rgb[0], rgb[1], rgb[2])
    tikz_lines.append(s)

def draw_bounding_box(bottom_left, top_right, line_width, line_color, fill_color, round_corners):
    opts = []
    if round_corners:
        opts.append("rounded corners")
    opts.extend([
        "color=%s" % line_color,
        "fill=%s" % fill_color,
        "line width=%fcm" % line_width
    ])
    opts_s = ", ".join(opts)
    s = '\\draw [%s](%fcm, %fcm) rectangle (%fcm, %fcm);' % (
        opts_s, bottom_left[0], bottom_left[1], top_right[0], top_right[1])
    tikz_lines.append(s)

# these functions generate the necessary TiKZ line.
def draw_node(center, diameter, line_width, line_color, fill_color):
    s = "\\draw [line width=%fcm, color=%s, fill=%s] (%fcm, %fcm) circle (%fcm);" % (
        line_width, line_color, fill_color, center[0], center[1], diameter / 2.0)
    tikz_lines.append(s)

def draw_arrow(start_center, end_center, line_width, line_color, node_diameter, margin,
        head_length, head_width):
    start_center = np.array(start_center, dtype='float64')
    end_center = np.array(end_center, dtype='float64')
    v = end_center - start_center
    v = v / np.sqrt(np.sum(v * v))
    v = (margin + node_diameter / 2.0) * v
    start_center = start_center + v
    end_center = end_center - v

    s = '\\draw [-{Latex[length=%fcm, width=%fcm]}][line width=%fcm, color=%s](%fcm, %fcm) -- (%fcm, %fcm);' % (
        head_length, head_width, line_width, line_color,
        start_center[0], start_center[1], end_center[0], end_center[1])
    tikz_lines.append(s)

def draw_logo(cfg):
    node_fn = lambda center: draw_node(center,
            cfg["node_diameter"], cfg["node_line_width"],
            "node_line_color", "node_fill_color")

    arrow_fn = lambda start_center, end_center: draw_arrow(
        start_center, end_center, cfg["arrow_line_width"], "arrow_line_color",
        cfg["node_diameter"], cfg["arrow_margin"],
        cfg["arrow_head_length"], cfg["arrow_head_width"])

    d_top = (0, cfg["D_height"])
    d_bottom = (0, 0)
    mid = (cfg["D_width"], cfg["center_node_height"])
    a_top = (cfg["D_width"] + 0.5 * cfg["A_width"], cfg["A_height"])
    a_mid_right = (cfg["D_width"] + cfg["A_width"], cfg["center_node_height"])
    a_left_bottom = (cfg["D_width"], 0)
    a_right_bottom = (cfg["D_width"] + cfg["A_width"], 0)

    # draw bounding box
    u = cfg["bbox_margin"]
    w = cfg["D_width"] + cfg["A_width"]
    h = cfg["A_height"]
    draw_bounding_box((-u, -u), (w + u, h + u),
        cfg["bbox_line_width"], "bbox_line_color", "bbox_fill_color",
        cfg["bbox_round_corners"])

    # draw all the nodes.
    node_fn(d_top)
    node_fn(d_bottom)
    node_fn(mid)
    node_fn(a_top)
    node_fn(a_mid_right)
    node_fn(a_left_bottom)
    node_fn(a_right_bottom)

    # # draw all arrows.
    arrow_fn(d_top, d_bottom)
    arrow_fn(d_top, mid)
    arrow_fn(d_bottom, mid)
    arrow_fn(a_top, mid)
    arrow_fn(a_top, a_mid_right)
    arrow_fn(a_mid_right, mid)
    arrow_fn(mid, a_left_bottom)
    arrow_fn(a_mid_right, a_right_bottom)

# angles in degrees, lengths in cm.
if __name__ == "__main__":
    cfg = ut.get_config()
    tikz_lines.extend([
        '\\documentclass{standalone}',
        '\\usepackage{tikz}',
        '\\usetikzlibrary{arrows.meta, positioning}',
        '\\begin{document}',
        '\\begin{tikzpicture}',
    ])
    for name in [
        "node_line_color", "node_fill_color", "arrow_line_color",
        "bbox_line_color", "bbox_fill_color"]:
        define_color(name, cfg[name])

    draw_logo(cfg)
    tikz_lines.extend([
        '\\end{tikzpicture}',
        '\\end{document}',
    ])
    ut.write_textfile('logo.tex', tikz_lines)

# https://tex.stackexchange.com/questions/5461/is-it-possible-to-change-the-size-of-an-arrowhead-in-tikz-pgf
# https://tex.stackexchange.com/questions/228724/how-do-i-make-tikz-make-a-curved-arrow-from-one-node-to-another-when-my-nodes-ar
# https://tex.stackexchange.com/questions/228724/how-do-i-make-tikz-make-a-curved-arrow-from-one-node-to-another-when-my-nodes-ar
# ipython dev/negrinho/logo.py -- --config_filepath dev/negrinho/logo.json && pdflatex logo.tex