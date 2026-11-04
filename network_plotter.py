import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Connection weights are drawn on a diverging scale: negative weights blue,
# zero purple, positive weights red. The scale is always relative to the
# strongest weight in the network being drawn, so the colour tells you where
# a weight sits "in this network", not any particular number.
#
# The ramp changes hue but barely changes lightness, which is deliberate: a
# weight near zero stays clearly visible instead of fading into the page, and
# no value turns muddy the way a through-black ramp does. The cost is that
# colour alone cannot tell you HOW big a weight is -- so line thickness
# carries that instead (see weight_linewidth). Sign from hue, size from width.
WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    'weight', ['#1040FF', '#C45BFF', '#F02000'])


def weight_vmax(*weight_sets):
    """Largest absolute weight across everything passed in.

    Pass every weight in the network, so that one colour scale covers the
    whole diagram:  vmax = netplot.weight_vmax(w_in_hid, w_hid_out)
    """
    values = np.concatenate([np.atleast_1d(np.asarray(w, dtype=float)).ravel()
                             for w in weight_sets])
    return max(float(np.abs(values).max()), 1e-9)


def weight_color(weight, vmax, gamma=0.55, floor=0.30):
    """Colour for a single weight on the blue-purple-red scale.

    The mapping is deliberately NOT linear. On a linear ramp a weight of +0.1
    and one of -0.1 both land almost exactly on the centre colour, so you
    cannot see their sign -- and in a trained network most weights are small.
    Any non-zero weight is therefore pushed at least `floor` of the way toward
    its end of the scale, and the rest of the distance is compressed by
    `gamma`, which spreads small weights out and squeezes large ones together.

    Sign is always legible; true magnitude is carried by line thickness (see
    weight_linewidth), which stays strictly linear. Pass gamma=1.0, floor=0.0
    for the plain linear mapping.
    """
    strength = min(1.0, abs(weight) / vmax) if vmax else 0.0
    if strength == 0:
        position = 0.5                      # exactly zero sits at the centre
    else:
        boosted = floor + (1.0 - floor) * strength ** gamma
        position = 0.5 + 0.5 * (1 if weight > 0 else -1) * boosted
    return WEIGHT_CMAP(position)


def weight_linewidth(weight, vmax, base=2.0):
    """How thick to draw a connection carrying this weight.

    Hue tells you the sign of a weight; thickness tells you its size. A weight
    at the top of the scale is drawn about six times thicker than one near
    zero, so strong connections stand out even in black and white.
    """
    strength = min(1.0, abs(weight) / vmax) if vmax else 0.0
    # The floor matters: below about 0.8pt a line is thinner than a pixel, so
    # anti-aliasing blends it with the white page and it renders as a washed
    # out grey no matter what colour we asked for. Keep the thinnest line wide
    # enough to show its actual colour.
    return base * (0.6 + 1.1 * strength)


def add_weight_colorbar(ax, vmax, label='Connection weight'):
    """Small colour key for a diagram drawn with vmax."""
    import matplotlib.cm as cm
    mappable = cm.ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax),
                                 cmap=WEIGHT_CMAP)
    bar = ax.figure.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label(label)
    return bar

def calculate_node_positions(columns, plot_width, fig_width, fig_height):
    positions = {}
    num_columns = len(columns)

    # Identify the maximum number of nodes in any column
    max_nodes_in_column = max(len(column) for column in columns)
    
    # Calculate top and bottom node positions based on the column with the most nodes
    top_node_pos = fig_height - (fig_height / (max_nodes_in_column + 2))
    bottom_node_pos = fig_height / (max_nodes_in_column + 2)

    column_spacing = plot_width / (num_columns - 1)
    start_x = (fig_width - plot_width) / 2
    
    #print(f'######## top_node_pos {top_node_pos}, bottom_node_pos {bottom_node_pos}')

    for i, column in enumerate(columns):
        num_nodes = len(column)
        x = start_x + i * column_spacing

        node_rank = list(range(num_nodes))
        #node_rank.reverse()

        if num_nodes > 1:
            # Evenly space other nodes between top and bottom
            vertical_spacing = (top_node_pos - bottom_node_pos) / (num_nodes - 1)
            for j, node in enumerate(column):
                y = top_node_pos - node_rank[j] * vertical_spacing
                positions[node] = (x, y)
                #print(f'Col {i} node {j} y = {y}')
        elif num_nodes == 1:
            # Center the single node vertically
            y = (top_node_pos + bottom_node_pos) / 2
            positions[column[0]] = (x, y)

    return positions

def draw_neuron_with_label(ax, pos, label, column_index, num_columns, show_label=True, plot_width=1, node_radius=0.05, font_size=1.5,
                          subtext="", subtext_color="red", nodetext="", nodetext_color="red"):
    font_size=font_size * plot_width
    #print(f'Neuron draw font size is {font_size}')
    node_radius = node_radius * plot_width
    # Drawing a larger neuron with label
    circle = patches.Circle(pos, radius=node_radius, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    if show_label:
        # Adjust label position and increase font size
        label_offset = node_radius + 0.05
        if column_index == 0:  # First column, label to the left
            text_pos = (pos[0] - label_offset, pos[1])
            horizontal_alignment = 'right'
        elif column_index == num_columns - 1:  # Last column, label to the right
            text_pos = (pos[0] + label_offset, pos[1])
            horizontal_alignment = 'left'
        else:  # Middle columns, label above
            text_pos = (pos[0], pos[1] + label_offset)
            horizontal_alignment = 'center'
        ax.text(text_pos[0], text_pos[1], label, horizontalalignment=horizontal_alignment, fontsize=font_size)
    # Print subtext if it is not empty
    if subtext:
        # Adjust subtext position to be directly below the node
        subtext_pos = (pos[0], pos[1] - node_radius - (label_offset / 2))  # Adjust the Y offset as needed
        ax.text(subtext_pos[0], subtext_pos[1], subtext, horizontalalignment='center', fontsize=font_size*.8, color=subtext_color)
    # Print nodetext if it is not empty
    if nodetext:
        # Adjust nodetext position to be centered inside the node
        nodetext_pos = (pos[0], pos[1] * .99)  # Adjust the Y offset as needed
        ax.text(nodetext_pos[0], nodetext_pos[1], nodetext, horizontalalignment='center', fontsize=font_size*.8, color=nodetext_color)

def draw_connection_with_weight(ax, node_positions, start_node, end_node, weight, show_weight=False, bow=False, line_width=2, 
                                arrow_size=6, plot_width=1, font_size=1.0, node_radius=0.05, edge_color='Default', bow_offset=0.07,
                                weight_fmt='+.4f', vmax=None, scale_width=True,
                                label_pos=None, label_gap=1.0):
    bow_offset = plot_width * bow_offset
    node_radius=node_radius*plot_width
    font_size = font_size * plot_width
    #print(f'-------> Connection draw font size is {font_size}')
    # Drawing a thicker directed arrow with larger arrowhead
    start = node_positions[start_node]
    end = node_positions[end_node]
    
    # Calculate direction vector
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = np.sqrt(dx**2 + dy**2)
    dir_x, dir_y = dx / dist, dy / dist

    # Adjust start and end points to connect at the edge of the nodes
    start_adj = (start[0] + node_radius * dir_x, start[1] + node_radius * dir_y) 
    end_adj = (end[0] - node_radius * dir_x, end[1] - node_radius * dir_y) 

    # Calculate angle for text rotation
    angle = np.degrees(np.arctan2(dy, dx))
    if start[0] > end[0]:  # Connection goes from right to left
        angle += 180

    # With a weight scale in play, thickness carries the magnitude so that
    # the hue is free to carry only the sign. Pass scale_width=False to keep
    # every connection the same thickness.
    if vmax is not None and scale_width:
        line_width = weight_linewidth(weight, vmax, base=line_width)

    if edge_color == 'Default':
        if vmax is not None:
            # Colour by the weight itself: blue negative, purple zero, red
            # positive, scaled against the strongest weight in the network.
            edge_color = weight_color(weight, vmax)
        elif start[0] > end[0]:
            edge_color = 'red'       # feedback connection, no weight scale given
        else:
            edge_color = 'black'

    text_color = edge_color

    # Where to put the weight label. Two things matter: keep it away from the
    # nodes at either end (a label sitting on a node circle is unreadable), and
    # push it clear of the connection itself. The jitter is what stops two
    # labels on converging lines from landing on top of each other, so it stays
    # -- just narrowed to the middle third of the line. Pass label_pos (0 = at
    # the start node, 1 = at the end node) to place labels deterministically
    # instead, which is what you want when several connections converge on one
    # unit and random placement keeps colliding.
    pos_adjust = np.random.uniform(0.34, 0.66) if label_pos is None else label_pos

    # The label sits just above the line, perpendicular to it. The gap is in
    # POINTS, not data units, so it stays a hairline whatever size the figure
    # is drawn at -- a data-space offset looks right at one scale and leaves
    # the label floating far above the line at another.
    perp_x, perp_y = -dir_y, dir_x
    if perp_y < 0:                       # keep the label on the upper side
        perp_x, perp_y = -perp_x, -perp_y

    text_pos_x = start_adj[0] + pos_adjust * (end_adj[0] - start_adj[0])
    text_pos_y = start_adj[1] + pos_adjust * (end_adj[1] - start_adj[1])

    style = f"Simple, head_width={arrow_size}, head_length={arrow_size}"
    
    if bow:
        # Creating a curved path for the arrow
        # Midpoint for placing text
        midpoint_x = (start_adj[0] + end_adj[0]) / 2
        midpoint_y = (start_adj[1] + end_adj[1]) / 2
        text_pos_x = midpoint_x
        text_pos_y = midpoint_y + bow_offset  # Offset to position text above the bow

        control_point = ((start_adj[0] + end_adj[0]) / 2, (start_adj[1] + end_adj[1]) / 2 + 1.0)
        #print(f'start_adj {start_adj}, end_adj {end_adj}')
        #start_adj = tuple(item - 0.1 for item in start_adj)
        more_start = (-.1, 0)
        more_end = (.1, 0.1)
        start_adj = tuple(a + b for a, b in zip(start_adj, more_start))
        end_adj = tuple(a + b for a, b in zip(end_adj, more_end))
        #print(f'start_adj {start_adj}, end_adj {end_adj}')
        path = patches.FancyArrowPatch(start_adj, end_adj, connectionstyle=f"arc3,rad=0.3", arrowstyle=style, color=edge_color, linewidth=line_width)
        ax.add_patch(path)
    else:
        ax.annotate('', xy=end_adj, xytext=start_adj, arrowprops=dict(arrowstyle='->', color=edge_color, linewidth=line_width))

    # Place the weight just above the line and rotated to match it. va='bottom'
    # with rotation_mode='anchor' sits the text ON the line; the offset below
    # lifts it clear of the stroke by label_gap points and no more.
    if show_weight:
        gap = label_gap + line_width / 2.0
        ax.annotate(format(weight, weight_fmt), xy=(text_pos_x, text_pos_y),
                    xycoords='data', textcoords='offset points',
                    xytext=(gap * perp_x, gap * perp_y),
                    horizontalalignment='center', verticalalignment='bottom',
                    rotation=angle, rotation_mode='anchor',
                    fontsize=font_size, color=text_color)

