import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
                                arrow_size=6, plot_width=1, font_size=1.0, node_radius=0.05, edge_color='Default', bow_offset=0.07):
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
        if edge_color == 'Default':
            edge_color = 'red'
    else:
        if edge_color == 'Default':
            edge_color = 'black'

    text_color = edge_color
    
    # Position for weight text: random between 10% and 90% of line length
    pos_adjust =  np.random.uniform(0.1, 0.9)

    text_pos_x = start_adj[0] + pos_adjust * (end_adj[0] - start_adj[0])
    text_pos_y = start_adj[1] + pos_adjust * (end_adj[1] - start_adj[1]) + .15 # + .15 moves it off the line

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

    # Placing weight text at the calculated position and perfectly aligned with the line
    if show_weight:
        ax.text(text_pos_x, text_pos_y, f'{weight:.4f}', horizontalalignment='center', verticalalignment='center', 
                rotation=angle, fontsize=font_size, color=text_color)

