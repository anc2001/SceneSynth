import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np

from main.compiler import convert_mask_to_image
from main.config import \
    constraint_types, direction_types, object_types, \
    constraint_types_map, \
    grid_size

def draw_graph(graph, label_dict, reference_dict, figsize=(15, 15)):
    # same layout using matplotlib with no labels
    fig = plt.figure(figsize = figsize)
    pos = graphviz_layout(graph, prog='dot')
    pos = {node: (x, y) for node, (x,y) in pos.items()}

    num_cols = len(reference_dict)

    num_rows = 8
    ax_tree = plt.subplot2grid((num_rows, num_cols), (1,0), fig = fig, rowspan=num_rows-1, colspan=num_cols)
    for i, (name, image) in enumerate(reference_dict.items()):
        ax = plt.subplot2grid((num_rows, num_cols), (0, i), fig = fig, rowspan=1, colspan=1, title=name)
        ax.imshow(image)
        ax.axis('off')

    nx.draw(
        graph, pos, 
        ax=ax_tree,
        arrows=True,
        arrowstyle="-"
#             min_source_margin=15,
#             min_target_margin=15,
            # font_size=15,
            # node_size=node_sizes,
            # node_color="white",
            # labels=label_dict, with_labels=True
    )
    tr_figure = ax_tree.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform
    icon_size = 0.2
    icon_center = icon_size / 2.0

    # Add the respective image to each node
    for n in graph.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        # get overlapped axes and plot icon
        label = label_dict[graph.nodes[n]['node_id']]
        a = plt.axes(
            [xa - icon_center, ya - (icon_center), icon_size, icon_size], 
            title= label
        )
        a.imshow(graph.nodes[n]["img"])
        a.axis("off")
    return fig

def visualize_program(program_tree, scene, query_object):
    graph = nx.DiGraph()

    scene_image = scene.convert_to_image()
    def add_nodes_recurs(node, id):
        if node.is_leaf():
            image = convert_mask_to_image(node.mask, scene_image)
            constraint = node.constraint

            label = [
                constraint_types[constraint[0]],
                f"object_{constraint[1]}",
                f"object_{constraint[2]}",
                direction_types[constraint[3]]
            ]
            
            label = "\n".join(label)
            graph.add_node(
                id, node_id=id, type='c', 
                constraint = label, img = image
            )

            return id + 1
        else:
            image = convert_mask_to_image(node.mask, scene_image)
            graph.add_node(
                id, node_id= id, type = node.type, img = image
            )

            left_id = id + 1
            right_id = add_nodes_recurs(node.left, left_id)
            next_id = add_nodes_recurs(node.right, right_id)
            
            graph.add_edge(id, left_id)
            graph.add_edge(id, right_id)
            
            return next_id

    add_nodes_recurs(program_tree.root, 0)
    final_image = convert_mask_to_image(program_tree.mask, scene_image)
    graph.add_node(-1, node_id = -1, type = 'final', img = final_image)
    graph.add_edge(-1, 0)

    label_dict = {}
    for x in graph.nodes:
        id = graph.nodes[x]['node_id']
        label = graph.nodes[x]['type']
        if label == 'c':
            label = graph.nodes[x]['constraint']
        label_dict[id] = label

    # Top bar of relevant images 
    reference_dict = dict()
    ground_truth = scene.convert_to_image(query_object=query_object, with_query_object=True)
    reference_dict["ground truth scene"] = ground_truth

    wall_image = scene.convert_to_image(empty=True)
    reference_dict["object_0 - wall"] = wall_image

    for i, object in enumerate(scene.objects[1:]):
        image = np.ones((grid_size, grid_size, 3))
        object.write_to_image(scene, image, normalize= True)
        image = np.rot90(image)
        reference_dict[f"object_{i+1} - {object_types[object.id]}"] = image

    image = np.ones((grid_size, grid_size, 3))
    query_object.write_to_image(scene, image, normalize= True)
    image = np.rot90(image)
    reference_dict[f"object_{len(scene.objects)} - {object_types[query_object.id]}"] = image
    
    fig = draw_graph(graph, label_dict, reference_dict, figsize=(15, 17))

    return fig

