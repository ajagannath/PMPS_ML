"""
Classification Module
"""
#!/usr/bin/env python3

__author__ = "Abhijith Jagannath"
__email__ = "jabhi@uw.edu"

import copy
from collections import Counter

def analyze_dtree(dtree_orig):
    '''
    Analyze the tree to determine how many nodes it has, etc..
    '''

    # Deep copy tree, we are about to recurse and go deep!
    dtree = copy.deepcopy(dtree_orig)

    node_order = []
    num_nodes = 0
    num_leaves = 0
   # invalid_vals = ['best_attr_index', 'most_common_value', 'least_common_value', 'decision']

    # Could have written a while loop, 
    # Python newbee here: never thought I would do something like this!
    def next_depth(d):
        nonlocal node_order
        nonlocal num_nodes
        nonlocal num_leaves

        if 'decision' in d:
            num_leaves += 1
        else:
            node_attr = d["best_attr_index"]
            node_order.append(node_attr)
            num_nodes += 1

        for k, v in d.items():
            if isinstance(v, dict):
                next_depth(v)

    next_depth(dtree)

    return {'num_nodes': num_nodes, 'node_order': node_order[:4], 'num_leaves': num_leaves}


def classify(data, dtree, target_attr_index):
    ''' Classifies the data according to the dicision tree given '''

    total_records = 0
    right_predictions = 0

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # To trace which path the record took to come to decision
    p_paths = []
    n_paths = []

    for row in data:
        total_records += 1

        path = []
        predicted_value = get_decision(row, dtree, path)
        actual_value    = row[target_attr_index]

        if actual_value == predicted_value:
            right_predictions += 1
            if predicted_value == 'True':
                true_positives += 1
        else:
            if predicted_value == 'True':
                false_positives += 1
            else:
                false_negatives += 1
        
        # Positively / negetively labled examples
        if actual_value == 'True':
            p_paths.append(path)
        else:
            n_paths.append(path)
        
    accuracy  = 100 * right_predictions / total_records
    precision = 100 * true_positives / (true_positives + false_positives)
    recall    = 100 * true_positives / (true_positives + false_negatives)

    return {"accuracy" : accuracy, "precision" : precision, "recall": recall, 'p_paths': p_paths, 'n_paths': n_paths }

def get_decision(row, dtree, path):
    ''' Decision for each row from the tree '''

    # Empty tree ? ( Should not happen for this project )
    if not dtree:
        print("DEBUG: No path leading to data")
        return None

    # Is dtree even a hash? 
    if not isinstance(dtree,dict):
        print("DEBUG: Invalid tree passed", dtree)
        return None

    # if we are at a leaf
    if 'decision' in dtree.keys():
        return dtree['decision']

    # prepare to go inside the node
    best_attr_index = dtree["best_attr_index"]
    
    # What is the path to take? (value of the best attribute in the test record.)
    attr_value = row[best_attr_index]
    
    # It is unknown, so use the most common we saw while training
    if not attr_value:
        attr_value = dtree["most_common_value"]
    
    # Its not unknown, but we didnt see this in training data ?
    # May be we did, but data in this branch did not have it, hence we dont have anything for it!
    # We can say that, for this branch, it is a rare occurance.
    # so lets assume it as least common value
    if attr_value not in dtree.keys():
        attr_value = dtree["least_common_value"]  

    # Add this attr and value to path to trace!
    path.append((best_attr_index, attr_value))

    return get_decision(row, dtree[attr_value], path)

def majority_path(paths):
    ''' Gets the path that is majority in the given paths '''

    # How many have we got?
    total = len(paths)

    ret = []

    # First get the most common root 
    roots = [ path[0] for path in paths if len(path)>0 ]
    root_counter = Counter(roots)
    mcp = root_counter.most_common()[0][0]
    mcp_count = root_counter.most_common()[0][1]
    ret.append(mcp)

    # If most common we got is less than half? 
    if mcp_count / total < 0.5:
        return ret

    # Try the next one
    first_nodes = [ path[1] for path in paths if (len(path)>1 and path[0]== mcp) ]
    node1_counter  = Counter(first_nodes)
    mcp_node1 = node1_counter.most_common()[0][0]
    mcp_node1_count = node1_counter.most_common()[0][1]
    ret.append(mcp_node1)
    
    # If most common we got is less than half? 
    if mcp_node1_count / total < 0.5:
        return ret
    
     # Try the next one
    second_nodes = [ path[2] for path in paths if (len(path)>2 and path[1] == mcp_node1) ]
    
    if not second_nodes:
        return ret
    
    node2_counter  = Counter(second_nodes)    
    mcp_node2 = node2_counter.most_common()[0][0]
    mcp_node2_count = node2_counter.most_common()[0][1]
    ret.append(mcp_node2)

    # If most common we got is less than half? 
    if mcp_node2_count / total < 0.5:
        return ret

    # Try the next one
    third_nodes = [ path[3] for path in paths if (len(path)>3 and path[2] == mcp_node2) ]
    
    if not third_nodes:
        return ret
    
    node3_counter  = Counter(third_nodes)
    mcp_node3 = node3_counter.most_common()[0][0]
    mcp_node3_count = node3_counter.most_common()[0][1]
    ret.append(mcp_node3)

     # If most common we got is less than half? 
    if mcp_node3_count / total < 0.5:
        return ret
    
    # Try the next one
    fourth_nodes = [ path[4] for path in paths if (len(path)>4 and path[3] == mcp_node3) ]
    
    if not fourth_nodes:
        return ret
    
    node4_counter  = Counter(fourth_nodes)
    mcp_node4 = node4_counter.most_common()[0][0]
    mcp_node4_count = node4_counter.most_common()[0][1]
    ret.append(mcp_node4)

     # If most common we got is less than half? 
    if mcp_node4_count / total < 0.5:
        return ret
    
    # Try the next one
    fifth_nodes = [ path[5] for path in paths if (len(path)>5 and path[4] == mcp_node4) ]
    
    if not fifth_nodes:
        return ret
    
    node5_counter  = Counter(fifth_nodes)
    mcp_node5 = node5_counter.most_common()[0][0]
    mcp_node5_count = node5_counter.most_common()[0][1]
    ret.append(mcp_node5)

    return ret