"""
Decision Tree module
"""
#!/usr/bin/env python3

__author__ = "Abhijith Jagannath"
__email__ = "jabhi@uw.edu"

import multiprocessing
import math
import copy
from scipy.stats import chi2

# Just to store shared data
class DTreeSharedVals:
    pass
__m = DTreeSharedVals()

def dtree_init(data, attr_details, target_attr, confidence):
    ''' 
    Initialization function
    Variables stored here are reused and not be modified anywhere in the program !!
    '''
    # Initial data and target variable
    __m.data = data
    __m.row_numbers = [ i for i in range(len(data))]
    __m.target_attr = target_attr
    __m.attr_details = attr_details[:]

    # For chi2 critical value
    if confidence and confidence >= 0.0:
        __m.confidence = confidence
    else:
        __m.confidence = None

    # For multi threading
    __m.threads = 8
    __m.pool = multiprocessing.Pool(processes=__m.threads)
        
def dtree_close():
    ''' Clear all shared data '''

     # Clsoing threads is what is important!
    __m.pool.close()
    __m.pool.join()
    __m.pool.terminate()
    
def create_dtree(data, attr_details, attribute_indecies, target_attr, confidence):
    ''' 
    Wrapper function to create the dtree
    This handels initializations and other things
    (training_data, training_attribute_details, non_target_attribute_indecies, target_attribute_index, confidence)
    '''

    # Initialize the common variables
    dtree_init(data, attr_details, target_attr, confidence)

    # create the tree
    dtree = grow_dtree(attribute_indecies, __m.row_numbers)

    # Take care of threads and other things
    dtree_close()

    # Return the created tree
    return dtree


def grow_dtree(attribute_indecies, row_numbers):
    ''' Creates decision tree  '''
    
    # Start with empty tree
    data = __m.data
    target_attr = __m.target_attr
    dtree = {}
    
    # If there is no more data or no attibutes, return empty tree ?
    if not row_numbers or (len(attribute_indecies) - 1) <= 0:
        # This should not happen. So check whats wrong!!
        print("DEBUG: No data or attributes. Returning empty tree.")
        return dtree

    # Process the given data and get the best attribute to split 
    pd = process_data_mt(row_numbers, attribute_indecies)
    best_attr = pd['best_attr']
    default_decision = pd['decision']

    if not best_attr:
        if not default_decision:
            print("DEBUG: Decision is None!!")
        dtree['decision'] = default_decision
        return dtree

    # Other info
    split_data = pd['split_data']

    # Perform test
    if not split_test(split_data):
        # print("INFO: Chi2 failed, Stopping!")
        dtree['decision'] = default_decision
        return dtree
    
    # Store most common and least common values of the best attribute
    dtree["best_attr_index"] = best_attr
    dtree["most_common_value"] = split_data['mcv']
    dtree["least_common_value"] = split_data['lcv']

    # Remaining attributes
    sub_attribute_indecies = [ attr for attr in attribute_indecies if attr != best_attr ]

    # Just cheking
    if best_attr in sub_attribute_indecies:
        print("DEBUG: What is going on")
        exit()

    # Go ahead and build the tree
    counter = split_data['counter']
    for val in counter:
        # Row numbers
        sub_row_nums = counter[val]['row_nums']
        
        # Create a subtree and assign it as a node
        dtree[val] = grow_dtree(sub_attribute_indecies, sub_row_nums)
   
    # debug check
    if not isinstance(dtree,dict):
        print("DEBUG: Invalid tree", dtree)

    return dtree

def split_test(split_data):
    ''' 
    Tests if we should split or not 
    True - Yes split, grow the tree
    False - No dont split. Stop here!
    '''

    # Node positives, negetives and total
    Np  = split_data['target']['p_count']
    Nn  = split_data['target']['n_count']
    Nt  = split_data['target']['count']

    # If the node is pure, dont grow stop!
    if Np == 0 or Nn ==0 :
        return False

     # If confidence is not given just let the tree grow
    if not __m.confidence:
        return True

    # To be calculated
    Chi2 = 0.0

    # Critical value calculation
    counter = split_data['counter']
    q  = 1 - (__m.confidence / 100.0)
    df = len(counter) - 1
    cv = chi2.isf(q, df)

    # Lets begin summing up!
    for val in counter:
        # Expected results
        Et = counter[val]['count']
        Ep = Et * Np / Nt
        En = Et * Nn / Nt
       
        # Observed results
        Op = counter[val]['p_count']
        On = counter[val]['n_count']

        # Chi2 summation
        Chi2 += ((Op - Ep) * (Op - Ep)) / Ep
        Chi2 += ((On - En) * (On - En)) / En

    # If calculated value is greater than the critical, then split ( continue growing )
    return (Chi2 > cv)

def entropy(counts):
    ''' Calculate the entropy, given the counts '''
    
    p_count = counts['p_count']
    n_count = counts['n_count']
    total   = counts['count']
    
    if p_count + n_count != total:
        print("DEBUG: counts do not add up!")
        exit()
    
    entropy = 0.0

    if p_count > 0.0:
        p_prob = p_count / total
        entropy += (-p_prob) * math.log2(p_prob)
    
    if n_count > 0.0:
        n_prob = n_count / total
        entropy += (-n_prob) * math.log2(n_prob)

    return entropy

def process_attr(attr, row_numbers):
    ''' Process data for each attribute and collect various information '''

    # use global data
    data = __m.data
    target_attr = __m.target_attr
    attr_valid_vals = __m.attr_details[attr][1]

    # What we want to return 
    counter = {}

    # Target's total, positive and negetive counts
    t_count  = 0
    t_pcount = 0
    t_ncount = 0
    
    # For this attribute
    highest_count = 0
    least_count   = float('inf')
    mcv = None
    lcv = None

    # For counting unknowns
    unknowns = {'count': 0, 'p_count': 0, 'n_count': 0, 'row_nums': []}

    # start processing one row at a time
    for row_num in row_numbers:    
        
        row = data[row_num]   # record we are considering
        attr_val = row[attr]  # value of the attribute in this record

        # Total records in data
        t_count += 1
            
        # Row result : T / F -> 1 / 0
        if row[target_attr] == 'True':
            p_row_result = 1 
            n_row_result = 0
        else:
            n_row_result = 1
            p_row_result = 0
        
        t_pcount += p_row_result
        t_ncount += n_row_result

        # If its an unkown value '?' or not in the valid list mentioned in arff
        if attr_val not in attr_valid_vals:
            unknowns['count'] += 1
            unknowns['p_count'] += p_row_result
            unknowns['n_count'] += n_row_result
            unknowns['row_nums'].append(row_num)

        # Count number of values
        elif attr_val in counter:
            counter[attr_val]['count'] += 1
            counter[attr_val]['p_count'] += p_row_result
            counter[attr_val]['n_count'] += n_row_result
            counter[attr_val]['row_nums'].append(row_num)
        else:
            counter[attr_val] = {
                'count': 1, 
                'p_count' : p_row_result,
                'n_count' : n_row_result,
                'row_nums': [row_num]
            } 

        if attr_val in counter:
            if counter[attr_val]['count'] >= highest_count:
                highest_count = counter[attr_val]['count']
                mcv = attr_val

            if counter[attr_val]['count'] <= least_count:
                least_count = counter[attr_val]['count']
                lcv = attr_val


    # Target's values which we got for free
    ret_val = {}
    ret_val['target'] = {}
    ret_val['target']['count']   = t_count
    ret_val['target']['p_count'] = t_pcount
    ret_val['target']['n_count'] = t_ncount

    # If there is no mcv, that means only unkonwns! this attribute will be useless at this point
    if mcv:
        # If there are unknowns, count them to most common value
        if unknowns['count'] > 0:
            counter[mcv]['count']    += unknowns['count']
            counter[mcv]['p_count']  += unknowns['p_count']
            counter[mcv]['n_count']  += unknowns['n_count']
            counter[mcv]['row_nums'] += unknowns['row_nums']

        # Only one value at this branch? 
        # This mostly not be picked for splitting !
        if not lcv:
            lcv = mcv
            if len(counter) != 1:
                print("DEBUG: counter issue: ", len(counter), "\nCK: ", counter.keys())

        # Store return values
        ret_val['lcv'] = lcv
        ret_val['mcv'] = mcv
        ret_val['counter'] = counter
    
    return ret_val

def process_data(row_numbers, attribute_indecies):
    ''' Compute the best attribute to split the data '''

    # use global data
    data = __m.data
    target_attr = __m.target_attr

    best_gain_ratio = 0.0    # Start with 0
    best_attr       = None   # Start with no best attribute
    node_entropy    = None   # Entropy of target attribute at this node.
    
    # lets not throw away all that we calcualted
    split_data = {}
    decision = None   # Used when best attribute is not available

    for attr in attribute_indecies:        

        # Gather all the information
        ret_val = process_attr(attr, row_numbers)

        # Use when you cant find best attr
        if not decision:
            if ret_val['target']['p_count'] >= ret_val['target']['n_count']:
                decision = 'True'
            else:
                decision = 'False'

        if 'counter' not in ret_val:
            # print("DEBUG: Attribute has only unknown values: ", attr)
            continue
        
        # Store data
        split_data[attr] = ret_val

        # Split info for this node
        node_split_info = 0.0

        counter = ret_val['counter']
        sum_sub_data_entropy = 0.0
        for val in counter:
            val_prob = counter[val]['count'] / ret_val['target']['count']
            sum_sub_data_entropy += val_prob * entropy(counter[val])
            node_split_info += (-val_prob) * math.log2(val_prob)

        # If split information is zero, ignore the variable
        if node_split_info <= 0.0:
            continue

        # Calculate total entropy once, its the same for all attributes
        if node_entropy == None:
            node_entropy = entropy(ret_val['target'])
        
        # Information gain and gain ratio
        node_info_gain  = node_entropy - sum_sub_data_entropy
        gain_ratio = node_info_gain / node_split_info

        if gain_ratio >= best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_attr = attr

    # Send a decision always, you may decide to stop after split test
    result = {'decision' : decision, 'best_attr': best_attr}
    
    # If we did get a candidate, send along all the data
    if best_attr != None:
        sd = split_data[best_attr]
        result[ 'gain_ratio'] = best_gain_ratio
        result[ 'split_data'] = sd

    # Let the garbage collector know, we are done with this.    
    del split_data

    return result

# Parallel processing
def process_data_mt(row_numbers, attribute_indecies):
    ''' Process attributes in parallel'''

    # if len(attribute_indecies) < __m.threads :
    #    print("DEBUG: less than thread indecies!")
    #    return process_data(row_numbers, attribute_indecies)

    # Number of attributes to process
    num_node_attrs = len(attribute_indecies)

    # Number of parts we need to split them into
    num_threads = __m.threads

    # Split them even to process parallely
    # // -> integer division
    # when parts = 1 => list[0:num_node_attrs]
    attribute_indecies_array = [ attribute_indecies[ (i * num_node_attrs // num_threads) : ((i+1) * num_node_attrs) // num_threads ] for i in range(num_threads) ]

    results = __m.pool.starmap(process_data, [(row_numbers, sub_indecies) for sub_indecies in attribute_indecies_array])

    # Collect all the results
    best_gain_ratio = 0.0
    combined_result = {}
    for result in results:
        if 'decision' not in combined_result:
            combined_result['decision'] = result['decision']
        
        if 'gain_ratio' in result:
            if result['gain_ratio'] >= best_gain_ratio:
                best_gain_ratio = result['gain_ratio']
                combined_result['gain_ratio'] = result['gain_ratio']
                combined_result['best_attr']  = result['best_attr']
                combined_result['split_data'] = result['split_data']
    
    if 'best_attr' not in combined_result:
        combined_result['best_attr'] = None

    return combined_result