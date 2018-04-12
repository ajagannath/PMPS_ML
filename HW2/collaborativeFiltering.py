"""
Collaborative filterinf module
"""
#!/usr/bin/env python3

import multiprocessing
import math
import operator


# Just to store shared data
class CFSharedVals:
    pass
__m = CFSharedVals()


def train(data):
    ''' Init with all shared data and multiprocessing '''
    
    # organize data
    result = organize_data(data)
    __m.item_users = result['item_users']
    __m.user_votes = result['user_votes']


    # For multi threading
    __m.threads = 8
    __m.pool = multiprocessing.Pool(processes=__m.threads)


def organize_data(data):
    ''' Get all the users who watched a movie  '''

    item_users = {}
    user_votes = {}
    for row in data:
        item = row[0]
        user = row[1]
        vote = float(row[2])

        if item in item_users:
            item_users[item]['users'].append(user)
            item_users[item]['sum_votes'] += vote
            item_users[item]['num_votes'] += 1
        else:
            item_users[item] = {'users': [user], 'sum_votes': vote, 'num_votes': 1}

        if user in user_votes:
            user_votes[user]['items'][item] = vote
            user_votes[user]['sum_votes'] += vote
            user_votes[user]['num_votes'] += 1
        else:
            user_votes[user] = {'items': {item:vote}, 'sum_votes': vote, 'num_votes': 1}

    return {'item_users': item_users, 'user_votes': user_votes}


def correlation(user_a_votes, mean_vote_a, user_i_votes, mean_vote_i):
    ''' Correlation of user_a with resepct to user_i '''
    
    n_sum = 0.0   # numerator sum
    d1_sum = 0.0  # denominator term1 sum
    d2_sum = 0.0  # denominator term2 sum
    
    for item in user_a_votes:
        if item in user_i_votes:
            t1 = user_a_votes[item] - mean_vote_a
            t2 = user_i_votes[item] - mean_vote_i
            n_sum  += (t1 * t2)
            d1_sum += (t1 * t1)
            d2_sum += (t2 * t2)
        
    if n_sum > 0.0 and d1_sum > 0.0 and d2_sum > 0.0:
        corr = n_sum / math.sqrt( d1_sum * d2_sum )
    else:
        corr = 0
    
    return corr


def predicted_vote(user_a, item):
    ''' gets the predicted vote of the given user for the given item '''

    user_votes = __m.user_votes
    item_users = __m.item_users[item]

    user_a_votes = user_votes[user_a]['items']
    mean_vote_a  = user_votes[user_a]['sum_votes'] / user_votes[user_a]['num_votes']

    n_sum = 0.0
    inv_alpha = 0.0
    for user_i in item_users['users']:

        # data corresponding to user i
        user_i_votes = user_votes[user_i]['items']
        mean_vote_i  = user_votes[user_i]['sum_votes'] / user_votes[user_i]['num_votes']

        # If user's mean vote is useless, dont count on this user
        if mean_vote_i < 1.5 or mean_vote_i > 4.5:
            continue
        
        corr = correlation(user_a_votes, mean_vote_a, user_i_votes, mean_vote_i)
        inv_alpha += corr
        n_sum += corr * (user_i_votes[item] - mean_vote_i)
    
    if inv_alpha != 0:
        pred_vote_a = mean_vote_a +  (n_sum / inv_alpha)
    else:
        pred_vote_a = mean_vote_a

    if pred_vote_a > 5.0:
        pred_vote_a = 5.0

    return pred_vote_a


def predicted_vote_helper(user_a_votes, mean_vote_a, item, user_is):
    ''' Helper function to help parallel processing  '''

    user_votes = __m.user_votes

    n_sum = 0.0
    inv_alpha = 0.0
    for user_i in user_is:
        # data corresponding to user i
        user_i_votes = user_votes[user_i]['items']
        mean_vote_i  = user_votes[user_i]['sum_votes'] / user_votes[user_i]['num_votes']

        # If user's mean vote is useless, dont count on this user
        if mean_vote_i < 1.5 or mean_vote_i > 4.5:
            continue

        corr = correlation(user_a_votes, mean_vote_a, user_i_votes, mean_vote_i)
        inv_alpha += corr
        n_sum += corr * (user_i_votes[item] - mean_vote_i)
    
    return {'n_sum': n_sum, 'inv_alpha':inv_alpha}


def predicted_vote_mt(user_a, item):
    '''  Multi threaded version of predicted vote '''

    user_votes = __m.user_votes
    item_users = __m.item_users[item]

    user_is = item_users['users']
    num_users = len(user_is)

    # Number of parts we need to split them into
    num_threads = __m.threads

    if num_users < num_threads:
        return predicted_vote(user_a, item)
    

    # Split them even to process parallely
    user_is_array = [ user_is[ (i * num_users // num_threads) : ((i+1) * num_users) // num_threads ] for i in range(num_threads) ]


    user_a_votes = user_votes[user_a]['items']
    mean_vote_a  = user_votes[user_a]['sum_votes'] / user_votes[user_a]['num_votes']

    results = __m.pool.starmap(predicted_vote_helper, [(user_a_votes, mean_vote_a, item, sub_user_is) for sub_user_is in user_is_array])

    n_sum = 0.0
    inv_alpha = 0.0
    for result in results:
        n_sum     += result['n_sum']
        inv_alpha += result['inv_alpha']
    
    if inv_alpha != 0:
        pred_vote_a = mean_vote_a +  (n_sum / inv_alpha)
    else:
        pred_vote_a = mean_vote_a

    if pred_vote_a > 5.0:
        pred_vote_a = 5.0

    return pred_vote_a


def predicted_movies(user):
    ''' Movie suggestions  '''
    all_movies = [ movie for movie in __m.item_users]

    ratings = {}
    for movie in all_movies:
        if movie not in __m.user_votes[user]['items']:
            ratings[movie] = predicted_vote_mt(user, movie)

    sorted_ratings = dict(sorted(ratings.items(), key=operator.itemgetter(1), reverse=True))
    
    return sorted_ratings