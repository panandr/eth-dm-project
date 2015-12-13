#!/usr/bin/env python2.7

import numpy as np
# import numpy.random
from numpy.linalg import inv

sigma = 0.3
# alpha =1.35
alpha = 1 + np.sqrt(np.log(2 / sigma) / 2)
Dim_user = 6    # dimension of user features
Dim_arti = 6    # dimension of article features

# set the input the data
M = dict()  # Key: article ID. Value: matrix M for Hybrid LinUCB algorithm.
M_inv = dict() # Key: article ID. Value: inverted matrix M.
b = dict()  # Key: article ID. Value: number b for Hybrid LinUCB algorithm.
w = dict()  # Key: article ID. Value: weights w for Hybrid LinUCB algorithm.
M_0 = dict() # Key: article ID. Value: matrix M_0 for Hybrid LinUCB algorithm.
M_0_inv = dict() # Key: article ID. Value: inverted matrix M_0
b_0 = dict() # Key: article ID. Value: number b_0 for Hybrid LinUCB algorithm.
B = dict()  # Key: article ID. Value: B for Hybrid LinUCB algorithm.
beta = dict() # Key: article ID. Value: beta.

article_list = None
article_feature = None

# Remember last article and user so we can use this information in update() function.
last_article_id = None
last_user_features = None

def set_articles(articles):
    """Initialise whatever is necessary, given the articles."""

    global M, M_inv, b, w, M_0, M_0_inv, b_0, beta, B
    global article_feature
    global article_list

    M_0 = np.identity(Dim_user)
    M_0_inv = np.identity(Dim_user)
    b_0 = np.zeros((Dim_user))

    # beta = M_0_inv.dot(b_0)

    # Make a list of article ID-s
    if isinstance(articles, dict):
        article_list = [x for x in articles]        # If 'articles' is a dict, get all the keys
    else:
        article_list = [(x[0]) for x in articles]     # If 'articles' is a matrix, get 1st element from each row
        article_feature = [x[1:] for x in articles]

    for article_id in article_list:
        # Initialise M and b
        M[article_id] = np.identity(Dim_arti)
        M_inv[article_id] = np.identity(Dim_arti)
        B[article_id] = np.zeros((Dim_arti, Dim_user))
        b[article_id] = np.zeros((Dim_arti, 1))
        w[article_id] = np.zeros((Dim_arti, 1))


def reccomend(time, user_features, articles):
    """Recommend an article."""
    best_article_id = None
    best_ucb_value = -1
    best_article_feature = None

    user_features = np.asarray(user_features)
    user_features.shape = (Dim_user, 1)
    z_t = user_features

    global beta

    beta = M_0_inv.dot(b_0)
    beta = np.asarray(beta)
    # beta.shape = (Dim_user, 1)
    # print(beta)

    # article_feature = np.asarray(article_feature)

    # print(np.asarray(article_feature))
    # print(article_list)
    for article_id in articles:
        # If we haven't seen article before
        if article_id not in M:
            # Initialise this article's variables
            M[article_id] = np.identity(Dim_arti)
            M_inv[article_id] = np.identity(Dim_arti)
            B[article_id] = np.zeros((Dim_arti, Dim_user))
            b[article_id] = np.zeros((Dim_arti, 1))
            w[article_id] = np.zeros((Dim_arti, 1))

            # Get at least 1 data point for this article
            best_article_id = article_id
            break

        # If we have seen article before
        else:
            temp = np.asarray(article_feature)
            # print(temp[1])

            x_t = temp[list(article_list).index(article_id)]
            x_t = np.asarray(x_t)
            x_t.shape = (Dim_arti, 1)

            w[article_id] = M_inv[article_id].dot(b[article_id] - (B[article_id].dot(beta)))
            # print(w[article_id])
            s_t = (z_t).T.dot(M_0_inv).dot(z_t) -\
                2 * (z_t).T.dot(M_0_inv).dot(B[article_id].T).dot(M_inv[article_id]).dot(x_t) +\
                (x_t.T).dot(M_inv[article_id]).dot(x_t) +\
                (x_t.T).dot(M_inv[article_id]).dot(B[article_id]).dot(M_0_inv).\
                dot(B[article_id].T).dot(M_inv[article_id]).dot(x_t)
            # print(beta)

            ucb_value = z_t.T.dot(beta) + x_t.T.dot(w[article_id]) + alpha * np.sqrt(s_t)
            ucb_value = np.min(np.min(ucb_value, 1))

            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_article_id = article_id
                best_article_feature = x_t

    global last_article_id
    last_article_id = best_article_id   # Remember which article we are going to recommend
    global last_user_features
    last_user_features = z_t  # Remember what the user features were
    global last_article_feature
    last_article_feature = best_article_feature

    return best_article_id

def update(reward):
    """Update our model given that we observed 'reward' for our last recommendation."""

    if reward == -1:    # If the log file did not have matching recommendation
        return
    else:
        # Update M, b and weights
        r_t = reward
        x_at = last_article_feature
        z_at = last_user_features

        M_0 += B[last_article_id].T.dot(M_inv[last_article_id]).dot(B[last_article_id])
        # M_0_inv = inv(M_0)
        b_0 += B[last_article_id].T.dot(M_inv[last_article_id]).dot(b[last_article_id])
        M[last_article_id] += x_at.dot(x_at.T)
        M_inv[last_article_id] = inv(M)
        B[last_article_id] += x_at.dot(z_at.T)
        b[last_article_id] += r_t * x_at
        M_0 += z_at.dot(z_at.T) - B[last_article_id].T.dot(M_inv[last_article_id]).dot(B[last_article_id])
        M_0_inv = inv(M_0)
        b_0 += r_t * z_at - B[last_article_id].T.dot(M_inv[last_article_id]).dot(B[last_article_id])
