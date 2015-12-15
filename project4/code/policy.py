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
A = dict()          # Key: article ID. Value: matrix M for Hybrid LinUCB algorithm.
A_inv = dict()      # Key: article ID. Value: inverted matrix M.
b = dict()          # Key: article ID. Value: number b for Hybrid LinUCB algorithm.
w = dict()          # Key: article ID. Value: weights w for Hybrid LinUCB algorithm.
B = dict()          # Key: article ID. Value: B for Hybrid LinUCB algorithm.

beta = None
A_0 = None
A_0_inv = None
b_0 = None

article_list = []
article_features = dict() # Key: article ID. Value: article features.

# Remember last article and user so we can use this information in update() function.
last_article_id = None
last_user_features = None

def set_articles(articles):
    """Initialise whatever is necessary, given the articles."""

    global A, A_inv, b, w, A_0, A_0_inv, b_0, beta, B
    global article_list, article_features

    A_0 = np.identity(Dim_user)
    A_0_inv = inv(A_0)
    b_0 = np.zeros(shape=(Dim_user, 1))

    beta = A_0_inv.dot(b_0)
    beta.shape = (Dim_user, 1)

    # Make a list of article ID-s
    if isinstance(articles, dict):
        article_list = [x for x in articles]        # If 'articles' is a dict, get all the keys
        article_features = articles
    else:
        article_list = [(x[0]) for x in articles]     # If 'articles' is a matrix, get 1st element from each row
        for row in articles:
            article_id = row[0]
            article_list.append(article_id)
            article_features[article_id] = row[1:]

    for article_id in article_list:
        # Initialise M and b
        A[article_id] = np.identity(Dim_arti)
        A_inv[article_id] = np.identity(Dim_arti)
        B[article_id] = np.zeros((Dim_arti, Dim_user))
        b[article_id] = np.zeros((Dim_arti, 1))
        w[article_id] = np.zeros((Dim_arti, 1))


def reccomend(time, user_features, articles):
    """Recommend an article."""
    best_article_id = None
    best_ucb_value = -1
    best_article_features = None

    user_features = np.asarray(user_features)
    user_features.shape = (Dim_user, 1)
    z_t = user_features

    for article_id in articles:

        # If we don't have article features, just take 1s (locally only -- on server we get all features)
        if article_id in article_features:
            x_t = np.asarray(article_features[article_id])
            x_t.shape = (Dim_arti, 1)
        else:
            x_t = np.ones(shape=(Dim_arti, 1))

        # If we haven't seen article before
        if article_id not in A:
            # Initialise this article's variables
            A[article_id] = np.identity(Dim_arti)
            A_inv[article_id] = np.identity(Dim_arti)
            B[article_id] = np.zeros((Dim_arti, Dim_user))
            b[article_id] = np.zeros((Dim_arti, 1))
            w[article_id] = np.zeros((Dim_arti, 1))

            # Get at least 1 datapoint for this article
            best_article_id = article_id
            break

        # If we have seen article before
        else:
            w[article_id] = A_inv[article_id].dot(b[article_id] - (B[article_id].dot(beta)))

            s_t = (z_t).T.dot(A_0_inv).dot(z_t) -\
                2 * (z_t).T.dot(A_0_inv).dot(B[article_id].T).dot(A_inv[article_id]).dot(x_t) +\
                (x_t.T).dot(A_inv[article_id]).dot(x_t) +\
                (x_t.T).dot(A_inv[article_id]).dot(B[article_id]).dot(A_0_inv).\
                dot(B[article_id].T).dot(A_inv[article_id]).dot(x_t)

            ucb_value = z_t.T.dot(beta) + x_t.T.dot(w[article_id]) + alpha * np.sqrt(s_t)

            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_article_id = article_id
                best_article_features = x_t

    global last_article_id
    last_article_id = best_article_id   # Remember which article we are going to recommend
    global last_user_features
    last_user_features = z_t  # Remember what the user features were
    global last_article_feature
    last_article_feature = best_article_features

    return best_article_id

def update(reward):
    """Update our model given that we observed 'reward' for our last recommendation."""
    global A, A_inv, b, w, A_0, A_0_inv, b_0, beta, B

    if reward == -1:    # If the log file did not have matching recommendation
        return
    else:
        # Update M, b and weights
        r_t = reward
        x_at = last_article_feature
        z_at = last_user_features

        A_0 += B[last_article_id].T.dot(A_inv[last_article_id]).dot(B[last_article_id])
        A_0_inv = inv(A_0)
        b_0 += B[last_article_id].T.dot(A_inv[last_article_id]).dot(b[last_article_id])
        A[last_article_id] += x_at.dot(x_at.T)
        A_inv[last_article_id] = inv(A[last_article_id])
        B[last_article_id] += x_at.dot(z_at.T)
        b[last_article_id] += r_t * x_at
        A_0 += z_at.dot(z_at.T) - B[last_article_id].T.dot(A_inv[last_article_id]).dot(B[last_article_id])
        A_0_inv = inv(A_0)
        b_0 += r_t * z_at - B[last_article_id].T.dot(A_inv[last_article_id]).dot(b[last_article_id])
