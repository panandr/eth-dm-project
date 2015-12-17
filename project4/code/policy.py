#!/usr/bin/env python2.7

import numpy as np
# import numpy.random
from numpy.linalg import inv

np.random.seed(42)

EPSILON = 0.25  # Probability of picking random article to show

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
A0inv_BT_Ainv_x = dict()
#xT_Ainv_B_A0inv_BT_Ainv_x = dict()
#xT_Ainv_x = dict()
xT_w = dict()
xT_Ainv_x_PLUS_xT_Ainv_B_A0inv_BT_Ainv_x = dict()

beta = None
A_0 = None
A_0_inv = None
b_0 = None

article_features = dict() # Key: article ID. Value: article features.
article_count = dict()

# Remember last article and user so we can use this information in update() function.
last_article_id = None
last_user_features = None

def set_articles(articles):
    """Initialise whatever is necessary, given the articles."""

    global A, A_inv, b, w, A_0, A_0_inv, b_0, beta, B
    global A0inv_BT_Ainv_x, xT_w, xT_Ainv_x_PLUS_xT_Ainv_B_A0inv_BT_Ainv_x
    global article_features

    A_0 = np.identity(Dim_user)
    A_0_inv = np.identity(Dim_user)
    b_0 = np.zeros(shape=(Dim_user, 1))

    beta = A_0_inv.dot(b_0)
    beta.shape = (Dim_user, 1)

    # Make a list of article ID-s
    article_features = articles

    for article_id in article_features:
        article_features[article_id] = np.asarray(article_features[article_id])
        article_features[article_id].shape = (Dim_arti, 1)

        # Initialise M and b
        A[article_id] = np.identity(Dim_arti)
        A_inv[article_id] = np.identity(Dim_arti)
        B[article_id] = np.zeros((Dim_arti, Dim_user))
        b[article_id] = np.zeros((Dim_arti, 1))
        w[article_id] = np.zeros((Dim_arti, 1))
        A0inv_BT_Ainv_x[article_id] = np.zeros(shape=(Dim_arti, 1))
        #xT_Ainv_B_A0inv_BT_Ainv_x[article_id] = 0
        #xT_Ainv_x[article_id] = 0
        xT_w[article_id] = 0
        xT_Ainv_x_PLUS_xT_Ainv_B_A0inv_BT_Ainv_x[article_id] = Dim_arti


def reccomend(time, user_features, articles):
    """Recommend an article."""
    best_article_id = None
    best_ucb_value = -1

    user_features = np.asarray(user_features)
    user_features.shape = (Dim_user, 1)
    z_t = user_features
    zT = z_t.T
    double_zT = 2 * zT
    zT_A0inv_z = zT.dot(A_0_inv).dot(z_t)
    #zT_beta = zT.dot(beta)

    if np.random.rand(1) < EPSILON:
        # Pick random article
        best_article_id = articles[np.random.randint(len(articles))]

    else:
        # Pick best article for recommendation
        for article_id in articles:

            # If we don't have article features, just take ones (locally only -- on server we get all features)
            if article_id not in article_features:
                article_features[article_id] = np.ones(shape=(Dim_arti, 1))

            # If we haven't seen article before
            if article_id not in A:
                # Initialise this article's variables
                A[article_id] = np.identity(Dim_arti)
                A_inv[article_id] = np.identity(Dim_arti)
                B[article_id] = np.zeros((Dim_arti, Dim_user))
                b[article_id] = np.zeros((Dim_arti, 1))
                w[article_id] = np.zeros((Dim_arti, 1))
                A0inv_BT_Ainv_x[article_id] = np.zeros(shape=(Dim_arti, 1))
                #xT_Ainv_B_A0inv_BT_Ainv_x[article_id] = 0
                #xT_Ainv_x[article_id] = Dim_arti  # correct initialisation if Ainv is identity matrix and x is ones vector
                xT_w[article_id] = 0
                xT_Ainv_x_PLUS_xT_Ainv_B_A0inv_BT_Ainv_x[article_id] = Dim_arti

            if article_id not in article_count:
                article_count[article_id] = 1
                best_article_id = article_id
                break

            # If we have seen article before
            else:

                s_t = zT_A0inv_z -\
                    double_zT.dot(A0inv_BT_Ainv_x[article_id]) +\
                    xT_Ainv_x_PLUS_xT_Ainv_B_A0inv_BT_Ainv_x[article_id]

                # We don't want to take sqrt of negative number
                if s_t < 0:
                    s_t = 0
                    
                # Don't need zT.dot(beta) because it's constant for all articles
                ucb_value = xT_w[article_id] + alpha * np.sqrt(s_t)

                if ucb_value > best_ucb_value:
                    best_ucb_value = ucb_value
                    best_article_id = article_id

    global last_article_id, last_user_features
    last_article_id = best_article_id   # Remember which article we are going to recommend
    last_user_features = z_t  # Remember what the user features were

    return best_article_id

def update(reward):
    """Update our model given that we observed 'reward' for our last recommendation."""
    global A, A_inv, b, w, A_0, A_0_inv, b_0, beta, B
    global A0inv_BT_Ainv_x, xT_w, xT_Ainv_x_PLUS_xT_Ainv_B_A0inv_BT_Ainv_x
    #global xT_Ainv_B_A0inv_BT_Ainv_x, xT_Ainv_x

    if reward == -1:    # If the log file did not have matching recommendation
        return
    else:
        # Update M, b and weights
        r_t = reward
        x_at = article_features[last_article_id]
        z_at = last_user_features
        xT = x_at.T
        zT = z_at.T

        BT_Ainv = B[last_article_id].T.dot(A_inv[last_article_id])  # Used in many places

        A_0 += BT_Ainv.dot(B[last_article_id])
        A_0_inv = inv(A_0)
        b_0 += BT_Ainv.dot(b[last_article_id])
        A[last_article_id] += x_at.dot(xT)
        A_inv[last_article_id] = inv(A[last_article_id])
        B[last_article_id] += x_at.dot(zT)
        BT_Ainv = B[last_article_id].T.dot(A_inv[last_article_id])  # Recalculate
        b[last_article_id] += r_t * x_at
        A_0 += z_at.dot(zT) - BT_Ainv.dot(B[last_article_id])
        A_0_inv = inv(A_0)
        b_0 += r_t * z_at - BT_Ainv.dot(b[last_article_id])

        w[last_article_id] = A_inv[last_article_id].dot(b[last_article_id] - (B[last_article_id].dot(beta)))

        A0inv_BT_Ainv_x[last_article_id] =\
            A_0_inv\
                .dot(BT_Ainv)\
                .dot(x_at)

        # xT_Ainv_B_A0inv_BT_Ainv_x[last_article_id] =\
        #     xT\
        #         .dot(A_inv[last_article_id])\
        #         .dot(B[last_article_id])\
        #         .dot(A_0_inv)\
        #         .dot(BT_Ainv)\
        #         .dot(x_at)
        #
        # xT_Ainv_x[last_article_id] =\
        #     xT\
        #         .dot(A_inv[last_article_id])\
        #         .dot(x_at)

        xT_Ainv = xT.dot(A_inv[last_article_id])

        xT_w[last_article_id] = xT.dot(w[last_article_id])

        xT_Ainv_x_PLUS_xT_Ainv_B_A0inv_BT_Ainv_x[last_article_id] =\
            xT_Ainv\
                .dot(B[last_article_id])\
                .dot(A_0_inv)\
                .dot(BT_Ainv)\
                .dot(x_at)+\
            xT_Ainv\
                .dot(x_at)