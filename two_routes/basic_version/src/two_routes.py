# Fri.Oct.03.152000.2025
######################################################################
# Blue-Red Game
# T periods: 0, ..., T-1
#
# Blue sends 1 unit along either route I or II
#   policy:
#     t = 0: Send along I or II w.p. 1/2
#     t > 0:
#       a) If Red struck in period t-1,
#          Blue stays on previous route w.p. pB and switches w.p. 1-pB
#       b) If Red did not strike in period t-1,
#          Blue stays on previous route w.p. qB and switches w.p. 1-qB
#   goal: maximize throughput
#
# Red can be located at route I or II
#   policy:
#     t = 0: Red starts at I (unknown to Blue)
#     t > 0:
#       a) If Red struck in period t-1,
#          Red stays on previous route w.p. pR and switches w.p. 1-pR
#       b) If Red did not struck in period t-1,
#          Red stays on previous route w.p. qR and switches w.p. 1-qR
#   goal: minimize Blue throughput
#
# *Actual probabilities used are chosen uniformly from region (prob. - delta, prob. + delta)
#
# state: whether Red struck in previous period
# action: whether Blue/Red switches routes or not
# Blue reward: 1 if Blue shipment is not struck by Red, 0 otherwise
# Red reward: 1 if Red strikes Blue shipment, 0 otherwise
#
#######################################################################
import numpy as np
import random


choose_to_stay = lambda pr, d: random.random() < random.uniform(max(0, pr - d), min(1, pr + d))

safe_inverse = lambda x: 1 / (x + 1e-6)


def play_game(T, thetaBlue, thetaRed, delta=0):
    '''
    Computes a trajectory for the given theta=(p, q).
    Returns that trajectory, i.e. the states, actions, and rewards over the course of the game.
    '''
    pB = thetaBlue[0]
    qB = thetaBlue[1]
    pR = thetaRed[0]
    qR = thetaRed[1]
    
    state = np.zeros(T+1, dtype=int)  # Whether Red struck previously, for each period t (i.e. did Red strike in t-1)
    actionBlue = np.zeros(T, dtype=int)  # Whether Blue switches routes in each period t
    actionRed = np.zeros(T, dtype=int)  # Whether Red switches routes in each period t
    reward = np.zeros(T, dtype=int)  # Whether Blue shipment gets through at each t (i.e. if Red not located on route Blue uses)

    # Handle first period separately:
    # state is 0, Red has not struck previously, tautologically
    # actions are 0, niether Blue or Red switch
    reward[0] = 0 if random.random() < 0.5 else 1  # Blue shipment gets through with probability 1/2
    state[1] = 1 - reward[0]  # state is now whether or not Red just struck, i.e. if Blue shipment did not go through

    # Handle all other periods
    for t in range(1, T):
        if state[t]:
            # Scenario (a): Red struck Blue in period t-1
            aB = 0 if choose_to_stay(pB, delta) else 1
            aR = 0 if choose_to_stay(pR, delta) else 1
            r = 0 if aB == aR else 1  # Red strikes again if both switch
        else:
            # Scenario (b): Red did not strike Blue shipment
            aB = 0 if choose_to_stay(qB, delta) else 1
            aR = 0 if choose_to_stay(qR, delta) else 1
            r = 0 if aB != aR else 1  # Red strikes if only one switches

        actionBlue[t] = aB
        actionRed[t] = aR
        reward[t] = r
        state[t+1] = 1 - r
        
    return state, actionBlue, actionRed, reward


def grad_log_prob(theta, action, state, t):
    '''
    Computes the gradient of the log probability of action[t] given state[t]
    '''
    # First period has no theta dependence
    if t == 0:
        return np.zeros(2, dtype=float)
    
    p = theta[0]
    q = theta[1]

    gradLogProb = np.zeros(2, dtype=float)
    if state[t]:
        # Scenario (a): Red struck Blue shipment in period t-1
        gradLogProb[0] = safe_inverse(p) if action[t] == 0 else safe_inverse(1 - p)
    else:
        # Scenario (b): Red did not strike Blue shipment in period t-1
        gradLogProb[1] = safe_inverse(q) if action[t] == 0 else safe_inverse(1 - q)

    return gradLogProb
