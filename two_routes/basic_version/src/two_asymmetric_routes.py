###############################################################################
# Methods for computing a trajectory of the game and the gradient of the
# logarithm of a policy (Blue and Red).
#
# author: Blake Sisson
# Fri.Oct.10.104100.2025
###############################################################################


# GAME DESCRIPTION
# Players: Blue and Red
# T periods: 0, ..., T-1
#
# Blue sends 1 unit along either route I or II
#   policy:
#     t = 0: Send along route i that has greater Bqi w.p. Bqi
#     t > 0:
#       1a. If used route I and Red struck in period t-1,
#           Blue stays on route I w.p. Bp1 and switches w.p. 1-Bp1
#       1b. If used route I and Red did not strike in period t-1,
#           Blue stays on route I w.p. Bq1 and switches w.p. 1-Bq1
#       2a. If used route II and Red struck in period t-1,
#           Blue stays on route II w.p. Bp2 and switches w.p. 1-Bp2
#       2b. If used route II and Red did not strike in period t-1,
#           Blue stays on route II w.p. Bq2 and switches w.p. 1-Bq2
#   goal: maximize throughput
#
# Red can be located at route I or II
#   Red always strikes (if located on rame route as Blue) and has unlimited budget,
#   but is only successful w.p. Ro1 on route I and Ro2 on route II. 
#   If successful, we say Red hit.
#   policy:
#     t = 0: Red starts on route that has greater Roi (unknown to Blue)
#     t > 0:
#       1a. If Red on route I and struck in period t-1,
#           Red stays on route I w.p. Rp1 and switches w.p. 1-Rp1
#       2b. If Red on route I and did not strike in period t-1,
#           Red stays on previous route w.p. Rq1 and switches w.p. 1-Rq1
#       2a. If Red on route II and struck in period t-1,
#           Red stays on route II w.p. Rp2 and switches w.p. 1-Rp2
#       1b. If Red on route II and did not strike in period t-1,
#           Red stays on previous route w.p. Rq2 and switches w.p. 1-Rq2
#   goal: minimize Blue throughput
#
# (*) Actual probabilities used are chosen uniformly from region (prob. - delta, prob. + delta)
#
# state: locations of Blue and Red at current and previous period (whether Red struck implicitly defined)
# action: whether Blue/Red switches routes or not
# Blue reward: 1 if Blue shipment is not hit by Red, 0 otherwise
# Red reward: 1 if Red hits Blue shipment, 0 otherwise

import numpy as np
import random


#Player indices for readability
Blue, Red = 0, 1

#Routes indices for readability
routeI, routeII = 0, 1

# Samples Bernoulli random variable with parameter pr
with_probability = lambda pr: random.random() < pr

# Determines whether a player (B or R) chooses to stay on current route, given their policy
choose_to_stay = lambda pr, d: with_probability(random.uniform(max(0, pr - d), min(1, pr + d)))

# Determines if Red's strike attempt is successful give the current route and Red's hit odds on said route
red_hits = lambda route, odds: random.random() < odds[route]

# Swtich routes
other_route = lambda route: 1 - route

# Prevents division by zero.
safe_inverse = lambda x: 1 / (x + 1e-6)


def observe_initial_state(hitOdds):
    ''''''
    locBlue = routeI if with_probability(0.5) else routeII
    # if Bq1 >= Bq2:
    #     location[0, Blue] = routeI if with_probability(Bq1) else routeII
    # else:
    #     location[0, Blue] = routeII if with_probability(Bq2) else routeI
    locRed = routeI if hitOdds[routeI] < hitOdds[routeII] else routeII
    return locBlue, locRed


def transition(state, actionBlue, actionRed, delta):
    ''''''
    locBlue = state[Blue] if choose_to_stay(actionBlue, delta) else other_route(state[Blue])
    locRed = state[Red] if choose_to_stay(actionRed, delta) else other_route(state[Red])
    reward = 1 if locBlue != locRed else 0
    return locBlue, locRed, reward


def play_game(T, thetaBlue, thetaRed, hitOdds, delta=0):
    '''
    Computes a trajectory for the given Blue and Red policies.

    Parameters
    ----------
    T : int
        Number of periods in the game.
    thetaBlue : array
        Blue's policy, (Bp1, Bq1, Bp2, Bq2).
    thetaRed : array
        Red's policy, (Rp1, Rq1, Rp2, Rq2).
    hitOdds : array
        Red's odds of hitting Blue shipment on the routes, (Ro1, Ro2).
    delta : float
        Tolerance for policy (aka theta) probabilities.

    Returns
    -------
    [array, array]
        The trajectory, i.e. the locations (embeds states, actions) and rewards over the course of the game.
    '''
    Bp1, Bq1, Bp2, Bq2 = thetaBlue[0], thetaBlue[1], thetaBlue[2], thetaBlue[3]
    Rp1, Rq1, Rp2, Rq2 = thetaRed[0], thetaRed[1], thetaRed[2], thetaRed[3]
    
    location = np.zeros((T, 2), dtype=int)  # Blue and Red locations in previous period t-1, for each period t
    reward = np.zeros(T, dtype=int)  # Whether Blue shipment gets through in each period t

    # Determine Blue and Red starting locations
    location[0, Blue], location[0, Red] = observe_initial_state(hitOdds)

    # Compute reward for period 0
    if location[0, Blue] == location[0, Red]:
        # Blue starts on route same route as Red, so Red attempts to strike
        r = 0 if red_hits(location[0, Red], hitOdds) else 1
    else:
        # Blue does not start on same route as Red
        reward[0] = 1

    # Handle all other periods
    for t in range(1, T):
        if location[t-1, Blue] == routeI:
            # Blue was on route I
            if location[t-1, Red] == routeI:
                # Red was also on route I, scenario 1a
                location[t, Blue] = routeI if choose_to_stay(Bp1, delta) else routeII
                location[t, Red] = routeI if choose_to_stay(Rp1, delta) else routeII
            else:
                # Red was on route II, scenario 1b
                location[t, Blue] = routeI if choose_to_stay(Bq1, delta) else routeII
                location[t, Red] = routeII if choose_to_stay(Rq2, delta) else routeI
        else:
            # Blue was on route II
            if location[t-1, Red] == routeII:
                # Red was also on route II, scenario 2a
                location[t, Blue] = routeII if choose_to_stay(Bp2, delta) else routeI
                location[t, Red] = routeII if choose_to_stay(Rp2, delta) else routeI
            else:
                # Red was on route I, scenario 2b
                location[t, Blue] = routeII if choose_to_stay(Bq2, delta) else routeI
                location[t, Red] = routeI if choose_to_stay(Rq1, delta) else routeII

        if location[t, Blue] == location[t, Red]:
            # Blue and Red choose same route this period, Red strikes
            reward[t] = 0 if red_hits(location[t, Red], hitOdds) else 1
        else:
            # Blue and Red choose different routes, Red cannot strike
            reward[t] = 1
        
    return location, reward


def gradLogPolicy_Blue(theta, location, t):
    '''
    Computes the gradient of the log of Blue's policy for the action and state 
    at the given time, t, of the trajectory.

    Parameters
    ----------
    theta : array
        Blue's policy, (Bp1, Bq1, Bp2, Bq2).
    location : array
        Locations of Blue and Red for every period of a trajectory.
    t : int
        Current time period.
    '''
    dim = len(theta)

    # First period has no theta dependence
    if t == 0:
        return np.zeros(dim, dtype=float)
    
    Bp1, Bp2 = theta[0], theta[2]
    Bq1, Bq2= theta[1], theta[3]

    # Action is whether Blue switched routes this period
    action = location[t, Blue] != location[t-1, Blue]

    gradLogPolicy = np.zeros(dim, dtype=float)
    if location[t-1, Blue] == routeI:
        # Blue was on route I
        if location[t-1, Red] == routeI:
            # Red was also on route I, scenario 1a
            gradLogPolicy[0] = safe_inverse(Bp1) if not action else safe_inverse(1 - Bp1)
        else:
            # Red was on route II, scenario 1b
            gradLogPolicy[1] = safe_inverse(Bq1) if not action else safe_inverse(1 - Bq1)
    else:
        # Blue was on route II
        if location[t-1, Red] == routeII:
            # Red was also on route II, scenario 2a
            gradLogPolicy[2] = safe_inverse(Bp2) if not action else safe_inverse(1 - Bp2)
        else:
            # Red was on route I, scenario 2b
            gradLogPolicy[3] = safe_inverse(Bq2) if not action else safe_inverse(1 - Bq2)

    return gradLogPolicy


def gradLogPolicy_Red(theta, location, t):
    '''
    Computes the gradient of the log of Red's policy for the action and state 
    at the given time, t, of the trajectory.

    Parameters
    ----------
    theta : array
        Red's policy, (Rp1, Rq1, Rp2, Rq2).
    location : array
        Locations of Blue and Red for every period of a trajectory.
    t : int
        Current time period.
    '''
    dim = len(theta)

    # First period has no theta dependence
    if t == 0:
        return np.zeros(dim, dtype=float)
    
    Rp1, Rp2 = theta[0], theta[2]
    Rq1, Rq2= theta[1], theta[3]

    # Action is whether Red switched routes this period
    action = location[t, Red] != location[t-1, Red]

    gradLogPolicy = np.zeros(dim, dtype=float)
    if location[t-1, Blue] == routeI:
        # Blue was on route I
        if location[t-1, Red] == routeI:
            # Red was also on route I, scenario 1a
            gradLogPolicy[0] = safe_inverse(Rp1) if not action else safe_inverse(1 - Rp1)
        else:
            # Red was on route II, scenario 1b
            gradLogPolicy[3] = safe_inverse(Rq2) if not action else safe_inverse(1 - Rq2)
    else:
        # Blue was on route II
        if location[t-1, Red] == routeII:
            # Red was also on route II, scenario 2a
            gradLogPolicy[2] = safe_inverse(Rp2) if not action else safe_inverse(1 - Rp2)
        else:
            # Red was on route I, scenario 2b
            gradLogPolicy[1] = safe_inverse(Rq1) if not action else safe_inverse(1 - Rq1)

    return gradLogPolicy
