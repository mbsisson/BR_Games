import random
import numpy as np

Blue, Red = 0, 1
routeI, routeII = 0, 1
location_space = [[routeI, routeI], [routeI, routeII], [routeII, routeI], [routeII, routeII]]
n_gridpts = 21
probability_space = np.linspace(0, 1, n_gridpts)

# Samples Bernoulli random variable with parameter pr
with_probability = lambda pr: random.random() < pr

# Swtich routes
other_route = lambda route: 1 - route

def getStateIndex(locations, Bsupply, Rbudget):
    Bstate_idx = Bsupply * len(location_space) + location_space.index(locations)
    Rstate_idx = Rbudget * len(location_space) + location_space.index(locations)
    return Bstate_idx, Rstate_idx

def getStateFromIndex(Bstate_idx, Rstate_idx):
    Bsupply = Bstate_idx // len(location_space)
    Rbudget = Rstate_idx // len(location_space)
    locations = location_space[Bstate_idx - (Bsupply * len(location_space))]
    return locations, Bsupply, Rbudget

def getBactionIndex(Bship, Bpr):
    return n_gridpts * Bship + probability_space.index(Bpr)

def getBactionFromIndex(idx):
    Bship = idx // n_gridpts
    Bpr = probability_space[idx - (Bship * n_gridpts)]
    return Bship, Bpr

def getRactionIndex(Rstrike, Rpr):
    return n_gridpts * Rstrike + probability_space.index(Rpr)

def getRactionFromIndex(idx):
    Rstrike = idx // n_gridpts
    Rpr = probability_space[idx - (Rstrike * n_gridpts)]
    return Rstrike, Rpr

def transition(locations, Bsupply, Rbudget,  # State
               Bship, Bpr,  # Blue action
               Rstrike, Rpr,  # Red action
               hitOdds):  # transition parameters
    Bloc = locations[Blue] if with_probability(Bpr) else other_route(locations[Blue])
    Rloc = locations[Red] if with_probability(Rpr) else other_route(locations[Red])
    if Bloc == Rloc and Rstrike:
        # R strikes
        if with_probability(hitOdds[Rloc]):
            # Red hits Blue shipment
            Breward = -0.5 * Bship
            Rreward = Bship
        else:
            # Red misses Blue shipment
            Breward = Bship
            Rreward = -0.1

        # Decrement R budget
        Rbudget_new = Rbudget - Rstrike

    else:
        # Red cannot or did not strike
        Breward = Bship
        Rreward = 0

    # Decrease supply
    Bsupply_new = Bsupply - Bship

    return [Bloc, Rloc], Bsupply_new, Rbudget_new, Breward, Rreward

def NashQ(T, Bsupply_initial, Rbudget_initial, hitOdds, gamma, eps=.05, training_episodes=1000):
    stateSpace_Blue = len(location_space) * (Bsupply_initial + 1)  # Player locations and Blue supply remaining
    stateSpace_Red = len(location_space) * (Rbudget_initial + 1)  # Player locations and Red budget remaining
    actionSpace_Blue = n_gridpts * (Bsupply_initial + 1)  # Choice of stay probability and shipment amount
    actionSpace_Red = n_gridpts * 2  # Choice of stay probability and whether to strike

    Qb = np.zeros((stateSpace_Blue, actionSpace_Blue, actionSpace_Red))
    Qr = np.zeros((stateSpace_Red, actionSpace_Blue, actionSpace_Red))
    Qb_new = np.zeros_like(Qb)
    Qr_new = np.zeros_like(Qr)

    for epoch in training_episodes:
        locations = random.choice(location_space)
        Bsupply = Bsupply_initial
        Rbudget = Rbudget_initial
        for t in range(T):
            stateIdx = getStateIndex(locations, Bsupply, Rbudget)

            if with_probability(eps):
                BactionIdx = random.randint((Bsupply * n_gridpts))
            else:
                BactionIdx = np.argmax(Qb[stateIdx])
            Bship, Bpr = getBactionFromIndex(BactionIdx)

            if with_probability(eps):
                RactionIdx = random.randint((Rbudget * n_gridpts))
            else:
                RactionIdx = np.argmax(Qr[stateIdx])
            Rstrike, Rpr = getRactionFromIndex(RactionIdx)

            locations_new, Bsupply_new, Rbudget_new, Breward, Rreward = transition(locations, Bsupply, Rbudget,
                                                                                   Bship, Bpr,
                                                                                   Rstrike, Rpr,
                                                                                   hitOdds)

