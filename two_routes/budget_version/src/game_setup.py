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
    if Bloc == Rloc and Bship > 0 and Rstrike:
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
        Rbudget_new = Rbudget

    # Decrease supply
    Bsupply_new = Bsupply - Bship

    return [Bloc, Rloc], Bsupply_new, Rbudget_new, Breward, Rreward

def sampleGivenQvalues(log, loud, Qb, Qr, T, Bsupply_initial, Rbudget_initial, hitOdds):
    n_deliveries = 0
    B_totalrewards = 0
    R_totalrewards = 0
    
    locations = random.choice(location_space)
    Bsupply = Bsupply_initial
    Rbudget = Rbudget_initial
    for t in range(T):
        BstateIdx, RstateIdx = getStateIndex(locations, Bsupply, Rbudget)

        # Decide Blue action
        BactionIdx = np.argmax(Qb[BstateIdx])
        Bship, Bpr = getBactionFromIndex(BactionIdx)

        # Decide Red action
        RactionIdx = np.argmax(Qr[RstateIdx])
        Rstrike, Rpr = getRactionFromIndex(RactionIdx)

        # Transition, get rewards and new state
        locations_new, Bsupply_new, Rbudget_new, Breward, Rreward = transition(locations, Bsupply, Rbudget, Bship, Bpr, Rstrike, Rpr, hitOdds)
        
        if loud:
            log.joint("t={}\n".format(t))
            log.joint("  (Bloc, Rloc) = ({}, {})\n".format(locations_new[Blue], locations_new[Red]))
            log.joint("  Blue supply = {}\n".format(Bsupply))
            log.joint("  Red budget = {}\n".format(Rbudget))
            log.joint("  Blue ships {} units on current route w.p. {}\n".format(Bship, Bpr))
            log.joint("  Red strike: {}, stays on current route w.p. {}\n".format(Rstrike, Rpr))
            log.joint("  (Breward, Rreward) = ({}, {})\n".format(Breward, Rreward))

        if Breward > 0: n_deliveries += Bship
        B_totalrewards += Breward
        R_totalrewards += Rreward

        # Update state
        locations = locations_new
        Bsupply = Bsupply_new
        Rbudget = Rbudget_new

    log.joint("Number of deliveries: {}\n".format(n_deliveries))
    log.joint("Total Blue rewards: {}\n".format(B_totalrewards))
    log.joint("Total Red rewards: {}\n".format(R_totalrewards))