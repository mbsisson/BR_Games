# Sun.Oct.18.191200.2025@jasper

import random
import numpy as np
from game_setup_discrete import *

def state_version(log):
    '''State the current version of the BRgames/two_routes/budget_version code.'''
    log.joint('Version: <2025.10.21.183700@jasper>\n\n')

def QLearn_two(log, T, Bsupply_initial, Rbudget_initial, hitOdds, gamma, eps=.05, learn_rate=.05, training_episodes=10000, loud=False):
    log.joint("Running Q-learning for two agents...\n")
    
    n_Bstates =  (Bsupply_initial + 1) * n_locations  # Player locations and Blue supply remaining
    n_Rstates = (Rbudget_initial + 1) * n_locations  # Player locations and Red budget remaining
    n_Bactions = (Bsupply_initial + 1) * n_gridpts  # Choice of stay probability and shipment amount
    n_Ractions = 2 * n_gridpts  # Choice of stay probability and whether to strike

    Qb = np.zeros((n_Bstates, n_Bactions))
    Qr = np.zeros((n_Rstates, n_Ractions))
    Qb_new = np.zeros_like(Qb)
    Qr_new = np.zeros_like(Qr)

    # Use first quarter of training episodes to explore action space
    explorationPhase = True
    eplorationPhaseLength = training_episodes // 4
    eps_actual = eps
    eps = 1

    for epoch in range(1, training_episodes+1):
        if explorationPhase and epoch >= eplorationPhaseLength:
            # Exit exploration phase
            explorationPhase = False
            eps = eps_actual

        locations = random.choice(locationSpace)
        Bsupply = Bsupply_initial
        Rbudget = Rbudget_initial
        for t in range(T):
            np.copyto(Qb, Qb_new)
            np.copyto(Qr, Qr_new)

            BstateIdx, RstateIdx = getStateIndex(locations, Bsupply, Rbudget)

            if with_probability(eps):
                BactionIdx = random.randint(0, n_Bactions - 1)  # Inclusive, so decrement
            else:
                BactionIdx = np.argmax(Qb[BstateIdx])
            Bship, Bpr = getBactionFromIndex(BactionIdx)

            if with_probability(eps):
                RactionIdx = random.randint(0, n_Ractions - 1)  # Inclusive, so decrement
            else:
                RactionIdx = np.argmax(Qr[RstateIdx])
            Rstrike, Rpr = getRactionFromIndex(RactionIdx)

            locations_new, Bsupply_new, Rbudget_new, Breward, Rreward = transition(locations, Bsupply, Rbudget, Bship, Bpr, Rstrike, Rpr, hitOdds)
            BstateIdx_new, RstateIdx_new = getStateIndex(locations_new, Bsupply_new, Rbudget_new)

            Qb_new[BstateIdx][BactionIdx] = (1 - learn_rate)*Qb[BstateIdx][BactionIdx] + learn_rate*(Breward + gamma*np.max(Qb[BstateIdx_new]))
            Qr_new[RstateIdx][RactionIdx] = (1 - learn_rate)*Qr[RstateIdx][RactionIdx] + learn_rate*(Rreward + gamma*np.max(Qr[RstateIdx_new]))

        Qb_diff = np.linalg.norm(Qb_new - Qb, ord=np.inf)
        Qr_diff = np.linalg.norm(Qr_new - Qr, ord=np.inf)

        if loud and epoch % 1000 == 0:
            log.joint("After epoch {},  |Qb_new - Qb| = {}  and  |Qr_new - Qr| = {}\n".format(epoch, Qb_diff, Qr_diff))

    return Qb_new, Qr_new
