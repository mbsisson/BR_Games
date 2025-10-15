# Wed.Oct.15.151400.2025@jasper

import random
import numpy as np
from game_setup import *

def state_version(log):
    '''State the current version of the BRgames/two_routes/budget_version code.'''
    log.joint('Version: <2025.10.15.161400@jasper>\n\n')

def QLearn_two(log, T, Bsupply_initial, Rbudget_initial, hitOdds, gamma, eps=.05, learn_rate=.05, training_episodes=10000, loud=False):
    log.joint("Running Q-learning for two agents...\n")
    
    stateSpace_Blue = len(location_space) * (Bsupply_initial + 1)  # Player locations and Blue supply remaining
    stateSpace_Red = len(location_space) * (Rbudget_initial + 1)  # Player locations and Red budget remaining
    actionSpace_Blue = n_gridpts * (Bsupply_initial + 1)  # Choice of stay probability and shipment amount
    actionSpace_Red = n_gridpts * 2  # Choice of stay probability and whether to strike

    Qb = np.zeros((stateSpace_Blue + 1, actionSpace_Blue + 1))
    Qr = np.zeros((stateSpace_Red + 1, actionSpace_Red + 1))
    Qb_new = np.zeros_like(Qb)
    Qr_new = np.zeros_like(Qr)

    for epoch in range(1, training_episodes+1):
        locations = random.choice(location_space)
        Bsupply = Bsupply_initial
        Rbudget = Rbudget_initial
        for t in range(T):
            np.copyto(Qb, Qb_new)
            np.copyto(Qr, Qr_new)

            BstateIdx, RstateIdx = getStateIndex(locations, Bsupply, Rbudget)

            if with_probability(eps):
                BactionIdx = random.randint(0, Bsupply * n_gridpts)
            else:
                BactionIdx = np.argmax(Qb[BstateIdx])
            Bship, Bpr = getBactionFromIndex(BactionIdx)

            if with_probability(eps):
                RactionIdx = random.randint(0, 2 * n_gridpts)
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
