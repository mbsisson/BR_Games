# Not Done
# Wed.Oct.15.134000.2025
import numpy as np
import random
from two_asymmetric_routes import routeI, routeII, with_probability, observe_initial_state, transition

state_space = [[routeI, routeI], [routeI, routeII], [routeII, routeI], [routeII, routeII]]

def qlearn_Blue(learn_rate, gamma, eps, policyRed, hitOdds, max_iter):
    # Initialize
    Q = np.zeros((4, 101))
    Q_old = np.zeros((4, 101))
    state = observe_initial_state(hitOdds)

    for t in range(1, max_iter):
        state_index = state_space.index(state)
        action = np.argmax(Q_old[state_index]) if with_probability(1 - eps) else random.randint(0, 100)
        locBlue, locRed, reward = transition(state, action, policyRed[state_index])
        new_state = [locBlue, locRed]
        new_state_index = state_space.index(new_state)

        Qsa = Q_old[state_index][action]
        maxQsa = np.max(Q_old[new_state_index][action])
        Q[state_index][action] = (1 - learn_rate) * Qsa + learn_rate * (reward + gamma * maxQsa)

