###############################################################################
# Methods for sampling trajectories of the game.
#
# author: Blake Sisson
# Fri.Oct.10.104100.2025
###############################################################################

import numpy as np
from two_asymmetric_routesV2 import play_game

def sample_trajectory(T, thetaBlue, thetaRed, hitOdds, delta=0.0):
    return play_game(T, thetaBlue, thetaRed, hitOdds, delta)


def sampleTrajectories_Blue(thetaBlue, T, thetaRed, hitOdds, delta=0.0, n_samples=1000):
    '''
    '''
    # Memory for sample Blue trajectories
    locations = np.zeros((n_samples, T, 2), dtype=int)
    rewards = np.zeros((n_samples, T), dtype=int)
    rewardsToGo = np.zeros((n_samples, T), dtype=int)
    total_reward = np.zeros(n_samples, dtype=int)

    # Compute all sample trajectories
    for sample in range(n_samples):
        sample_location, sample_reward = play_game(T, thetaBlue, thetaRed, hitOdds, delta)
        np.copyto(locations[sample], sample_location)
        np.copyto(rewards[sample], sample_reward)
        np.copyto(rewardsToGo[sample], np.cumsum(sample_reward[::-1])[::-1])
        total_reward[sample] = rewardsToGo[sample][0]
        
    return locations, rewards, rewardsToGo, total_reward


def sampleTrajectories_Red(thetaRed, T, thetaBlue, hitOdds, delta=0.0, n_samples=1000):
    '''
    '''
    # Memory for sample Red trajectories
    locations = np.zeros((n_samples, T, 2), dtype=int)
    rewards = np.zeros((n_samples, T), dtype=int)  # Opposite of Blue rewards
    rewardsToGo = np.zeros((n_samples, T), dtype=int)
    total_reward = np.zeros(n_samples, dtype=int)

    # Compute all sample trajectories
    for sample in range(0, n_samples):
        sample_location, sample_reward = play_game(T, thetaBlue, thetaRed, hitOdds, delta)
        np.copyto(locations[sample], sample_location)
        np.copyto(rewards[sample], 1 - sample_reward) # invert rewards: Red rewarded when Blue is not
        np.copyto(rewardsToGo[sample], np.cumsum(sample_reward[::-1])[::-1])
        total_reward[sample] = np.sum(sample_reward)

    return locations, rewards, rewardsToGo, total_reward