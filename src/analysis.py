###############################################################################
# Methods for in depth analysis of policies.
#
# author: Blake Sisson
# Wed.Oct.08.153900.2025
###############################################################################

import numpy as np
#import matplotlib.pyplot as plt
#from scipy.stats import truncnorm
from optimizer import sampleTrajectories_Blue


def evaluate_policy(log, T, thetaBlue, thetaRed, hitOdds, delta=0.0, n_samples=500, loud=True):
    '''
    Computes several statistics for the number of deliveries (Blue reward) of given policy using sample average.
    
    Parameters
    ----------
    log : danologger
        Log to use.
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
    n_samples : int
        Number of samples to use for averaging.
    loud : bool
        Whether to report data to log.

    Returns
    -------
    [float, float, float, float]
        The average, variance, minimum, and maximum statistics for the number of deliveries
    '''
    _, _, _, total_rewards = sampleTrajectories_Blue(thetaBlue, T, thetaRed, hitOdds, delta, n_samples)

    avg_delivs = np.average(total_rewards)
    var_delivs = np.var(total_rewards)
    min_delivs = np.min(total_rewards)
    max_delivs = np.max(total_rewards)

    if loud:
        log.joint("Game statistics\n")
        log.joint("average reward:  %f\n"%(avg_delivs))
        log.joint("variance reward: %f\n"%(var_delivs))
        log.joint("min reward:      %f\n"%(min_delivs))
        log.joint("max reward:      %f\n"%(max_delivs))
        log.joint("\n")

    return avg_delivs, var_delivs, min_delivs, max_delivs


# def compute_statistics(T, p, q, pR, qR, loud=True):
#     '''
#     Computes the mean and variance of the score for a game with the given length and parameters.
#     '''
#     # P(t,s,c): Probability of ending up with score s and condition c, either (a)=0 or (b)=1, after period 0 <= t <= T-1.
#     P = np.zeros((T, T+1, 2))

#     # Handle initial period separately.
#     P[0][0][0] = 0.5 # score 0, condition (a)
#     P[0][1][1] = 0.5 # score 1, condition (b)

#     # prob_ij: Probability of transitioning to condition (i) to condition (j)
#     prob_aa = p*pR + (1-p)*(1-pR)
#     prob_ab = p*(1-pR) + (1-p)*pR
#     prob_ba = q*(1-qR) + (1-q)*qR
#     prob_bb = q*qR + (1-q)*(1-qR)
    
#     for t in range(1, T):
#         # Handle zero score case separately, haven't entered condition (b) yet.
#         P[t][0][0] = prob_aa * P[t-1][0][0]
        
#         for s in range(1, t+2):
#             # Condition (a), cannot have scored in period t
#             P[t][s][0] = prob_aa * P[t-1][s][0] + prob_ba * P[t-1][s][1]

#             # Condition (b), scored in period t
#             P[t][s][1] = prob_ab * P[t-1][s-1][0] + prob_bb * P[t-1][s-1][1]

#     mean = 0
#     for s in range(0, T+1):
#         mean += (P[T-1][s][0] + P[T-1][s][1]) * s

#     variance = 0
#     for s in range(0, T+1):
#         variance += (P[T-1][s][0] + P[T-1][s][1]) * (s - mean)**2
#     sigma = np.sqrt(variance)

#     #samples_norm = np.random.normal(mean, sigma, size=100000)
#     #var_norm = np.percentile(samples_norm, 100 * alpha)
#     #cvar_norm = samples_norm[samples_norm <= var_norm].mean()    
    
#     alpha = 0.05
#     lower, upper = 0, 100
#     trunc_dist = truncnorm((lower - mean) / sigma, (upper - mean) / sigma, loc=mean, scale=sigma)
#     quantile = trunc_dist.ppf(alpha)
#     x_vals = np.linspace(lower, quantile, 1000)
#     pdf_vals = trunc_dist.pdf(x_vals)
#     cvar = np.trapezoid(x_vals * pdf_vals, x_vals) / trunc_dist.cdf(quantile)
        
#     if loud: print("    mean=%.3f  std=%.3f  VaR=%.3f  CVaR=%.3f"%(mean, sigma, quantile, cvar))
#     return mean, variance


# def plot_value_function(state, action, reward):
#     T = len(state)
#     rewards_to_go = np.zeros(T, dtype=int)
#     np.copyto(rewards_to_go, np.cumsum(reward[::-1])[::-1])
#     plt.plot(range(T), rewards_to_go, label='Value Function')
#     plt.xlabel('Time Step t')
#     plt.ylabel('V(s_t)')
#     plt.title('Value Function Along Trajectory')
#     plt.legend()
#     plt.show()

    
# def policy_string(p, q, pR, qR):
#     return "(p=%.2f,  q=%.2f,  pR=%.2f  qR=%.2f)"%(p, q, pR, qR)
