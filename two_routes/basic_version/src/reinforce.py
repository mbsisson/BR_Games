###############################################################################
# Policy gradient computation and optimization methods.
#
# author: Blake Sisson
# Fri.Oct.10.104100.2025
###############################################################################

import numpy as np
from two_asymmetric_routesV2 import gradLogPolicy_Blue, gradLogPolicy_Red
from sampler import sampleTrajectories_Blue, sampleTrajectories_Red
#import matplotlib.pyplot as plt  # for plotting


def state_version(log):
    '''State the current version of the BRgames code.'''
    log.joint('Version: <2025.10.22.095100@jasper>\n\n')


# Display functions
arrayToString = lambda theta: ", ".join(f"{x:.4f}" for x in theta)

# Projects theta (if necessary) to ensure it is in the range [0,1]
project_theta = lambda theta: np.clip(theta, 0.05, 0.95)


def policy_gradient(traj_sampler, gradLogProb_func, thetaVariable, T, thetaParameter, hitOdds, delta=0.0, n_samples=1000):
    '''
    Implements policy gradient with reward-to-go and using on-policy value as baseline

    Parameters:
        traj_sampler (callable): trajectory sampler function to use
        thetaVariable (array): theta for the policy being optimized, hence the variable
        T (int): number of periods
        thetaParameter (array): theta for the other agent's policy, hence a parameter
        delta (float): tolerance for theta probabilities
        n_samples (int): number of samples to compute expectation with

    Returns:
        (array): the policy gradient
    '''
    locations, _, rewardsToGo, _ = traj_sampler(thetaVariable, T, thetaParameter, hitOdds, delta, n_samples)
        
    # Compute baseline, on-policy value
    baseline = np.sum(rewardsToGo, axis=0, dtype=float) / n_samples
    
    # Calculate policy gradient
    sum_samples = np.zeros(len(thetaVariable), dtype=float)
    for sample in range(n_samples):
        sum_periods = np.zeros(len(thetaVariable), dtype=float)
        for t in range(0, T):
            gradLogProb = gradLogProb_func(thetaVariable, locations[sample], t)
            rewardsToGo_minus_baseline = rewardsToGo[sample][t] - baseline[t]

            sum_periods += gradLogProb * rewardsToGo_minus_baseline
        sum_samples += sum_periods

    return sum_samples / n_samples


def optimize_policy(log, player, method, params, T, thetaBlue, thetaRed, hitOdds, delta=0.0,
                    max_iter=500, loud=True, verbose=False):
    ''''''
    if loud: log.joint("running policy gradient method...\n")

    if player.lower() == "b":
        # Optimize Blue policy
        theta, k = method(log, sampleTrajectories_Blue, gradLogPolicy_Blue, params, thetaBlue,
                          T, thetaRed, hitOdds, delta, max_iter, verbose)
    
    elif player.lower() == "r":
        # Optimize Red policy
        theta, k = method(log, sampleTrajectories_Red, gradLogPolicy_Red, params, thetaRed, 
                          T, thetaBlue, hitOdds, delta, max_iter, verbose)
    
    else:
        # Invalid player
        log.joint("Invalid player: %s, cannot optimize\n"%(player))
        return None
    
    if loud:
        log.joint("Optimization stopped after %d iterations.\n"%(k))
        log.joint("Optimal policy found: %s\n"%", ".join(f"{x:.4f}" for x in theta))
    return theta


###############################################################################
#                           Optimization Methods                              #
###############################################################################

def pureFirstOrder(log, traj_sampler, gradLogPolicy_func, params, thetaVariable_initial, T, thetaParameter, 
                   hitOdds, delta=0.0, max_iter=500, verbose=False):
    '''
    '''
    n_samples = params["n_samples"]
    tol = params["tol"]
    learning_rate = params["learning_rate"]

    # For plotting
    #theta_history = []
    #theta_history.append(thetaVariable_initial)

    theta = thetaVariable_initial
    gradTheta = policy_gradient(traj_sampler, gradLogPolicy_func, theta, T, thetaParameter, hitOdds, delta, n_samples)

    for k in range(1, max_iter + 1):
        step = learning_rate * gradTheta
        theta = project_theta(theta + step)
        gradTheta = policy_gradient(traj_sampler, gradLogPolicy_func, theta, T, thetaParameter, hitOdds, delta, n_samples)
        #theta_history.append(theta)  # For plotting

        # Converge if projected gradient is smaller than tolerance
        projected_grad = project_theta(theta + gradTheta) - theta
        if np.linalg.norm(projected_grad, np.inf) < tol:
            break

        if verbose: log.joint("Iteration %d,  theta=%s  gradient=%s\n"%(k, arrayToString(theta), arrayToString(gradTheta)))

    # Plot theta over time
    '''
    thetaHistory = np.vstack(theta_history)
    plt.plot(thetaHistory[:, 0], label='p1')
    plt.plot(thetaHistory[:, 1], label='q1')
    plt.plot(thetaHistory[:, 2], label='p2')
    plt.plot(thetaHistory[:, 3], label='q2')
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.title("Policy over time")
    plt.legend()
    plt.show()
    '''

    return theta, k


def adam(log, traj_sampler, gradLogPolicy_func, params, thetaVariable_initial, T, thetaParameter, 
         hitOdds, delta=0.0, max_iter=500, verbose=False):
    '''
    '''
    n_samples = params["n_samples"]
    tol = params["tol"]
    beta1 = params["beta1"]
    beta2 = params["beta2"]
    eta = params["eta"]
    eps = params["eps"]

    theta = thetaVariable_initial
    gradTheta = policy_gradient(traj_sampler, gradLogPolicy_func, theta, T, thetaParameter, hitOdds, delta, n_samples)
    momentum = np.zeros(len(theta))
    avgSquaredGrad = np.zeros(len(theta))
    
    for k in range(1, max_iter + 1):
        # Update exponential moving averages
        momentum = beta1 * momentum + (1 - beta1) * gradTheta
        avgSquaredGrad = beta2 * avgSquaredGrad + (1 - beta2) * np.square(gradTheta)

        # Correct for bias
        unbiased_momentum = momentum / (1 - beta1**k)
        unbiased_avgSquaredGrad = avgSquaredGrad / (1 - beta2**k)

        # Update theta and gradient
        step = eta * np.divide(unbiased_momentum, np.sqrt(unbiased_avgSquaredGrad) + eps)
        theta = project_theta(theta + step)
        gradTheta = policy_gradient(traj_sampler, gradLogPolicy_func, theta, T, thetaParameter, hitOdds, delta, n_samples)

        # Converge if projected gradient is smaller than tolerance
        projected_grad = project_theta(theta + gradTheta) - theta
        if np.linalg.norm(projected_grad, np.inf) < tol:
            break

        if verbose: log.joint("Iteration %d,  theta=%s,  gradient=%s,  prevStepLen=%f\n"%(k, arrayToString(theta), 
                                                                                          arrayToString(gradTheta), 
                                                                                          np.linalg.norm(step)))

    return theta, k