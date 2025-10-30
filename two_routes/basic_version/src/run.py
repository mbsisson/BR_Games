###############################################################################
# Run games from here.
# Game description found in optimizer.py
#
# author: Blake Sisson
# Fri.Oct.10.104100.2025
###############################################################################

import sys
import numpy as np
from datetime import datetime
from logger import danoLogger
from reader import readConfigFile
from analysis import evaluate_policy, compute_VaR
from reinforce import state_version, optimize_policy, arrayToString, pureFirstOrder, adam

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("arguments: configFile [logFileName]")

    configfile = sys.argv[1]

    # Create log file
    logfilename = "log.txt"
    if len(sys.argv) > 2:
        logfilename = sys.argv[2]
    datetime_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    logfile = datetime_string + "_" + logfilename

    # Create log
    log = danoLogger(logfile)
    state_version(log)

    gameData = {}
    if readConfigFile(log, configfile, gameData):
        log.joint("problem reading configFile %s"%(configfile))
        sys.exit()

    T = gameData["T"]
    Bp = gameData["Bp"]
    Bq = gameData["Bq"]
    Rp = gameData["Rp"]
    Rq = gameData["Rq"]
    Ro = gameData["Ro"]
    delta = gameData["delta"]

    thetaBlue = []
    thetaRed = []
    hitOdds = []
    for i in range(gameData["num_routes"]):
        thetaBlue.append(Bp[i])
        thetaBlue.append(Bq[i])
        thetaRed.append(Rp[i])
        thetaRed.append(Rq[i])
        hitOdds.append(Ro[i])
    thetaBlue = np.array(thetaBlue)
    thetaRed = np.array(thetaRed)

    if (thetaBlue > 1).any() or (thetaBlue < 0).any() or (thetaRed > 1).any() or (thetaRed < 0).any():
        log.joint("Invalid probabilities\n")
        sys.exit()

    # The optimization method to use
    optMethod1 = pureFirstOrder
    optMethod2 = adam
    maxiters = 1000

    # Parameters for the optimization methods
    params = {}
    params["n_samples"] = 250
    params["learning_rate"] = 0.01
    params["beta1"] = 0.9
    params["beta2"] = 0.999
    params["eta"] = 0.01
    params["eps"] = 1e-10
    params["tol"] = 1e-3

    # Record optimization settings
    log.joint("Optimization settings:\n")
    #log.joint("  method %s\n"%optMethod.__name__)
    for x in [("n_samples", params["n_samples"]),
              ("tol", params["tol"]),
              ("learning_rate", params["learning_rate"]),
              ("beta1 (adam)", params["beta1"]),
              ("beta2 (adam)", params["beta2"]),
              ("eta (adam)", params["eta"]),
              ("eps (adam)", params["eps"])]:
        log.joint("  {} {}\n".format(x[0], x[1]))
    log.joint("\n")

    # # Sample one trajectory
    # locations, rewards = sample_trajectory(T, thetaBlue, thetaRed, hitOdds, delta=delta)
    # log.joint("One sample trajectory\n")
    # log.joint("  locations: %s\n"%(", ".join(map(str, locations))))
    # log.joint("  rewards: %s\n"%(", ".join(map(str, rewards))))

    # Evaluate current Blue policy given Red policy
    evaluate_policy(log, T, thetaBlue, thetaRed, hitOdds, delta, loud=True)
    log.joint("\n")

    # Number of trials to run (multi-starting)
    n_trials = 1
    
    # Confidence level for VaR computation
    alpha = 0.05

    # # Optimize Blue policy on initial Red policy
    # log.joint("Optimizing Blue policy using %s\n"%optMethod.__name__)
    # for trial in range(1, n_trials+1):
    #     log.joint("Trial %d\n"%trial)
    #     log.joint("initial Blue policy: %s\n"%", ".join(map(str, thetaBlue)))
    #     log.joint("fixed Red policy: %s\n"%", ".join(map(str, thetaRed)))
    #     thetaBlue_star = optimize_policy(log, "B", optMethod, params, T, thetaBlue, thetaRed, hitOdds, delta=delta,
    #                                     max_iter=maxiters, loud=True, verbose=False)
    #     evaluate_policy(log, T, thetaBlue_star, thetaRed, hitOdds, delta)
    #     log.joint("\n")

    # log.joint("Optimizing Blue policy using %s\n"%optMethod2.__name__)
    # for trial in range(1, n_trials+1):
    #     log.joint("Trial %d\n"%trial)
    #     log.joint("initial Blue policy: %s\n"%", ".join(map(str, thetaBlue)))
    #     log.joint("fixed Red policy: %s\n"%", ".join(map(str, thetaRed)))
    #     thetaBlue_star = optimize_policy(log, "B", optMethod2, params, T, thetaBlue, thetaRed, hitOdds, delta=delta,
    #                                     max_iter=maxiters, loud=True, verbose=False)
    #     evaluate_policy(log, T, thetaBlue_star, thetaRed, hitOdds, delta)
    #     log.joint("\n")

    # # Optimize Red policy on initial Blue policy
    # log.joint("Optimizing Red policy using %s\n"%optMethod.__name__)
    # for trial in range(1, n_trials+1):
    #     log.joint("initial Red policy: %s\n"%", ".join(map(str, thetaRed)))
    #     log.joint("fixed Blue policy: %s\n"%", ".join(map(str, thetaBlue)))
    #     thetaRed_star = optimize_policy(log, "R", optMethod, params, T, thetaBlue, thetaRed, hitOdds, delta=delta, 
    #                                     max_iter=maxiters, loud=True, verbose=False)
    #     evaluate_policy(log, T, thetaBlue, thetaRed_star, hitOdds, delta)
    #     log.joint("\n")

    # log.joint("Optimizing Red policy using %s\n"%optMethod2.__name__)
    # for trial in range(1, n_trials+1):
    #     log.joint("initial Red policy: %s\n"%", ".join(map(str, thetaRed)))
    #     log.joint("fixed Blue policy: %s\n"%", ".join(map(str, thetaBlue)))
    #     thetaRed_star = optimize_policy(log, "R", optMethod2, params, T, thetaBlue, thetaRed, hitOdds, delta=delta, 
    #                                     max_iter=maxiters, loud=True, verbose=False)
    #     evaluate_policy(log, T, thetaBlue, thetaRed_star, hitOdds, delta)
    #     log.joint("\n")

    #Optimize Blue and Red simultaneously
    step_count = 5  # Number of first-order steps each player can take during their training turn
    epoch_count = 1000
    thetaBlue_hat, thetaRed_hat = np.zeros(len(thetaBlue)), np.zeros(len(thetaBlue))

    log.joint("Optimizing Blue and Red policies in tanget using %s\n"%optMethod2.__name__)
    log.joint(f"each gets {step_count} steps per round, {epoch_count} rounds total\n")
    for trial in range(1, n_trials+1):
        log.joint("trail %d\n"%trial)
        thetaBlue_random = thetaBlue #np.random.rand(len(thetaBlue))
        thetaRed_random = thetaRed #np.random.rand(len(thetaRed))
        log.joint("Initial policies:  ThetaBlue=%s  ThetaRed%s\n"%(arrayToString(thetaBlue_random), arrayToString(thetaRed_random)))
        np.copyto(thetaBlue_hat, thetaBlue_random)
        np.copyto(thetaRed_hat, thetaRed_random)
        for epoch in range(1, epoch_count+1):
            thetaBlue_hat = optimize_policy(log, "B", optMethod2, params, T, thetaBlue, thetaRed, hitOdds, delta=delta,
                                            max_iter=step_count, loud=False, verbose=False)
            thetaRed_hat = optimize_policy(log, "R", optMethod2, params, T, thetaBlue, thetaRed, hitOdds, delta=delta,
                                            max_iter=step_count, loud=False, verbose=False)

            if epoch % 250 == 0:
                log.joint("After round %d:  ThetaBlue=(%s)  ThetaRed=(%s)\n"%(epoch, arrayToString(thetaBlue_hat), arrayToString(thetaRed_hat)))
                avg_delivs, var_delivs, min_delivs, max_delivs = evaluate_policy(log, T, thetaBlue_hat, thetaRed_hat, hitOdds, delta)
                VaR, CVaR = compute_VaR(alpha, avg_delivs, var_delivs, min_delivs, max_delivs)
                log.joint("Game statistics\n")
                log.joint("average reward:  %f\n"%(avg_delivs))
                log.joint("variance reward: %f\n"%(var_delivs))
                log.joint("min reward:      %f\n"%(min_delivs))
                log.joint("max reward:      %f\n"%(max_delivs))
                log.joint("VaR:             %f\n"%(VaR))
                log.joint("CVaR:            %f\n"%(CVaR))
                # log.joint("VaR_trunc:       %f\n"%(VaR_trunc))
                # log.joint("CVaR_trunc:      %f\n"%(CVaR_trunc))
                log.joint("\n")

    # log.joint("Optimizing Blue and Red policies in tanget using %s\n"%optMethod2.__name__)
    # log.joint(f"each gets {step_count} steps per round, {epoch_count} rounds total\n")
    # log.joint("Initial policies:  ThetaBlue=%s  ThetaRed%s\n"%(arrayToString(thetaBlue_hat), arrayToString(thetaRed_hat)))
    # for trial in range(1, n_trials+1):
    #     log.joint("trail %d\n"%trial)
    #     np.copyto(thetaBlue_hat, thetaBlue)
    #     np.copyto(thetaRed_hat, thetaRed)
    #     for epoch in range(1, epoch_count+1):
    #         thetaBlue_hat = optimize_policy(log, "B", optMethod2, params, T, thetaBlue, thetaRed, hitOdds, delta=delta,
    #                                         max_iter=step_count, loud=False, verbose=False)
    #         thetaRed_hat = optimize_policy(log, "R", optMethod2, params, T, thetaBlue, thetaRed, hitOdds, delta=delta,
    #                                         max_iter=step_count, loud=False, verbose=False)

    #         if epoch % 100 == 0:
    #             log.joint("After round %d:  ThetaBlue=(%s)  ThetaRed=(%s)\n"%(epoch, arrayToString(thetaBlue_hat), arrayToString(thetaRed_hat)))
    #             evaluate_policy(log, T, thetaBlue_hat, thetaRed_hat, hitOdds, delta)
    #     log.joint("\n")

    # Counterfactual Analysis
    n_tests = 10
    log.joint("Blue policy (ThetaBlue) found after training: %s\n"%(arrayToString(thetaBlue_hat)))
    log.joint("Running Counterfactual analysis on random Red policies\n")
    for trial in range(n_tests):
        log.joint("Trial %d\n"%(trial))
        thetaRed_random = np.random.rand(len(thetaRed))
        log.joint("Red Policy (ThetaRed): %s\n"%(arrayToString(thetaRed_random)))
        evaluate_policy(log, T, thetaBlue_hat, thetaRed_random, hitOdds, delta, loud=True)