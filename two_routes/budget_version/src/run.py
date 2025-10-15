###############################################################################
# Run algorithms from here
#
# author: Blake Sisson
# Wed.Oct.10.104100.2025
###############################################################################

import sys
import numpy as np
from datetime import datetime
from logger import danoLogger
from reader import readConfigFile
from game_setup import sampleGivenQvalues
from reinforce import state_version, QLearn_two


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
    supply = gameData["supply"]
    budget = gameData["budget"]
    hitOdds = gameData["hitOdds"]

    # Set parameters
    gamma = 0.9
    eps = .05
    learn_rate = .1
    log.joint("Parameters: for Q Learning\n")
    for x in [("gamma", gamma),
              ("eps", eps),
              ("learn_rate", learn_rate)]:
        log.joint("  {} {}\n".format(x[0], x[1]))
    log.joint("\n")

    Qb, Qr = QLearn_two(log, T, supply, budget, hitOdds, gamma, eps, learn_rate, training_episodes=100000, loud=True)

    sampleGivenQvalues(log, True, Qb, Qr, T, supply, budget, hitOdds)