# Tue.Oct.07.114100.2025
from utils import myreadfile

# Reads game settings from config file.
def readConfigFile(log, filename, dictionary):
    log.joint('Reading config file %s\n' %(filename))
    code, lines = myreadfile(log, filename)
    if code: return code

    # Game settings
    T = 0
    supply = 0
    budget = 0
    hitOdds = []
    
    linenum = 0
    # Read lines of data file and save options
    while linenum < len(lines):
        thisline = lines[linenum].split()

        if len(thisline) <= 0:   # skip empty lines 
            linenum += 1
            continue

        if thisline[0][0] == '#':   # skip commented lines
            linenum += 1
            continue

        elif thisline[0] == 'T':
            T = int(thisline[1])

        elif thisline[0] == 'supply':
            supply = int(thisline[1])

        elif thisline[0] == 'budget':
            budget = int(thisline[1])

        elif thisline[0] == 'hitOdds':
            for i in range(1, len(thisline)):
                hitOdds.append(float(thisline[i]))
            
        elif thisline[0] == 'END':
            break

        else:
            log.joint("Error: Illegal input %s\n"%thisline[0])

        linenum += 1

    dictionary["num_routes"] = len(hitOdds)

    log.joint("Game Settings:\n")
    for x in [('T', T),
              ('supply', supply),
              ('budget', budget)]:
        dictionary[x[0]] = x[1]
        log.joint("  {} {}\n".format(x[0], x[1]))
    for x in [('hitOdds', hitOdds)]:
        dictionary[x[0]] = x[1]
        log.joint("  {} {}\n".format(x[0], ", ".join(map(str, x[1]))))
    log.joint("\n")
  
    return code