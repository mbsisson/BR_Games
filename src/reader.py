# Tue.Oct.07.114100.2025
from utils import myreadfile

# Reads game settings from config file.
def readConfigFile(log, filename, dictionary):
    log.joint('Reading config file %s\n' %(filename))
    code, lines = myreadfile(log, filename)
    if code: return code

    # Game settings
    T = 0
    Bp = []
    Bq = []
    Rp = []
    Rq = []
    Ro = []
    delta = 0
    
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

        elif thisline[0] == 'Bp':
            for i in range(1, len(thisline)):
                Bp.append(float(thisline[i]))

        elif thisline[0] == 'Bq':
            for i in range(1, len(thisline)):
                Bq.append(float(thisline[i]))

        elif thisline[0] == 'Rp':
            for i in range(1, len(thisline)):
                Rp.append(float(thisline[i]))

        elif thisline[0] == 'Rq':
            for i in range(1, len(thisline)):
                Rq.append(float(thisline[i]))

        elif thisline[0] == 'Ro':
            for i in range(1, len(thisline)):
                Ro.append(float(thisline[i]))

        elif thisline[0] == 'delta':
            delta = float(thisline[1])
            
        elif thisline[0] == 'END':
            break

        else:
            log.joint("Error: Illegal input %s\n"%thisline[0])

        linenum += 1

    dictionary["num_routes"] = len(Bp)

    log.joint("Game Settings:\n")
    for x in [('T', T),
              ('delta', delta)]:
        dictionary[x[0]] = x[1]
        log.joint("  {} {}\n".format(x[0], x[1]))
    for x in [('Bp', Bp),
              ('Bq', Bq),
              ('Rp', Rp),
              ('Rq', Rq),
              ('Ro', Ro)]:
        dictionary[x[0]] = x[1]
        log.joint("  {} {}\n".format(x[0], ", ".join(map(str, x[1]))))
    log.joint("\n")
  
    return code