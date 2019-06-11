import json
import collections
import numpy as np

N = 5

def nextPlayer(p, n=N):
    
    if n == 5:
        if p == "a":
            return "e"
        if p == "b":
            return "a"
        if p == "c":
            return "b"
        if p == "d":
            return "c"
        if p == "e":
            return "d"

    if n == 6:
        if p == "a":
            return "f"
        if p == "b":
            return "a"
        if p == "c":
            return "b"
        if p == "d":
            return "c"
        if p == "e":
            return "d"
        if p == "f":
            return "e"
    
    if n == 7:
        if p == "a":
            return "g"
        if p == "b":
            return "a"
        if p == "c":
            return "b"
        if p == "d":
            return "c"
        if p == "e":
            return "d"
        if p == "f":
            return "e"
        if p == "g":
            return "f"

def playerIdx(tar, m11):
    
    idx = 0
    p = m11
    while p != tar:
        p = nextPlayer(p)
        idx += 1
    return idx

    if p == "a":
        return 4
    if p == "b":
        return 3
    if p == "c":
        return 2
    if p == "d":
        return 1
    if p == "e":
        return 0
    
def parseData():
    # read file
    with open('Data/gameRecordsDataAnon.json', 'r') as myfile:
        data=myfile.read()

    # parse file
    obj = json.loads(data)

    print(len(obj))
    np.random.shuffle(obj)

    res_wins = collections.defaultdict(int)
    game_count = collections.defaultdict(int)
    game_missions = []
    features = []
    resIdxs = []
    merlinIdxs = []
    percivalIdxs = []
    vtIdxs = []
    mostCorrects = []
    human_shots = []

    cnt = 0
    never_clean = 0
    for game in obj:
        winner = game["winningTeam"]
        spies = game["spyTeam"]
        res = game["resistanceTeam"]
        num_p = len(spies) + len(res)
        if num_p != N:
            continue
        roles = game["roles"]
        if len(roles) != 4:
            continue
        if "Merlin" not in roles or "Percival" not in roles or "Assassin" not in roles or "Morgana" not in roles:
            continue

        numSucc = 0
        for s in game["missionHistory"]:
            if s == "succeeded":
                numSucc += 1
        if numSucc != 3:
            continue

        vh = game["voteHistory"]


        m11 = ""
        for player in vh:
            row = vh[player]
            if "VHleader" in row[0][0]:
                m11 = player
                break

        res_idx = []
        for r in res:
            res_idx.append(playerIdx(r, m11))

        merlin = "e"
        for i in range(N):
            if game["playerRoles"][merlin]["role"] == "Merlin":
                break
            merlin = nextPlayer(merlin)
        merlin_idx = playerIdx(merlin, m11)

        percival = "e"
        for i in range(N):
            if game["playerRoles"][percival]["role"] == "Percival":
                break
            percival = nextPlayer(percival)
        percival_idx = playerIdx(percival, m11)

        vt = "e"
        for i in range(N):
            if game["playerRoles"][vt]["role"] == "Resistance":
                break
            vt = nextPlayer(vt)
        vt_idx = playerIdx(vt, m11)

        human_shot = 0
        if game['winningTeam'] != 'Resistance':
            human_shot = 1

        mh = game['missionHistory']


        
        missions = getMatrixFeatures(vh, spies)
        correctVotes = getCorrectVote(missions)
        correctPicks = getCorrectPicks(missions)
        correct3Approves = correct3Approve(missions)
        correct3Rejects = correct3Reject(missions)
        correct3Picks= correct3Pick(missions)
        firstCleanProposals = firstCleanProposal(missions)
        firstDirtyProposals = firstDirtyProposal(missions)
        numDirty = numberDirtyMissions(missions)
        numFailed = numberFailedMissions(missions, mh)
        wrongVotes = getWrongPicks(missions)
        dirtyM1 = dirtyM1s(missions)

        feature = np.zeros( (N, 7) )
        feature[:, 0] = correctVotes
        feature[:, 1] = wrongVotes
        feature[:, 2] = correct3Approves
        #feature[:, 2] = correct3Rejects
        feature[:, 3] = firstCleanProposals
        feature[:, 4] = numDirty
        feature[:, 5] = correctPicks
        #feature[:, 6] = firstDirtyProposals
        feature[:, 6] = dirtyM1
        #feature[:, 4] = numFailed


        if len(features) <= 5:
            print(feature)
            print(merlin_idx)

        #if len(features) == 10:
        #    break
        if sum(firstCleanProposals) == 0:
            never_clean += 1
            #print(vh)
            #print(missions)
            #break

        
        game_missions.append(missions)
        features.append(feature)
        resIdxs.append(res_idx)
        merlinIdxs.append(merlin_idx)
        percivalIdxs.append(percival_idx)
        vtIdxs.append(vt_idx)
        mostCorrects.append(correctVotes)
        human_shots.append(human_shot)
        #break

    print(len(game_missions))
    print("games with no clean proposals by res: " + str(never_clean))
    #return game_missions, resIdxs, merlinIdxs, mostCorrects, percivalIdxs, vtIdxs
    return features, resIdxs, merlinIdxs, mostCorrects, percivalIdxs, vtIdxs, human_shots


def numberFailedMissions(missions, mh, verbose=0):
    
    correct = np.zeros(N)
    for m in range(5):
        for mm in range(N):
            isApproved = False
            numApps = 0
            onMission = []
            isDirty = False
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == 1:
                    onMission.append(p)
                if  missions[m][mm][p][2] == 1:
                    numApps += 1 
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True

            if numApps >= 3:
                isApproved = True

            if isApproved and isDirty and mh[m] == 'failed':
                if verbose == 1:
                    print(onMission)
                correct[onMission] += 1
            
    return correct

def numberDirtyMissions(missions, verbose=0):
    
    correct = np.zeros(N)
    for m in range(5):
        for mm in range(N):
            isApproved = False
            numApps = 0
            onMission = []
            isDirty = False
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == 1:
                    onMission.append(p)
                if  missions[m][mm][p][2] == 1:
                    numApps += 1 
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True

            if numApps >= 3:
                isApproved = True

            if isApproved and isDirty:
                if verbose == 1:
                    print(onMission)
                correct[onMission] += 1
            
    return correct


def dirtyM1s(missions):
    
    correct = np.zeros(N)
    for m in range(1):
        for mm in range(N):
            isDirty = False
            numSpies = 0
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True
                    numSpies += 1

            for p in range(N):
                approved = missions[m][mm][p][2] == 1
                picked = missions[m][mm][p][1] == 1
                leader = missions[m][mm][p][0] == 1
                if not leader:
                    continue
                value = 1
                if missions[m][mm][p][3] == 1:
                    if not isDirty and not approved:
                        #correct[p] += value
                        correct[p] += value
                    if isDirty and approved:
                        correct[p] += value
                    if isDirty and not approved:
                        if not picked and numSpies > 1:
                            correct[p] += value
    return correct


def firstDirtyProposal(missions):
    
    correct = np.zeros(N)
    for m in range(5):
        for mm in range(N):
            isDirty = False
            numSpies = 0
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True
                    numSpies += 1

            for p in range(N):
                approved = missions[m][mm][p][2] == 1
                picked = missions[m][mm][p][1] == 1
                leader = missions[m][mm][p][0] == 1
                if not leader:
                    continue
                value = 1
                if missions[m][mm][p][3] == 1:
                    if not isDirty and not approved:
                        #correct[p] += value
                        correct[p] += value
                        return correct
                    if isDirty and approved:
                        correct[p] += value
                        return correct
                    if isDirty and not approved:
                        if not picked and numSpies > 1:
                            correct[p] += value
                            return correct
    return correct

def firstCleanProposal(missions):
    
    correct = np.zeros(N)
    for m in [1,3,4]:
        for mm in range(N):
            isDirty = False
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True
                    break
                    
            for p in range(N):
                approved = missions[m][mm][p][2] == 1
                picked = missions[m][mm][p][1] == 1
                leader = missions[m][mm][p][0] == 1
                value = 1
                if missions[m][mm][p][3] == 1:
                    if not isDirty:
                        #correct[p] += value
                        if leader and picked:
                            correct[p] += value
                            return correct
    return correct

def correct3Pick(missions):

    correct = np.zeros(N)
    for m in [1,3,4]:
        for mm in range(N):
            isDirty = False
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True
                    break
                    
            for p in range(N):
                approved = missions[m][mm][p][2] == 1
                picked = missions[m][mm][p][1] == 1
                leader = missions[m][mm][p][0] == 1
                value = 1
                if missions[m][mm][p][3] == 1:
                    #if isDirty and not approved:
                    #    if leader:
                    #        correct[p] += value
                    if approved and not isDirty:
                        #correct[p] += value
                        if leader and picked:
                            correct[p] += value
    return correct

def correct3Approve(missions):

    correct = np.zeros(N)
    for m in [1,3,4]:
        for mm in range(N):
            isDirty = False
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True
                    break
                    
            for p in range(N):
                approved = missions[m][mm][p][2] == 1
                picked = missions[m][mm][p][1] == 1
                value = 1
                if missions[m][mm][p][3] == 1:
                    if approved and not isDirty:
                        #correct[p] += value
                        if picked:
                            correct[p] += value
    return correct

def correct3Reject(missions):

    correct = np.zeros(N)
    for m in [1,3,4]:
        for mm in range(N):
            isDirty = False
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True
                    break
                    
            for p in range(N):
                approved = missions[m][mm][p][2] == 1
                picked = missions[m][mm][p][1] == 1
                value = 1
                if missions[m][mm][p][3] == 1:
                    if isDirty and not approved:
                        #correct[p] += value
                        if picked:
                            correct[p] += value
    return correct


def getWrongPicks(missions):

    correct = np.zeros(N)
    for m in range(N):
        for mm in range(N):
            isDirty = False
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True
                    break
            value = 1
            #if m in [1,3,4]:
            #    value = 3
            for p in range(N):
                approved = missions[m][mm][p][2] == 1
                picked = missions[m][mm][p][1] == 1
                leader = missions[m][mm][p][0] == 1
                if missions[m][mm][p][3] == 1:
                    if isDirty and approved:
                        #correct[p] += value
                        if leader:
                            correct[p] += value
                    if not approved and not isDirty:
                        #correct[p] += value
                        if leader:
                            correct[p] += value
    return correct

def getCorrectPicks(missions):

    correct = np.zeros(N)
    for m in range(N):
        for mm in range(N):
            isDirty = False
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True
                    break
            value = 1
            #if m in [1,3,4]:
            #    value = 3
            for p in range(N):
                approved = missions[m][mm][p][2] == 1
                picked = missions[m][mm][p][1] == 1
                leader = missions[m][mm][p][0] == 1
                if missions[m][mm][p][3] == 1:
                    if isDirty and not approved:
                        #correct[p] += value
                        if leader:
                            correct[p] += value
                    if approved and not isDirty:
                        #correct[p] += value
                        if leader:
                            correct[p] += value
    return correct

def getCorrectVote(missions):

    correct = np.zeros(N)
    for m in range(N):
        for mm in range(N):
            isDirty = False
            for p in range(N):
                if missions[m][mm][p][1] == 1 and missions[m][mm][p][3] == -1:
                    isDirty = True
                    break
            value = 1
            #if m in [1,3,4]:
            #    value = 3
            for p in range(N):
                approved = missions[m][mm][p][2] == 1
                picked = missions[m][mm][p][1] == 1
                if missions[m][mm][p][3] == 1:
                    if isDirty and not approved:
                        correct[p] += value
                        #if picked:
                        #    correct[p] += value
                    if approved and not isDirty:
                        correct[p] += value
                        if not picked:
                            correct[p] += value
    return correct
        

# 5x5 for m1.1-m5.5
# 4 vectors (of length 5) representing:
#   VHleader (1)
#   picked? (1)
#   approved? (1)
#   res? (1)
#   correct vote? (1)
def getMatrixFeatures(vh, spies):
    missions = np.full( (N,N,N,5), 0)
        
    m11 = ""
    for player in vh:
        row = vh[player]
        if "VHleader" in row[0][0]:
            m11 = player
            break

    #p = "e"
    p = m11
    m = 0
    mm = 0
    while m < N:

        p_idx = playerIdx(p, m11)
        d = vh[p]
        if len(d) <= m:
            break
        d = d[m]
        if len(d) <= mm:

            for mm in range(N-1): 
                isDirty = False
                for pp in range(N):
                    if missions[m][mm][pp][1] == 1 and missions[m][mm][pp][3] == -1:
                        isDirty = True
                        break
                if missions[m][mm][pp][0] == 0:
                    continue
                for pp in range(N):
                    if isDirty and missions[m][mm][pp][2] == -1:
                        missions[m][mm][pp][4] = 1
                    elif not isDirty and missions[m][mm][pp][2] == 1:
                        missions[m][mm][pp][4] = 1
                    else:
                        missions[m][mm][pp][4] = -1

            mm = 0

            p = nextPlayer(p)
            #if p == "e":
            if p == m11:
                m += 1
            continue
        d = d[mm]

        #print(m)
        #print(mm)
        #print(p_idx)
        #print("------------")
        if p in spies:
            missions[m][mm][p_idx][3] = -1
            #print(p)
            #print(p_idx)
        else:                     
            missions[m][mm][p_idx][3] = 1
                                  
        if "VHleader" in d:       
            missions[m][mm][p_idx][0] = 1
        else:                     
            missions[m][mm][p_idx][0] = -1
                                  
        if "VHpicked" in d:       
            missions[m][mm][p_idx][1] = 1
        else:                     
            missions[m][mm][p_idx][1] = -1
                                  
        if "VHapprove" in d:
            missions[m][mm][p_idx][2] = 1
        elif "VHreject" in d:
            missions[m][mm][p_idx][2] = -1

        mm += 1
    return missions


