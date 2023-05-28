

"""
https://www.insider.com/mlb-used-two-balls-again-this-year-and-evidence-points-to-a-third-2022-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pitcher_model(order):
    """based on pitcher order for a given team, how many innings do we expect?"""
    prediction = -25*order+215.0
    uncertainty = 25.0
    return prediction,uncertainty

qualitativefile = 'data/qualitative-order-2023-02-06.csv'
EST = pd.read_csv(qualitativefile)

# who are the teams?
teams = np.unique(EST['Team'].values)
print(teams)

# bring in last year's stats
year = '2023'
DF23 = pd.read_csv('data/yearlystats-'+year+'.csv')
year = '2022'
DF22 = pd.read_csv('data/yearlystats-'+year+'.csv')
year = '2021'
DF21 = pd.read_csv('data/yearlystats-'+year+'.csv')
year = '2020'
DF20 = pd.read_csv('data/yearlystats-'+year+'.csv')
year = '2019'
DF19 = pd.read_csv('data/yearlystats-'+year+'.csv')
#print(np.array(DF['Name'].values))

keys = ['Pitcher1','Pitcher2','Pitcher3','Pitcher4','Pitcher5','Pitcher6']

f = open('predictions/consolidated-estimate-2023.csv','w')
print('Team,Player,Model23,Actual22,Actual21,Actual20,Actual19,consolidated,uncertainty',file=f)

playermatrix = []
playername = []

for team in teams:
    for key in keys:
        ord = float(key.split('Pitcher')[1])
        print(ord)
        pred,unc = pitcher_model(ord)
        plr = EST[EST['Team']==team][key].values
        #print(plr)
        w = np.where(np.array(DF22['Name'].values)==plr)

        try:
            ip22 = DF22['IP'][np.array(DF22['Name'].values)==plr].values[0]
        except:
            print('No 2022 for {}'.format(plr))
            ip22 = 0.0

        try:
            ip21 = DF21['IP'][np.array(DF21['Name'].values)==plr].values[0]
        except:
            print('No 2021 for {}'.format(plr))
            ip21 = 0.0

        try:
            ip20 = np.round((162./60.)*DF20['IP'][np.array(DF20['Name'].values)==plr].values[0],1)
        except:
            print('No 2020 for {}'.format(plr))
            ip20 = 0.0

        try:
            ip19 = DF19['IP'][np.array(DF19['Name'].values)==plr].values[0]
        except:
            print('No 2019 for {}'.format(plr))
            ip19 = 0.0

        try:
            ip23 = DF23['IP'][np.array(DF23['Name'].values)==plr].values[0]
        except:
            print('No 2023 for {}'.format(plr))
            ip23 = 0.0


        if ip23>1.0:
            playermatrix.append([pred,ip22,ip21,ip20,ip19,ip23])
            playername.append(plr[0])

        consolidated = 0.25*pred + 0.5*ip22 + 0.15*ip21 + 0.05*ip20 + 0.05*ip19
        uncertainty  = np.std([pred,ip22,ip21])

        print('{},{},{},{},{},{},{},{},{}'.format(team,plr[0],pred,ip22,ip21,ip20,ip19,consolidated,uncertainty),file=f)

f.close()


PM = np.array(playermatrix)
print('matrixshape:',PM.shape,PM[:,0:5].shape)
npitchers = PM.shape[0]

maxyear = 2
gamesplayed = 52.
wts = np.array([0.25,0.5,0.15,0.05,0.05])
oldpredictions = np.dot(PM[:,0:5],wts)

# now do some basic linear algebra to get the most predictive weights
# https://realpython.com/python-linear-algebra/
# A*x = b
# where A is the vector of our observations from this previous years (npitchers x 5)
# and b is the vector of observations from this year (npitchers)
x = np.dot(np.linalg.pinv(PM[:,0:maxyear]),(162./gamesplayed)*PM[:,5])
print(x)
print(np.sum(x))

newpredictions = np.dot(PM[:,0:maxyear],x)
actual = (162./gamesplayed)*PM[:,5]
uncertainty = np.std(PM[:,0:3],axis=1)


errornew = newpredictions - actual
errorold = oldpredictions - actual
terrornew = np.sqrt(np.sum(np.abs(errornew)**2)/npitchers)
terrorold = np.sqrt(np.sum(np.abs(errorold)**2)/npitchers)


f = open('validations/consolidated-validation-2023.csv','w')
print('player,ip2023real,ip2023scaled,new2023prediction,old2023prediction,old2023uncertainty,new2023sigma,old2023sigma',file=f)

for p in range(0,npitchers):
    try:
        print('{0:22s} {1:6.1f} {2:6.1f} {3:6.1f} {4:6.1f} {5:6.1f} {6:6.1f}'.format(playername[p],actual[p],newpredictions[p],oldpredictions[p],uncertainty[p],errornew[p]/uncertainty[p],errorold[p]/uncertainty[p]))
        print('{0},{1},{2},{3},{4},{5},{6},{7}'.format(playername[p],actual[p]*gamesplayed/162.,actual[p],newpredictions[p],oldpredictions[p],uncertainty[p],errornew[p]/uncertainty[p],errorold[p]/uncertainty[p]),file=f)
    except:
        print(playername[p])

f.close()
print('total error new={}'.format(terrornew))
print('total error old={}'.format(terrorold))

#print(DF)

# Index(['Unnamed: 0', '#', 'Name', 'Team', 'W', 'L', 'ERA', 'G', 'GS', 'CG',
#'ShO', 'SV', 'HLD', 'BS', 'IP', 'TBF', 'H', 'R', 'ER', 'HR', 'BB',
#'IBB', 'HBP', 'WP', 'BK', 'SO'],
#dtype='object')
#print(DF.columns)


"""
#print(DF['Team'].loc[DF['Team']=='ARI'])
plt.figure()

iprange = np.linspace(1,8,100)
plt.plot(iprange,-25*iprange+215,color='red',lw=4.)
plt.axis([1,7,0,250])
plt.xlabel('Pitcher Order')
plt.ylabel('IP')
plt.tight_layout()
plt.savefig('figures/summary{}.png'.format(year),dpi=300)
"""
