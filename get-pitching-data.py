


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

year = '2017'
DF = pd.read_csv('data/yearlystats-'+year+'.csv')

#print(DF)

# Index(['Unnamed: 0', '#', 'Name', 'Team', 'W', 'L', 'ERA', 'G', 'GS', 'CG',
#'ShO', 'SV', 'HLD', 'BS', 'IP', 'TBF', 'H', 'R', 'ER', 'HR', 'BB',
#'IBB', 'HBP', 'WP', 'BK', 'SO'],
#dtype='object')
print(DF.columns)

teams = np.unique(DF['Team'].values)
print(teams)

#print(DF['Team'].loc[DF['Team']=='ARI'])
plt.figure()

f = open('data/teamsummary{}.csv'.format(year),'w')

for team in teams:
    print('{0},'.format(team),end='',file=f)
    DFt = DF.loc[DF['Team']==team]
    nplrs = np.array(DFt['Name'].values).size
    nstarts = np.array(DFt['GS'].values)
    nnames = np.array(DFt['Name'].values)
    nips = np.array(DFt['IP'].values)
    nstartssort = nstarts[(-1.*nstarts).argsort()]
    nnamessort = nnames[(-1.*nstarts).argsort()]
    nipssort = nips[(-1.*nstarts).argsort()]
    abvzero = np.where(nstartssort>0)[0]
    plt.plot(np.arange(1,abvzero.size+1,1),nipssort[abvzero],color='black')
    for plr in range(0,nplrs):
        if nstartssort[plr]>0:
            #plt.scatter(plr,nipssort[plr],color='black')
            print(nstartssort[plr],nnamessort[plr])
            print('{0},'.format(nnamessort[plr]),end='',file=f)
    print(file=f)
    #print(DFt)

f.close()

iprange = np.linspace(1,8,100)
plt.plot(iprange,-25*iprange+215,color='red',lw=4.)
plt.axis([1,7,0,250])
plt.xlabel('Pitcher Order')
plt.ylabel('IP')
plt.tight_layout()
plt.savefig('figures/summary{}.png'.format(year),dpi=300)

"""
#get into stat-scraping
import stat_scraping as ss
year = '2023'
DF = ss.scrape_fangraphs_leaders('pit', year = 2023, data_type = 'Standard', agg_type='Player', split=0)
DF.to_csv('/Users/mpetersen/FantasyBaseball/pitching-rotations/data/yearlystats-'+year+'.csv')
"""
