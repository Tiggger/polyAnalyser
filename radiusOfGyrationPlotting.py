import os
import re
import numpy as np 

import matplotlib.pyplot as plt
plt.style.use('shendrukGroupStyle')
import shendrukGroupFormat as ed

directory = "/Users/johnwhitfield/Desktop/sims/n100/output"

radiusOfGyrations=[]

numSims=0

#loop through output files
for item in os.listdir(directory):
    simResults=[]

    #join dir path 
    item_path = os.path.join(directory, item)

    #check if it exists
    if os.path.isdir(item_path):

        #create analysis path
        analysis_path=os.path.join(item_path, 'analysis')

        #check if the analysis path exists
        if os.path.isdir(analysis_path):

            #print('found analysis path', analysis_path)
            rg_path=os.path.join(analysis_path, 'Rg_polymer_rg.dat')

            
            if os.path.isdir(analysis_path):
                #code here to find unknotting time from kymo output file

                #print('can see the radius of gyration file')

                #open the kymofile
                with open(rg_path, 'r') as f:
                    lines = f.readlines()


                #only doing for simulations that ran
                if len(lines)!=0:
                    numSims+=1
                
                for line in lines:
                    #clean line up and put into list to be able to access certain data
                    cleaned_line = re.sub(r'\s+', ' ', line.strip())
                    parts = cleaned_line.split(' ')

                    

                    #get radius of gyration and append to sim results
                    rg=parts[1]
                    simResults.append(float(rg))


    if len(simResults)!=0:
        radiusOfGyrations.append(simResults)



print(len(radiusOfGyrations), 'length of radiusOfGyrations (should be 25)')



#convert to numpy and take average
radiusOfGyrations=np.array(radiusOfGyrations)

#need to take sqrt

radiusOfGyrations=np.sqrt(radiusOfGyrations)


avgRadiusOfGyrations=np.mean(radiusOfGyrations, axis=0)
stdRadiusOfGyration=np.std(radiusOfGyrations, axis=0)

stdErrorRadiusOfGyration=stdRadiusOfGyration/np.sqrt(25)


print(len(avgRadiusOfGyrations), 'should be 6001')

plt.plot(np.linspace(0, 600, 6001), avgRadiusOfGyrations, label='N=100')

plt.fill_between(np.linspace(0, 600, 6001), 
                 avgRadiusOfGyrations - stdErrorRadiusOfGyration,
                 avgRadiusOfGyrations + stdErrorRadiusOfGyration,
                 alpha=0.3, color='blue', label='± Standard Error')

plt.xlabel(r'$t_\mathrm{MPCD}$')
plt.ylabel(r'$R_g$')
plt.title('Radius of Gyration Averaged for 25 Simulations')
plt.legend(loc='upper right')


plt.show()
