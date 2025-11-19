import os
import re

import matplotlib.pyplot as plt
plt.style.use('shendrukGroupStyle')
import shendrukGroupFormat as ed

directory = "/Users/johnwhitfield/Desktop/sims/n100/output"

unknotTimes=[]

numSims=0

#loop through output files
for item in os.listdir(directory):

    #join dir path 
    item_path = os.path.join(directory, item)

    #check if it exists
    if os.path.isdir(item_path):

        #create analysis path
        analysis_path=os.path.join(item_path, 'analysis')

        #check if the analysis path exists
        if os.path.isdir(analysis_path):

            #print('found analysis path', analysis_path)
            bu_path=os.path.join(analysis_path, 'BU__KN_polymer.dat')

            
            if os.path.isdir(analysis_path):
                #code here to find unknotting time from kymo output file

                #open the kymofile
                with open(bu_path, 'r') as f:
                    lines = f.readlines()

                #skip header

                lines = [line for line in lines[3:] if line.strip()]
                print(len(lines), bu_path)

                if len(lines)!=0:
                    numSims+=1
                
                for line in lines:
                    #clean line up and put into list to be able to access certain data
                    cleaned_line = re.sub(r'\s+', ' ', line.strip())
                    parts = cleaned_line.split(' ')

                    

                    #if the knot type is unknot, append timestep / 10 to get mpcd time for unknotting
                    #/10 rather than * 0.1 avoid floating point errors
                    if parts[3]=='UN':
                        unknottingTime=int(parts[0])/10
                        unknotTimes.append(unknottingTime)
                    
                        #kill rest of loop to avoid unecessary looping after knot has unknotted
                        break


print(unknotTimes)
print(numSims, 'numSims')

#handling conventional value for simulations which didn't unknot in the frame we checked them for
if len(unknotTimes)<numSims:
    for i in range((numSims-len(unknotTimes))):
        unknotTimes.append(-1)

print(' ')
print(unknotTimes)
print(len(unknotTimes))

#get valid data
validData=[x for x in unknotTimes if x!=-1]
neverUnknotted=unknotTimes.count(-1)

# plt.hist(unknotTimes, bins=len(unknotTimes), density=True, label='N=100')
# plt.title('Unknotting times, 25 Simulations')
# plt.legend(loc='upper right')
# plt.show()

fig, ax = plt.subplots(1, 2)

# Plot histogram for valid data with 5 bins
ax[0].set_title('Histogram of Unknot Times')
ax[0].hist(validData, bins=7, alpha=0.7, density=True, label='N=100', color='green')
ax[0].set_xlabel(r'$\tau_\mathrm{Unknot}$')
ax[0].set_ylabel('Frequency Density')
ax[0].legend(loc='upper right')

#normalise data
neverUnknottedProb=neverUnknotted/(neverUnknotted+len(validData))
unknottedProb=len(validData)/(neverUnknotted+len(validData))
ax[1].set_title(f"Outcomes for {numSims} Simulations")
ax[1].bar('Never Unknotted', neverUnknottedProb, color='red')
ax[1].bar('Unknotted', unknottedProb, color='green')
ax[1].set_ylabel('Frequency')

plt.show()