import os
import re
import glob
import matplotlib.pyplot as plt
plt.style.use('shendrukGroupStyle')
import shendrukGroupFormat as ed

from polyAnalyser import *

from vas_faster import *

directory = "/Users/johnwhitfield/Desktop/localsims/n100/output"

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
                        #parts[0] is the timestep, so need to convert
                        unknottingTime=int(parts[0])/10
                        unknotTimes.append(unknottingTime)
                    
                        #kill rest of loop to avoid unecessary looping after knot has unknotted
                        break


print(unknotTimes)
print(numSims, 'numSims')

base_dir = "/Users/johnwhitfield/Desktop/localSims/n100/output"
base_sim_dir = "XYZ_knot3_1.n100.*_output"
timeSteps = 6000
vassilievs = []

sim_dirs = glob.glob(os.path.join(base_dir, base_sim_dir))

def extract_number(path):
    match = re.search(r'\.(\d+)_output', path)
    if match:
        return int(match.group(1))
    return 0

sim_dirs_sorted = sorted(sim_dirs, key=extract_number)

# Use enumerate to get both index and directory
for i, sim_dir in enumerate(sim_dirs_sorted):
    print(f"Sim {i+1} being processed")
    
    # Find the timestamp subdirectory
    for item in os.listdir(sim_dir):
        item_path = os.path.join(sim_dir, item)
        if os.path.isdir(item_path) and item.replace('-', '').isdigit():
            vtf_file = os.path.join(item_path, f"{item}-vmd.vtf")
            if os.path.exists(vtf_file):
                try:
                    a = polyAnalyser(vtf_file, timeSteps)
                    
                    # Check if we have an unknot time for this simulation
                    if i < len(unknotTimes):
                        unknot_time = unknotTimes[i]
                        vassiliev = a.getVassilievInvariants(unknot_time)
                        vassilievs.append(vassiliev)
                        print(f"  Unknot time: {unknot_time}")
                    else:
                        print(f"  No unknot time available for simulation {i+1}")
                        vassilievs.append(None)  # or some default value
                    
                except Exception as e:
                    print(f"  ERROR: {e}")
                    vassilievs.append(None)
                break
    else:
        print(f"  No timestamp directory found in {sim_dir}")
        vassilievs.append(None)

print("Vassiliev invariants:", vassilievs)


def plot_vassiliev_statistics(vassilievs):
    # Filter out None values and ensure all arrays have the same length
    valid_data = [v for v in vassilievs if v is not None]
    
    if not valid_data:
        print("No valid data to plot")
        return
    
    # Find the minimum length among all arrays
    min_length = min(len(arr) for arr in valid_data)
    
    # Truncate all arrays to the same length
    truncated_data = [arr[:min_length] for arr in valid_data]
    
    # Convert to numpy array for easier calculations
    data_array = np.array(truncated_data)
    
    # Calculate statistics
    mean_vals = np.mean(data_array, axis=0)
    std_vals = np.std(data_array, axis=0)
    sem_vals = std_vals / np.sqrt(len(valid_data))  # Standard Error of the Mean
    
    # Create x-axis (time points relative to unknotting time)
    # Since you're using range(unknottingTime-20, unknottingTime+20)
    time_points = np.arange(-400, 400)  # Relative to unknotting time
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot mean line
    plt.plot(time_points, mean_vals, 'b-', linewidth=2, label='Mean Vassiliev Invariant')
    
    # Fill between mean ± standard error
    plt.fill_between(time_points, 
                    mean_vals - sem_vals, 
                    mean_vals + sem_vals, 
                    alpha=0.3, color='blue', label='± SEM')
    
    # Optional: also show standard deviation
    # plt.fill_between(time_points, 
    #                 mean_vals - std_vals, 
    #                 mean_vals + std_vals, 
    #                 alpha=0.2, color='red', label='± STD')
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Unknotting Time')
    plt.xlabel('Time Relative to Unknotting (frames)')
    plt.ylabel('Vassiliev Invariant')
    plt.title('Vassiliev Invariant Around Unknotting Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Number of valid simulations: {len(valid_data)}")
    print(f"Mean at unknotting time (t=0): {mean_vals[20]:.3f} ± {sem_vals[20]:.3f}")
    print(f"Overall mean: {np.mean(mean_vals):.3f}")
    print(f"Overall SEM range: {np.mean(sem_vals):.3f}")




plot_vassiliev_statistics(vassilievs)


#WILL NEED TO LOOK AT MORE FRAMES AROUND THE UNKNOT TIME, 20 IS VERY SMALL







#handling conventional value for simulations which didn't unknot in the frame we checked them for
# if len(unknotTimes)<numSims:
#     for i in range((numSims-len(unknotTimes))):
#         unknotTimes.append(-1)

# print(' ')
# print(unknotTimes)
# print(len(unknotTimes))

###############
#Plotting
###############


#get valid data
# validData=[x for x in unknotTimes if x!=-1]
# neverUnknotted=unknotTimes.count(-1)

# # plt.hist(unknotTimes, bins=len(unknotTimes), density=True, label='N=100')
# # plt.title('Unknotting times, 25 Simulations')
# # plt.legend(loc='upper right')
# # plt.show()

# fig, ax = plt.subplots(1, 2)

# # Plot histogram for valid data with 5 bins
# ax[0].set_title('Histogram of Unknot Times')
# ax[0].hist(validData, bins=7, alpha=0.7, density=True, label='N=100', color='green')
# ax[0].set_xlabel(r'$\tau_\mathrm{Unknot}$')
# ax[0].set_ylabel('Frequency Density')
# ax[0].legend(loc='upper right')

# #normalise data
# neverUnknottedProb=neverUnknotted/(neverUnknotted+len(validData))
# unknottedProb=len(validData)/(neverUnknotted+len(validData))
# ax[1].set_title(f"Outcomes for {numSims} Simulations")
# ax[1].bar('Never Unknotted', neverUnknottedProb, color='red')
# ax[1].bar('Unknotted', unknottedProb, color='green')
# ax[1].set_ylabel('Frequency')

# plt.show()