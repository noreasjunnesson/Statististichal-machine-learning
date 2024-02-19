import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#
#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('png')
#from IPython.core.pylabtools import figsize
#figsize(10, 6) # Width and hight
#plt.style.use('seaborn-white')


url = 'siren_data_train.csv'
file_path = 'siren_data_train.csv'
siren_data= pd.read_csv(url, na_values='?', dtype={'index': str}).dropna().reset_index()

#siren_data.info()
#pd.plotting.scatter_matrix(siren_data.iloc[:,1:13],figsize=(10,10))


x_coor=siren_data.xcoor
y_coor=siren_data.ycoor
near_x=siren_data.near_x
near_y=siren_data.near_y
coord=np.array([x_coor,y_coor])
near= np.array([near_x, near_y])


# Calculate Euclidean distances for each pair
distances = np.sqrt((x_coor - near_x)**2 + (y_coor - near_y)**2)


#----------------------------------------------------------
# Percentage heard for different distances
#----------------------------------------------------------
distance_ranges = [0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,27000]
percentage_heard_dist = []

for i in range(len(distance_ranges) - 1):
    lower_bound = distance_ranges[i]
    upper_bound = distance_ranges[i + 1]
    in_range=[]
    for dist,stat in zip(distances,siren_data.heard):
        if lower_bound <= dist < upper_bound:
            in_range.append(stat)
    # Calculate the percentage of people who heard the siren in this range
    if in_range:
        percentage = (sum(in_range) / len(in_range)) * 100
    else:
        percentage = 0
    percentage_heard_dist.append(percentage)

# Bar plot for percentage of people who heard the siren in each distance range
plt.bar(range(len(percentage_heard_dist)), percentage_heard_dist, align='center')
plt.xticks(range(len(distance_ranges) - 1), [f'{lower}-{upper}' for lower, upper in zip(distance_ranges[:-1], distance_ranges[1:])], rotation=45,fontsize=6)
plt.title('Percentage of People Heard Siren in Each Distance Range')
plt.xlabel('Distance Range')
plt.ylabel('Percentage Heard')
plt.grid(True)
plt.show()

#----------------------------------------------------------
# Percentage heard for different ages
#----------------------------------------------------------
age_ranges= [18,28,38,48,58,68,78,88]
percentage_heard_age=[]

for i in range(len(age_ranges) -1):
    lower_bound = age_ranges[i]
    upper_bound = age_ranges[i + 1]
    in_range=[]
    for stat,age in zip(siren_data.heard,siren_data.age):
        if lower_bound<=age<upper_bound:
            in_range.append(stat)
    # Calculate the percentage of people who heard the siren in this range
    if in_range:
        percentage = (sum(in_range) / len(in_range)) * 100
    else:
        percentage = 0
    percentage_heard_age.append(percentage)
    
plt.figure()
plt.bar(range(len(percentage_heard_age)), percentage_heard_age, align='center')
plt.xticks(range(len(age_ranges) - 1), [f'{lower}-{upper}' for lower, upper in zip(age_ranges[:-1], age_ranges[1:])])
plt.title('Heard dependent on age')
plt.xlabel('Age Range')
plt.ylabel('Heard Percentage')
plt.show()

#----------------------------------------------------------
# Percentage heard for different angles
#----------------------------------------------------------
angle_ranges= [20,60,100,140,180]
percentage_heard_angle=[]

for i in range(len(angle_ranges) -1):
    lower_bound = angle_ranges[i]
    upper_bound = angle_ranges[i + 1]
    in_range=[]
    for stat,ang in zip(siren_data.heard,siren_data.near_angle):
        if lower_bound<=abs(ang)<upper_bound:
            in_range.append(stat)
    # Calculate the percentage of people who heard the siren in this range
    if in_range:
        percentage = (sum(in_range) / len(in_range)) * 100
    else:
        percentage = 0
    percentage_heard_angle.append(percentage)
    
plt.figure()
plt.bar(range(len(percentage_heard_angle)), percentage_heard_angle, align='center')
plt.xticks(range(len(angle_ranges) - 1), [f'{lower}-{upper}' for lower, upper in zip(angle_ranges[:-1], angle_ranges[1:])])
plt.title('Heard dependent on angle')
plt.xlabel('Angle Range')
plt.ylabel('Heard Percentage')
plt.show()



# Calculate the correlation matrix
correlation_matrix =siren_data.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
