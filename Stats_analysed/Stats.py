import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'siren_data_train.csv'
file_path = 'siren_data_train.csv'
siren_data= pd.read_csv(url, na_values='?', dtype={'index': str}).dropna().reset_index()


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
    
fit_range = 10
coefficients = np.polyfit(range(fit_range), percentage_heard_dist[:fit_range], 1)
linear_fit = np.poly1d(coefficients)

# Bar plot for percentage of people who heard the siren in each distance range
plt.bar(range(len(percentage_heard_dist)), percentage_heard_dist, align='center')
plt.plot(range(fit_range), linear_fit(range(fit_range)), 'r-', label='Linear Fit')
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

x_data = np.arange(len(age_ranges) - 1)
popt, pcov = curve_fit(square_function, x_data, percentage_heard_age)
y_fit = square_function(x_data, *popt)
    
plt.figure()
plt.bar(range(len(percentage_heard_age)), percentage_heard_age, align='center')
plt.plot(x_data, y_fit, 'r-', label='Fitted Square Function')
plt.xticks(range(len(age_ranges) - 1), [f'{lower}-{upper}' for lower, upper in zip(age_ranges[:-1], age_ranges[1:])])
plt.title('Heard dependent on age')
plt.xlabel('Age Range')
plt.ylabel('Heard Percentage')
plt.show()

#----------------------------------------------------------
# Percentage heard for different angles
#----------------------------------------------------------
angle_ranges= [-180,-160,-140,-120,-100,-80,-60,-40,-20,-0,20,40,60,80,100,120,140,160,180]
percentage_heard_angle=[]

for i in range(len(angle_ranges) -1):
    lower_bound = angle_ranges[i]
    upper_bound = angle_ranges[i + 1]
    in_range=[]
    for stat,ang in zip(siren_data.heard,siren_data.near_angle):
        if lower_bound<=ang<upper_bound:
            in_range.append(stat)
    # Calculate the percentage of people who heard the siren in this range
    if in_range:
        percentage = (sum(in_range) / len(in_range)) * 100
    else:
        percentage = 0
    percentage_heard_angle.append(percentage)

coefficients_first = np.polyfit(range(first_fit_range), percentage_heard_angle[:first_fit_range], 1)
linear_fit_first = np.poly1d(coefficients_first)
coefficients_last = np.polyfit(range(last_fit_range), percentage_heard_angle[-last_fit_range:], 1)
linear_fit_last = np.poly1d(coefficients_last)
    
plt.figure()
plt.bar(range(len(percentage_heard_angle)), percentage_heard_angle, align='center')
plt.plot(range(first_fit_range), linear_fit_first(range(first_fit_range)), 'r-')
plt.plot(range(len(percentage_heard_angle)-last_fit_range, len(percentage_heard_angle)), linear_fit_last(range(last_fit_range)), 'g-')
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
