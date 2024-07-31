import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load("hand_measurements.npy", allow_pickle=True)



# Set the style for Seaborn
sns.set_style("whitegrid")

print(data.shape)
# Generate some example data
data1 = data[:, 0]
data2 = data[:, 1]
data3 = data[:, 2]
data4 = data[:, 3]
data5 = data[:, 4]


# [8.130227853389986, 7.254204211141307, 8.065768925214446, 7.413432260445578, 6.091503812837406]

# right pinky = 5.9
# right ring = 7.5
# right middle = 8.4
# right index = 7.3
# right_hand_dims = [7.3, 8.4, 7.5, 6.2]
# left_hand_dims = [7.3, 8.2, 7.4, 5.9]
left_hand_paul= [7.5, 8.3, 7.8, 6.2]
right_hand_dims = [7.5, 8.2, 7.5, 6.2] # paul
right_hand_dims = [7.3, 8.2, 7.4, 6] # jagger
right_hand_dims = [7.3, 7.9, 7.5, 6.2]
# Create a figure and axis
plt.figure(figsize=(12, 6))

# Plot histograms
sns.histplot(data1, kde=False, color="blue", label="Thumb")
sns.histplot(data2, kde=True, color="red", label="Index")
plt.axvline(x=right_hand_dims[0], color='red', linestyle='--')
sns.histplot(data3, kde=True, color="green", label="Middle")
plt.axvline(x=right_hand_dims[1], color='green', linestyle='--')
sns.histplot(data4, kde=True, color="purple", label="Ring")
plt.axvline(x=right_hand_dims[2], color='purple', linestyle='--')
sns.histplot(data5, kde=True, color="orange", label="Pinky")
plt.axvline(x=right_hand_dims[3], color='orange', linestyle='--')

# Customize the plot
plt.title("Hand Measurements as Histograms", fontsize=16)
plt.xlabel("Value", fontsize=12)
plt.xlim(3, 10)
plt.ylabel("Frequency", fontsize=12)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
