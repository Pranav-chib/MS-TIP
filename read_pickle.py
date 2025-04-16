import pickle

# Specify the path to your .pkl file
file_path = ''

# Open the file in binary mode
with open(file_path, 'rb') as file:
    # Load the content of the Pickle file
    data = pickle.load(file)

# Print the content
print(data)
