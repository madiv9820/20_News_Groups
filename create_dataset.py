import os
import pandas as pd

# Define a class to create the dataset from text files in subfolders
class CreateDataset:
    def __init__(self, folderPath: str) -> None:
        # Initialize with the path to the folder containing category subfolders
        self.__folderPath = folderPath

     # String representation of the class
    def __repr__(self):
        return 'class CreateDataset'

    def create_Text_Dataset(self) -> None:
        # Create an empty list to store the data
        data = []

        # Loop through each subfolder (representing categories) in the main folder
        for categoryFolder in os.listdir(self.__folderPath):
            categoryPath = os.path.join(self.__folderPath, categoryFolder)  # Get the full path of the category folder
            category = categoryPath.split('/')[-1]  # Extract the folder name as the category label

            # Check if it's a directory (category folder)
            if os.path.isdir(categoryPath):
                try:
                    # Loop through each file in the category folder
                    for fileName in os.listdir(categoryPath):
                        filePath = os.path.join(categoryPath, fileName)  # Full file path
                        
                        # Check if it is a file (and not a directory or something else)
                        if os.path.isfile(filePath):
                            # Open the file and read its content
                            with open(filePath, 'r', encoding='ISO-8859-1') as file:
                                text = file.read()  # Read the content of the file
                                
                                # Append the file content and the corresponding category to the data list
                                data.append({'File Content': text, 'Category': category})
                except Exception as e:
                    # If there is an issue reading the files, print the error message
                    print('Unable to fetch Data')
                    raise e

        # Convert the list of dictionaries into a pandas DataFrame
        dataset = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        dataset.to_csv('News_Data.csv', index = False)