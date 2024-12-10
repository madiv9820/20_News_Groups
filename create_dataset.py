import os
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Class to create a dataset from text files in subfolders
class TextDataset:
    def __init__(self, folderPath: str) -> None:
        """
        Initialize the class with the path to the folder containing category subfolders.

        Args:
        - folderPath (str): Path to the root folder containing subfolders of text files.
        """
        self.__folderPath = folderPath

    def __repr__(self):
        """
        String representation of the class.
        """
        return 'class TextDataset'

    def create_Text_Dataset(self) -> None:
        """
        Create a dataset from text files in subfolders and save it as a CSV file.
        """
        # Initialize an empty list to store the data
        data = []

        # Loop through each subfolder in the main folder (representing categories)
        for categoryFolder in os.listdir(self.__folderPath):
            categoryPath = os.path.join(self.__folderPath, categoryFolder)  # Get full path of the category folder
            category = categoryPath.split('/')[-1]  # Extract category name (subfolder name)

            # Check if the path is a directory (to ignore non-folder entries)
            if os.path.isdir(categoryPath):
                try:
                    # Loop through all files in the category folder
                    for fileName in os.listdir(categoryPath):
                        filePath = os.path.join(categoryPath, fileName)  # Full path of the file
                        
                        # Check if the path is a valid file
                        if os.path.isfile(filePath):
                            # Read the file content
                            with open(filePath, 'r', encoding = 'ISO-8859-1') as file:
                                text = file.read()  # Read file content
                                
                                # Append the file content and category to the data list
                                data.append({'File Content': text, 'Category': category})
                except Exception as e:
                    # Handle exceptions (e.g., file reading errors)
                    print('Unable to fetch Data')
                    raise e

        # Convert the list of dictionaries into a pandas DataFrame
        dataset = pd.DataFrame(data)

        # Save the DataFrame to a CSV file
        dataset.to_csv('News_Data.csv', index = False)

# Class to process the text dataset and extract word-level features
class WordDataset:
    def __init__(self):
        """
        Initialize the class and load stopwords from an external source.
        """
        # URL to fetch the stopwords
        stopWordLink = "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt"
        stopwords_list = requests.get(stopWordLink).content
        stopWords = set(stopwords_list.decode().splitlines())
        
        self.__stopWords = stopWords  # Set of stopwords
        self.__featureWords = []  # List of feature words (words that pass the frequency threshold)

    def __remove_Unnecessary_Punctuations(self, content: str) -> str:
        """
        Remove unnecessary punctuations and special characters from the content.

        Args:
        - content (str): The input string.

        Returns:
        - str: The cleaned content.
        """
        # List of unwanted characters to remove
        unwanted_chars = ['.', '/', '!', '@', '#', '&', '*', '<', '>', '?', ',', 
                          '+', '-', '_', '\\', '\'', '"', ':', '\t', '=', '(', ')', 
                          '^', '~', '`', '[', ']', '{', '}', '|', '%', ';', '\n']
        
        # Replace each unwanted character with a space
        for char in unwanted_chars:
            content = content.replace(char, ' ')
            
        return content
    
    def __find_featureWords(self, x: pd.DataFrame) -> None:
        """
        Find words with a frequency above a threshold and visualize their frequencies.

        Args:
        - x (pd.DataFrame): The input DataFrame containing the text content.
        """
        word_Frequency = defaultdict(int)  # Dictionary to store word frequencies

        # Loop through each piece of content
        for content in x:
            # Remove punctuations and split into words
            content = self.__remove_Unnecessary_Punctuations(content=content)
            words = content.split(' ')

            # Count word frequencies (excluding stopwords and non-alphabetic words)
            for word in words:
                if word != '' and word.lower() not in self.__stopWords and word.isalpha():
                    word_Frequency[word.lower()] += 1

        # Analyze word frequency distribution
        frequency_For_Each_Word_Count = list(Counter(word_Frequency.values()).items())
        frequency_For_Each_Word_Count.sort(reverse = True, key = lambda x: x[1])
        frequency_For_Each_Word_Count = np.array(frequency_For_Each_Word_Count, dtype=int)

        # Plot word frequency distribution
        maximum_Frequency = frequency_For_Each_Word_Count[0][1]
        plt.figure(figsize=(19, 18))
        plt.scatter(frequency_For_Each_Word_Count[:, 0], frequency_For_Each_Word_Count[:, 1])
        plt.plot(frequency_For_Each_Word_Count[:, 0], frequency_For_Each_Word_Count[:, 1])
        plt.axis((0, 41, 0, maximum_Frequency + 1))
        plt.yticks(np.arange(0, maximum_Frequency + 500, 200))
        plt.xticks(np.arange(0, 41, 1))
        plt.xlabel('<----- No of Words ----->', fontsize = 25)
        plt.ylabel('<--- Frequency ---->', fontsize = 25)
        plt.title('No of words vs Frequency', fontsize = 25)
        plt.grid()
        plt.show()

        # Allow the user to set the minimum frequency threshold
        self.__k = int(input('Enter k for which words having frequency at least k: '))
        self.__featureWords = [word for word, frequency in word_Frequency.items() if frequency >= self.__k]

        # Display feature word statistics
        print('Total No of Words as Feature: ', len(self.__featureWords))
        print('Words as Feature: ', self.__featureWords)

    def create_Dataset(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Create a dataset based on feature words.

        Args:
        - x (pd.DataFrame): The input text data.
        - y (pd.DataFrame): The corresponding labels (categories).

        Returns:
        - pd.DataFrame: The transformed dataset with feature word frequencies.
        """
        if len(self.__featureWords):
            wordData = []  # List to store transformed data
            x = np.array(x)
            y = np.array(y)

            # Loop through each piece of content
            for index in range(len(x)):
                # Create a dictionary to store feature word frequencies
                content_dictionary = {word: 0 for word in self.__featureWords}
                content = self.__remove_Unnecessary_Punctuations(content=x[index])
                words = content.split(' ')

                # Count feature word occurrences in the content
                for word in words:
                    if word in self.__featureWords:
                        content_dictionary[word] += 1
                
                # Add the category label to the dictionary
                content_dictionary['Category'] = y[index]
                wordData.append(content_dictionary)
            
            # Convert the list of dictionaries to a DataFrame
            wordDataset = pd.DataFrame(wordData)
            return wordDataset

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the model to the data by finding feature words and creating the dataset.

        Args:
        - x (pd.DataFrame): The input text data.
        - y (pd.DataFrame): The corresponding labels (categories).

        Returns:
        - pd.DataFrame: The transformed dataset with feature word frequencies.
        """
        self.__find_featureWords(x = x)
        return self.create_Dataset(x = x, y = y)