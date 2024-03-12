# Data python file to load the data
import os
import pandas as pd

print(f'If its working will print all Dataset shapes ⏳')

# Get the current directory of the Python file
current_directory = os.path.realpath('data.py')

# Navigate to the parent directory
parent_directory = os.path.dirname(current_directory)


class ImportData:

    def __ini__(self):
        self.current_directory = current_directory
        self.parent_directory = parent_directory

    def goemotions():

        data_goemotions = pd.read_csv(os.path.join(parent_directory, 'data','raw','GoEmotions_full_raw.csv'))

        return data_goemotions

    def abbreviations():

        data_abbreviations = pd.read_csv(os.path.join(parent_directory, 'data','raw','Abbreviations_and_Slang.csv'))

        return data_abbreviations

    def slangs():

        data_slangs = pd.read_csv(os.path.join(parent_directory, 'data','raw','slang.csv'))[['acronym', 'expansion']]

        return data_slangs

print(f'GoEmotions Dataset shape: {ImportData.goemotions().shape}')
print(f'Data Abbreviations shape: {ImportData.abbreviations().shape}')
print(f'Data Slangs shape: {ImportData.slangs().shape}')
