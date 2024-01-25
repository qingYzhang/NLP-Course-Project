import pandas as pd
import numpy as np
import sys
import os


# # Replace 'your_token_here' with the token you obtained from Genius
# genius = Genius('stTfOxjIJBB-bRuzVpKoLB1sCXJjAMy6Np3zl0LFClqavd2a51z0fAH8dxZXs1Wp')



# # Search for Drake and attempt to retrieve all songs
# # The `max_songs` parameter is set to None to try and get all songs
# # Note: This may take a long time and can be subject to rate limiting
# artists = ["Drake", "Lil Wayne", "Migos"]
# for p in artists:
#     artist = genius.search_artist(p, max_songs=2)
    
#     # Loop through each song in the artist object and print the lyrics
#     for song in artist.songs:
#         print(song.title)
#         print(song.lyrics)

# lyrics_df["Rhyme pairs"] = lyrics_df.apply(annotator.annotate_rhyme_pairs, axis=1)
#     # Filter out poems without rhyme pairs
#     lyrics_df = lyrics_df[lyrics_df["Rhyme pairs"].apply(lambda x: len(x)) > 0]

# # Add the parent directory to the system path
# sys.path.append(os.path.abspath('...'))

top30k= pd.read_csv("./data/lyrics_dataset_raw.csv")
# Assuming top30k is your DataFrame and 'lyrics' is the column to be processed
square_brackets = r'\[.*?\]\n'
parenthesis = r'\(.*?\)'
no_punc = r'(?<=[\w\s])[!"#$%&\'*+,-./:;<=>?@\[\\\]^_`{|}~]+(?=\n)'
extra_line = r'\n\n+'
white_space = r' \n'


top30k['Content'] = (top30k['lyrics'].str.replace(square_brackets, '', regex=True)
                  .str.lstrip('\n')
                  .str.replace(no_punc, '', regex=True)
                  .str.replace(extra_line, '\n', regex=True)                  
                  .str.replace(parenthesis, '', regex=True)
                  .str.replace(white_space, '\n', regex=True))

import sys

# Path to the common base directory
base_dir = './data'

# Add the base directory to the system path
sys.path.append(base_dir)
# Path to the directory which should be the current working directory
new_working_directory = './'

# Change the current working directory
os.chdir(new_working_directory)

# Now you can import and use your Annotator class as expected
from annotator import Annotator
annotator = Annotator();
point1=top30k

top30k["Rhyme pairs"] = top30k.apply(annotator.annotate_rhyme_pairs, axis=1)
# Filter out poems without rhyme pairs
top30k = top30k[top30k["Rhyme pairs"].apply(lambda x: len(x)) > 0]
point2=top30k


def extract_and_pair_lyrics(row):
    """
    Extracts lyrics from one column and pairs them based on indices in another column.

    Args:
    row (pd.Series): A row of the DataFrame.

    Returns:
    list of tuples: Paired lyrics.
    """
    lyrics = row['lyrics_column']  # Adjust 'lyrics_column' to your DataFrame's column name
    selected_indices = row['Rhyme pairs']   # Adjust 'indices_column' to your DataFrame's column name
    paired_lyrics = [(lyrics[i[0]], lyrics[i[1]]) for i in selected_indices]

    return paired_lyrics

## Lyrics column consists of lyrics in list
top30k['lyrics_column'] = top30k['Content'].str.split('\n')
top30k['paired_lyrics'] = top30k.apply(extract_and_pair_lyrics, axis=1)

all_rhymed = top30k.paired_lyrics.to_list()

songs = [i for i in all_rhymed]
lines = [line for song in songs for line in song]



# Extracting all first elements of the tuples
first_elements = [elem[0] for sublist in songs[:5000] for elem in sublist]

# Extracting all second elements of the tuples
second_elements = [elem[1] for sublist in songs[:5000] for elem in sublist]

rhymed = pd.DataFrame({'first_line': first_elements, 'next_line': second_elements})

rhymed.to_csv("./data/lyrics_rhyming_pairs_large.csv")


# import pandas as pd

# # Load the two CSV files into DataFrames
# file1 = pd.read_csv('./data/lyrics_dataset_raw.csv')
# file2 = pd.read_csv('./data/lyrics_dataset_raw.csv')

# # Find rows in file1 that are not in file2
# result = pd.merge(file1, file2, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)

# # Save the result to a new CSV file
# result.to_csv('./data/result.csv', index=False)





# for i in range(0, len(flat_data) - sequence_length):
#     input_sequences.append(flat_data[i:i + sequence_length])
#     target_sequences.append(flat_data[i + sequence_length])
