import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer

def load_data(fname):
    """
    Reads in a csv file and return a dataframe. A dataframe df is similar to dictionary.
    You can access the label by calling df['label'], the content by df['content']
    the rating by df['rating']
    """
    return pd.read_csv(fname)

def multiclass_extract_dictionary(df):
    """
    Reads a panda dataframe, and returns a dictionary of distinct words
    mapping from each distinct word to its index (ordered by when it was found).
    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary of distinct words that maps each distinct word
        to a unique index corresponding to when it was first found while
        iterating over all words in each review in the dataframe df"""
    #replace all the punctuations (except ! and ' ) with space, then lowercase all the letters
    no_punc_lower = df['content'].str.replace(r'[1234567890"#$%&()*+,-./:;<=>?@\[\]^_`{|}~\\]', ' ').str.lower()
    #keep exclamation marks, keep
    no_punc_lower = no_punc_lower.str.replace(r'[!]',' !')

    #replace '(white) and '(non-white) pattern with just whitespace
    no_punc_lower = no_punc_lower.str.replace(r'(\'\s)|(\'\S)', ' ')

    #dictionary to return and the index to keep track of for each unique word
    word_dict = {}
    index = 0

    #go through each review in our dataset
    for review in no_punc_lower:
        #go through each word in the review
        for word in review.split():
            #if we encounter a unique word, add it to our dict and increment our index
            if(word not in word_dict):
                word_dict[word] = index
                index += 1
    
    return word_dict

def multiclass_generate_feature_matrix(df, word_dict):
    """
    Reads a dataframe and the dictionary of unique words
    to generate a matrix of {1, 0} feature vectors for each review.
    Use the word_dict to find the correct index to set to 1 for each place
    in the feature vector. The resulting feature matrix should be of
    dimension (number of reviews, number of words).
    Input:
        df: dataframe that has the ratings and labels
        word_list: dictionary of words mapping to indices
    Returns:
        a feature matrix of dimension (number of reviews, number of words)
    """
    #review count == rows | word count == columns
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)

    #feature matrix X = n x d
    feature_matrix = np.zeros((number_of_reviews, number_of_words))

    #replace all the punctuations (except ! and ' ) with space, then lowercase all the letters
    no_punc_lower = df['content'].str.replace(r'[1234567890"#$%&()*+,-./:;<=>?@\[\]^_`{|}~\\]', ' ').str.lower()

    #keep exclamation marks, keep
    no_punc_lower = no_punc_lower.str.replace(r'[!]', ' !')

    #replace '(white) and '(non-white) pattern with just whitespace
    no_punc_lower = no_punc_lower.str.replace(r'(\'\s)|(\'\S)', ' ')

    #go through every review
    for review_index in range(0, number_of_reviews ):
        #go through every word in each review
        for word in no_punc_lower[review_index].split():
            #check if current word is in the word_dict. if not, then skip it
            if(word in word_dict ):
                #if so, then add 1 to the word count features
                feature_matrix[review_index][word_dict[word ] ] = 1

    #create the tfidf function to return the tf idf feature matrix
    transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True )

    return transformer.fit_transform(feature_matrix).toarray()

def get_multiclass_training_data():
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are multiclass as follows
        -1: poor
        0: average
        1: good
    Also returns the dictionary used to create X_train.
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    dictionary = multiclass_extract_dictionary(dataframe)
    X_train = multiclass_generate_feature_matrix(dataframe, dictionary)
    Y_train = dataframe['rating'].values.copy()

    return (X_train, Y_train, dictionary)

def get_heldout_reviews(dictionary):
    """
    Reads in the data from data/dataset.csv and returns it as a feature
    matrix based on the functions extract_dictionary and generate_feature_matrix
    Input:
        dictionary: the dictionary created by get_multiclass_training_data
    """
    fname = "data/heldout.csv"
    dataframe = load_data(fname)
    X = multiclass_generate_feature_matrix(dataframe, dictionary)

    return X

def generate_challenge_labels(y, filename):
    """
    Takes in a numpy array that stores the prediction of your multiclass
    classifier and output the prediction to held_out_result.csv. Please make sure that
    you do not change the order of the ratings in the heldout dataset since we will
    this file to evaluate your classifier.
    """
    pd.Series(np.array(y)).to_csv(filename + '.csv', header=['rating'], index=False)
    return
