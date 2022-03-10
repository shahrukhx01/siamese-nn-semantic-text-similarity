import re

"""
Performs basic text cleansing on the unstructured field 
and adds additional column to the input dataframe
"""


class Preprocess:
    def __init__(self, stpwds_file_path):
        """
        Initializes regex patterns and load stopwords
        """
        self.USERNAME_PATTERN = (
            r"@([A-Za-z0-9_]+)"  ## regex pattern form removing user names
        )
        self.PUNCTUATION_PATTERN = (
            "'’|!@$%^&*()_+<>?:.,;-"  ## all punctuation symbols to be removed
        )
        self.STOPWORDS_PATH = stpwds_file_path  ## set stopwords file path
        self.load_stopwords()  ## load stopwords from file

    def load_stopwords(self):
        """
        Loads stopwords from file
        """
        stopwords_hindi_file = open(
            self.STOPWORDS_PATH, "r", encoding="utf-8"
        )  ## open file
        self.stopwords_hindi = [
            line.replace("\n", "") for line in stopwords_hindi_file.readlines()
        ]  ## add keywords to list for later use

    def remove_punctuations(self, text):
        """
        Removes punctuations from text field
        """
        return "".join([c for c in text if c not in self.PUNCTUATION_PATTERN])

    def remove_stopwords(self, text):
        """
        Removes stopwords from text field
        """
        return " ".join(
            [word for word in text.split() if word not in self.stopwords_hindi]
        )

    def remove_usernames(self, text):
        """
        Removes usernames from text field
        """
        return re.sub(self.USERNAME_PATTERN, "", text)

    def perform_preprocessing(self, data, columns_mapping):
        ## normalizing text to lower case
        data["clean_sent1"] = data[columns_mapping["sent1"]].apply(
            lambda text: text.lower()
        )
        data["clean_sent2"] = data[columns_mapping["sent2"]].apply(
            lambda text: text.lower()
        )
        ## removing usernames
        data["clean_sent1"] = data.clean_sent1.apply(self.remove_usernames)
        data["clean_sent2"] = data.clean_sent2.apply(self.remove_usernames)
        ## removing punctuations
        data["clean_sent1"] = data.clean_sent1.apply(self.remove_punctuations)
        data["clean_sent2"] = data.clean_sent2.apply(self.remove_punctuations)
        ## removing stopwords
        data["clean_sent1"] = data.clean_sent1.apply(self.remove_stopwords)
        data["clean_sent2"] = data.clean_sent2.apply(self.remove_stopwords)

        return data
