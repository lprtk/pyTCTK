# -*- coding: utf-8 -*-
"""
Author:
    lprtk

Description:
    It is a Python library to perform text cleaning. The purpose of this library
    is to give tools to prepare your text data without having to install anything.
    Some text cleaning libraries can't be used on professional computers because
    they need to download files from servers. With pyTCTK, you just need Python
    and access to GitHub.

License:
    MIT License
"""

import numpy as np
import os
import pandas as pd
import re
from urllib import request


#------------------------------------------------------------------------------


class TextNet:
    def __init__(self, data, column: str) -> None:
        """
        Function that allows to build the TextNet class and initialise the
        parameters.

        Parameters
        ----------
        data : pandas.core.frame.DataFrame or pandas.cores.series.Series
            Dataset to be cleaned.
        
        column : str
            - If 'data' parameter is a pandas.core.frame.DataFrame, 'column'
            parameter should be the 
            name assigned to the column to clean.
            - If 'data' parameter is a pandas.core.series.Series, 'column'
            parameter will be its output name.
        
        Raises
        ------
        TypeError
            - To use this class, the 'data' parameter must be a
            pandas.core.frame.DataFrame or pandas.cores.series.Series.
            - To use this class, the 'column' parameter must be a string.

        Returns
        -------
        None
            NoneType.

        """
        if isinstance(data, pd.core.frame.DataFrame):
            data.reset_index(drop=True, inplace=True)
            self.data = data
        elif isinstance(data, pd.core.series.Series):
            data.to_frame(
                name=column
            )
            data.reset_index(drop=True, inplace=True)
            self.data = data
        else:
            raise TypeError(
                f"'data' parameter must be a pandas.core.frame.DataFrame or pandas.cores.series.Series: got {type(data)}"
            )
        
        if isinstance(column, str):
            self.column = column
        else:
            raise TypeError(
                f"'column' parameter must be a str: got {type(column)}"
            )
    
    
    def downcast(self, category: bool=False) -> pd.core.frame.DataFrame:
        """
        Function that allows to cast the format of each column of a
        pandas.core.frame.DataFrame to optimize the RAM storage space.

        Parameters
        ----------
        category : bool, optional, default=False
            With Pandas, you can choose category or object to cast the type of
            a text. If category=False, then the text will be considered as an
            object otherwise as a category. Default is False.

        Raises
        ------
        TypeError
            To use this function, the 'category' parameter must be a boolean.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset optimized.

        """
        if isinstance(category, bool):
            pass
        else:
            raise TypeError(
                f"'category' parameter must be a bool: got {type(category)}"
            )
        
        columns = self.data.dtypes.index.tolist()
        types = self.data.dtypes.values.tolist()
        
        for i, t in enumerate(types):
            # interger format
            if "int" in str(t):
                if self.data[columns[i]].min() > np.iinfo(np.int8).min and\
                    self.data[columns[i]].max() < np.iinfo(np.int8).max:
                    self.data[columns[i]] = self.data[columns[i]].astype(np.int8)
                elif self.data[columns[i]].min() > np.iinfo(np.int16).min and\
                    self.data[columns[i]].max() < np.iinfo(np.int16).max:
                    self.data[columns[i]] = self.data[columns[i]].astype(np.int16)
                elif self.data[columns[i]].min() > np.iinfo(np.int32).min and\
                    self.data[columns[i]].max() < np.iinfo(np.int32).max:
                    self.data[columns[i]] = self.data[columns[i]].astype(np.int32)
                else:
                    self.data[columns[i]] = self.data[columns[i]].astype(np.int64)
            # float format
            elif "float" in str(t):
                if self.data[columns[i]].min() > np.finfo(np.float16).min and\
                    self.data[columns[i]].max() < np.finfo(np.float16).max:
                    self.data[columns[i]] = self.data[columns[i]].astype(np.float16)
                elif self.data[columns[i]].min() > np.finfo(np.float32).min and\
                    self.data[columns[i]].max() < np.finfo(np.float32).max:
                    self.data[columns[i]] = self.data[columns[i]].astype(np.float32)
                else:
                    self.data[columns[i]] = self.data[columns[i]].astype(np.float64)
            # object format
            elif "object" in str(t):
                # timestamp format
                if columns[i] in ["date", "Date", "DATE", "dates", "Dates", "DATES"]:
                    self.data[columns[i]] = pd.to_datetime(
                        self.data[columns[i]],
                        format="%Y-%m-%d"
                    )
                else:
                    if category == False:
                        self.data[columns[i]] = self.data[columns[i]].astype("object")
                    else:
                        self.data[columns[i]] = self.data[columns[i]].astype("category")
        
        return self.data
    
    
    def lowercase(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to transform to lowercase each word from a sentence
        in a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        self.data[self.column] = self.data[self.column].str.lower()
        
        return self.data
    
    
    def word_count_filter(self, min_words: int=2) -> pd.core.frame.DataFrame:
        """
        Function that allows to filter the number of words for each sentence in
        a dataset.

        Parameters
        ----------
        min_words : int, optional, default=2
            Number of words at least the length of the sentence should make.
            Default is 2.

        Raises
        ------
        TypeError
            To use this function, the 'min_words' parameter must be a integer.

        Returns
        -------
        dataframe_filter : pandas.core.frame.DataFrame
            Dataset filtered.

        """
        if isinstance(min_words, int):
            pass
        else:
            raise TypeError(
                f"'min_words' parameter must be an int: got {type(min_words)}"
            )
        
        list_wc = []
        
        for i in range(0, self.data[self.column].shape[0]):
            list_wc.append(
                len(
                    self.data[self.column][i].split(" ")
                )
            )
        
        dataframe_wc = pd.DataFrame(
            {
                "Word count": list_wc
            }
        )
        dataframe_filter = pd.concat(
            [
                self.data, dataframe_wc["Word count"]
            ],
            axis=1
        )
        dataframe_filter = dataframe_filter[
            dataframe_filter["Word count"] > min_words
        ]
        dataframe_filter.drop("Word count", axis=1, inplace=True)
        dataframe_filter.reset_index(drop=True, inplace=True)
        
        return dataframe_filter
    
    
    def remove_punctuation(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all punctuation marks (./,;?!#~\@) from
        each sentence in a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = re.sub(
                r"[^\w\s\-–'‘’]",
                "",
                self.data[self.column][i]
            )
            
            self.data[self.column][i] = re.sub(
                r"[-–'‘’]",
                " ",
                self.data[self.column][i]
            )
        
        return self.data
    
    
    def remove_url(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all Uniform Resource Locators (URLs) from
        each sentence in a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = re.sub(
                r"https?://\S+|www\.\S+",
                "",
                self.data[self.column][i],
                flags=re.IGNORECASE
            )
        
        return self.data
    
    
    def remove_html(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all HTML tags (<...> or </...>) from each
        sentence in a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = re.sub(
                r"<.*?>",
                "",
                self.data[self.column][i],
                flags=re.IGNORECASE
            )
        
        return self.data
    
    
    def remove_email(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all email adresses (...@...) from each
        sentence in a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = re.sub(
                r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
                "",
                self.data[self.column][i],
                flags=re.IGNORECASE
            )
        
        return self.data
    
    
    def remove_digit(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all digits ([0-9]) from each sentence in
        a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = re.sub(
                r"\d+",
                "",
                self.data[self.column][i]
            )
        
        return self.data
    
    
    def remove_space(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove the spaces at the beginning and end of
        each sentence in a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = self.data[self.column][i].strip()
        
        return self.data
    
    
    def remove_whitespace(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to replace all extra spaces between two words with
        a single space in each sentence of a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = re.sub(
                r"\s+",
                " ",
                self.data[self.column][i]
            )
        
        return self.data
    
    
    def remove_mention(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all mentions (@...) from each sentence in
        a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = re.sub(
                r"@\w+",
                "",
                self.data[self.column][i]
            )
        
        return self.data
    
    
    def remove_hastag(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all hastags (#...) from each sentence in
        a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = re.sub(
                r"#\w+",
                "",
                self.data[self.column][i]
            )
        
        return self.data
    
    
    def remove_emoji(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all emojis (😂, 🤔, 🙈, 😌, 💕, 👭, 👙)
        from each sentence in a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """        
        regex_emojis = re.compile(
            r"["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002500-\U00002BEF"
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"
            "]+",
            re.UNICODE
        )
        
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = re.sub(
                regex_emojis,
                "",
                self.data[self.column][i]
            )
        
        return self.data
    
    
    def additional_cleaning(self, add_regexs: list=None) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove other non-textual characters (¤¶‰™©®▶➤¿
        ∎≤≥⋅﹣°☒) from each sentence in a dataset. You can add regexs to delete
        characters that are unique to your text with 'add_regexs' parameter.

        Parameters
        ----------
        add_regexs : list, optional, default=None
            You can add regex to delete characters that are unique to your text.
            Default is None.
            Exemple of use: add_regexs = [r"a-zA-Z", r"0-9", r"\d+"]

        Raises
        ------
        TypeError
            To use this function, the 'add_regexs' parameter must be None or a list.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        list_regexs = [
            r"\n",
            r"\t",
            r"\r",
            r"[‘’„“„”“”「」『』…]",
            r"[¤¶‰™©®]",
            r"[▶➤¿∎≤≥⋅﹣°☒]"
        ]
        
        if isinstance(add_regexs, list):
            for regex in add_regexs:
                list_regexs.append(regex)
        elif add_regexs == None:
            pass
        else:
            raise TypeError(
                f"'add_regexs' parameter must be None or a list: got {type(add_regexs)}"
            )
        
        for i in range(0, self.data[self.column].shape[0]):
            for regex in list_regexs:
                self.data[self.column][i] = re.sub(
                    regex,
                    "",
                    self.data[self.column][i],
                    flags=re.IGNORECASE
                )
        
        return self.data
    
    
    def remove_accent(self, lowercase: bool=True) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all accents (with encoding or not) from
        each sentence in a dataset.

        Parameters
        ----------
        lowercase : bool, optional, default=True
            If true, the text will be transform to lowercase before cleaning.
            Otherwise the cleaning is applied to the text as in input.
            Default is True.

        Raises
        ------
        TypeError
            To use this function, the 'lowercase' parameter must be a boolean.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        if isinstance(lowercase, bool):
            if lowercase == True:
                self.data = TextNet(data=self.data, column=self.column).lowercase()
            else:
                pass
        else:
            raise TypeError(
                f"'lowercase' parameter must be a bool: got {type(lowercase)}"
            )
        
        url = "https://raw.githubusercontent.com/lprtk/pyTCTK/main/ressources/accents/accents.txt"
        path = os.getcwd()
        filename = "\\accents.txt"
        
        _DownloadDeleteFile(path=path, filename=filename)._download_file(url=url)
        
        dict_regexs = {}
        
        with open(path+filename, "r") as file:
            for line in file:
                (key, value) = line.split()
                dict_regexs[str(key)] = str(value)
        
        _DownloadDeleteFile(path=path, filename=filename)._remove_file()
        
        for i in range(0, self.data[self.column].shape[0]):
            for regex in dict_regexs:
                self.data[self.column][i] = re.sub(
                    regex,
                    dict_regexs[regex],
                    self.data[self.column][i],
                    flags=re.IGNORECASE
                )
        
        return self.data
    
    
    def remove_single_character(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove the unique characters from each sentence
        in a dataset.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = " ".join(
                [
                    word for word in self.data[self.column][i].split(" ") if len(word) > 1
                ]
            )
        
        return self.data
    
    
    def remove_plural(self, word_length: int=5) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all the s character from words in the plural
        from each sentence in a dataset.

        Parameters
        ----------
        word_length : int, optional, default=5
            Length of the words to which the s must be deleted. Default is 5.

        Raises
        ------
        TypeError
            To use this function, the 'word_length' parameter must be an integer.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        if isinstance(word_length, int):
            pass
        else:
            raise TypeError(
                f"'word_length' parameter must be an int: got {type(word_length)}"
            )
        
        for i in range(0, self.data[self.column].shape[0]):
            list_words = []
            
            for word in self.data[self.column][i].split(" "):
                if len(word) > word_length:
                    word = re.sub(
                        r"s\b",
                        "",
                        word,
                        flags=re.IGNORECASE
                    )
                else:
                    pass
                
                list_words.append(word)
        
            self.data[self.column][i] = " ".join(list_words)
        
        return self.data


#------------------------------------------------------------------------------


class WordNet:
    def __init__(self, data, column: str) -> None:
        """
        Function that allows to build the WordNet class and initialise the parameters.

        Parameters
        ----------
        data : pandas.core.frame.DataFrame or pandas.cores.series.Series
            Dataset to be cleaned.
        
        column : str
            - If 'data' parameter is a pandas.core.frame.DataFrame, 'column'
            parameter should be the 
            name assigned to the column to clean.
            - If 'data' parameter is a pandas.core.series.Series, 'column'
            parameter will be its output name.
        
        Raises
        ------
        TypeError
            - To use this class, the 'data' parameter must be a 
            pandas.core.frame.DataFrame or pandas.cores.series.Series.
            - To use this class, the 'column' parameter must be a string.

        Returns
        -------
        None
            NoneType.

        """
        if isinstance(data, pd.core.frame.DataFrame):
            data.reset_index(drop=True, inplace=True)
            self.data = data
        elif isinstance(data, pd.core.series.Series):
            data.to_frame(
                name=column
            )
            data.reset_index(drop=True, inplace=True)
            self.data = data
        else:
            raise TypeError(
                f"'data' parameter must be a pandas.core.frame.DataFrame or pandas.cores.series.Series: got {type(data)}"
            )
        
        if isinstance(column, str):
            self.column = column
        else:
            raise TypeError(
                f"'column' parameter must be a str: got {type(column)}"
            )
    
    
    def remove_stopword(self, language: str="english", lowercase: bool=True,
                        remove_accents: bool=False, remove_stopwords: list=None,
                        add_stopwords: list=None) -> pd.core.frame.DataFrame:
        """
        Function that allows to remove all stopwords from each sentence in a dataset.

        Parameters
        ----------
        language : {"english", "french"}, str, optional, default="english"
            Language of stopwords to be removed. Default is "english".
            
        lowercase : bool, optional, default=True
            If true, the text will be transform to lowercase before cleaning.
            Otherwise the cleaning is applied to the text as in input.
            Default is True.
            
        remove_accents : bool, optional, default=False
            If false, the accents on words will not be removed before cleaning.
            Otherwise, the cleaning is applied on the words without accents.
            Default is False.

        remove_stopwords : list, optional, default=None
            You can specify a list of stopwords that you do not want to delete
            from your textual corpus in order to keep them.
            Default is None.

        add_stopwords : list, optional, default=None
            You can specify a list of stopwords that you want to remove from your
            text corpus in addition to the basic stopwords.
            Default is None.

        Raises
        ------
        TypeError
            - To use this function, the 'language' parameter must be a string.
            - To use this function, the 'lowercase' parameter must be a boolean.
            - To use this function, the 'remove_accents' parameter must be a boolean.
            - To use this function, the 'remove_stopwords' parameter must be a list.
            - To use this function, the 'add_stopwords' parameter must be a list.

        ValueError
            To use this function, the 'language' parameter must be {"english", "french"}

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        if isinstance(language, str):
            pass
        else:
            raise TypeError(
                f"'language' parameter must be a str: got {type(language)}"
            )
        
        if isinstance(lowercase, bool):
            if lowercase == True:
                self.data = TextNet(data=self.data, column=self.column).lowercase()
            else:
                pass
        else:
            raise TypeError(
                f"'lowercase' parameter must be a bool: got {type(lowercase)}"
            )
        
        if language == "english":
            url = "https://raw.githubusercontent.com/lprtk/pyTCTK/main/ressources/stopwords/english.txt"
            path = os.getcwd()
            filename = "\\english.txt"
            
            _DownloadDeleteFile(path=path, filename=filename)._download_file(url=url)
            
            with open(path+filename, "r") as file:
                list_stopwords = file.read().split()
            
            _DownloadDeleteFile(path=path, filename=filename)._remove_file()
        elif language == "french":
            url = "https://raw.githubusercontent.com/lprtk/pyTCTK/main/ressources/stopwords/french.txt"
            path = os.getcwd()
            filename = "\\french.txt"
            
            _DownloadDeleteFile(path=path, filename=filename)._download_file(url=url)
            
            with open(path+filename, "r") as file:
                list_stopwords = file.read().split()
            
            _DownloadDeleteFile(path=path, filename=filename)._remove_file()
        else:
            raise ValueError(
                "'language' parameter must be in {'english', 'french'}: default='english'"
            )
        
        if isinstance(remove_stopwords, list):
            for stopword in remove_stopwords:
                if stopword in list_stopwords:
                    list_stopwords.remove(stopword)
                else:
                    pass
        elif remove_stopwords == None:
            pass
        else:
            raise TypeError(
                f"'remove_stopwords' parameter must be None or a list: got {type(remove_stopwords)}"
            )
        
        if isinstance(add_stopwords, list):
            for stopword in add_stopwords:
                list_stopwords.append(stopword)
        elif add_stopwords == None:
            pass
        else:
            raise TypeError(
                f"'add_stopwords' parameter must be None or a list: got {type(add_stopwords)}"
            )
        
        list_stopwords_len = len(list_stopwords)
        
        if isinstance(remove_accents, bool):
            if remove_accents == True:
                self.data = TextNet(
                    data=self.data,
                    column=self.column
                ).remove_accent(
                    lowercase=lowercase
                )
                df_stopwords = pd.DataFrame(list_stopwords, columns=["Stopwords"])
                df_stopwords = TextNet(
                    data=df_stopwords,
                    column="Stopwords"
                ).remove_accent(
                    lowercase=lowercase
                )
                list_stopwords = df_stopwords["Stopwords"].tolist()
                
                assert len(list_stopwords) == list_stopwords_len, "list_stopwords' shape must remain the same as before deletion"
            else:
                pass
        else:
            raise TypeError(
                f"'remove_accents' parameter must be a bool: got {type(remove_accents)}"
            )
        
        list_stopwords = list(set(list_stopwords))
        
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = " ".join(
                [
                    word for word in self.data[self.column][i].split(" ") if word not in list_stopwords
                ]
            )
        
        return self.data
    
    
    def lemmatize(self, language: str="english", lowercase: bool=True,
                  remove_accents: bool=False) -> pd.core.frame.DataFrame:
        """
        Function that allows each sentence of a dataset to be lemmatized.
        In other words, for each sentence, nouns are replaced by their radical
        and verbs by their infinitive.
        
        Example of use: saw > see, mice > mouse, took > take, recommended > recommend

        Parameters
        ----------
        language : {"english", "french"}, str, optional, default="english"
            Language of stopwords to be removed. Default is "english".
            
        lowercase : bool, optional, default=True
            If true, the text will be transform to lowercase before cleaning.
            Otherwise the cleaning is applied to the text as in input.
            Default is True.
            
        remove_accents : bool, optional, default=False
            If false, the accents on words will not be removed before cleaning.
            Otherwise, the cleaning is applied on the words without accents.
            Default is False.

        Raises
        ------
        TypeError
            - To use this function, the 'language' parameter must be a string.
            - To use this function, the 'lowercase' parameter must be a boolean.
            - To use this function, the 'remove_accents' parameter must be a boolean.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        if isinstance(language, str):
            pass
        else:
            raise TypeError(
                f"'language' parameter must be a str: got {type(language)}"
            )
        
        if isinstance(lowercase, bool):
            if lowercase == True:
                self.data = TextNet(data=self.data, column=self.column).lowercase()
            else:
                pass
        else:
            raise TypeError(
                f"'lowercase' parameter must be a bool: got {type(lowercase)}"
            )
        
        if language == "english":
            url = "https://raw.githubusercontent.com/lprtk/pyTCTK/main/ressources/lemme/english.txt"
            path = os.getcwd()
            filename = "\\english.txt"
            
            _DownloadDeleteFile(path=path, filename=filename)._download_file(url=url)
            
            dict_regexs = {}

            with open(path+filename, "r") as file:
                for line in file:
                    (key, value) = line.split()
                    dict_regexs[str(key)] = str(value)
        
            _DownloadDeleteFile(path=path, filename=filename)._remove_file()
        elif language == "french":
            url = "https://raw.githubusercontent.com/lprtk/pyTCTK/main/ressources/lemme/french.txt"
            path = os.getcwd()
            filename = "\\french.txt"
            
            _DownloadDeleteFile(path=path, filename=filename)._download_file(url=url)
            
            dict_regexs = {}

            with open(path+filename, "r") as file:
                for line in file:
                    (key, value) = line.split()
                    dict_regexs[str(key)] = str(value)
        
            _DownloadDeleteFile(path=path, filename=filename)._remove_file()
        
        if isinstance(remove_accents, bool):
            if remove_accents == True:
                self.data = TextNet(
                    data=self.data,
                    column=self.column
                ).remove_accent(
                    lowercase=lowercase
                )
                df_regexs = pd.DataFrame(
                    list(dict_regexs.items()),
                    columns=["Keys", "Values"]
                )
                df_regexs = TextNet(
                    data=df_regexs,
                    column="Keys"
                ).remove_accent(
                    lowercase=lowercase
                )
                df_regexs = TextNet(
                    data=df_regexs,
                    column="Values"
                ).remove_accent(
                    lowercase=lowercase
                )
                liste_keys = df_regexs["Keys"].tolist()
                liste_values = df_regexs["Values"].tolist()
                dict_regexs = dict(zip(liste_keys, liste_values))
            else:
                pass
        else:
            raise TypeError(
                f"'remove_accents' parameter must be a bool: got {type(remove_accents)}"
            )
        
        for i in range(0, self.data[self.column].shape[0]):
            for regex in dict_regexs:
                self.data[self.column][i] = re.sub(
                    regex,
                    dict_regexs[regex],
                    self.data[self.column][i],
                    flags=re.IGNORECASE
                )
        
        return self.data
    
    
    def stemmatize(self, language: str="english", lowercase: bool=True, remove_accents: bool=False) -> pd.core.frame.DataFrame:
        """
        Function that allows each sentence of a dataset to be stemmatized.
        In other words, for each sentence, verbs and words are replaced by their
        root / base word.
        
        Example of use: ponies > poni, dogs > dog, running > runn

        Parameters
        ----------
        language : {"english", "french"}, str, optional, default="english"
            Language of stopwords to be removed. Default is "english".
            
        lowercase : bool, optional, default=True
            If true, the text will be transform to lowercase before cleaning.
            Otherwise the cleaning is applied to the text as in input.
            Default is True.
            
        remove_accents : bool, optional, default=False
            If false, the accents on words will not be removed before cleaning.
            Otherwise, the cleaning is applied on the words without accents.
            Default is False.

        Raises
        ------
        TypeError
            - To use this function, the 'language' parameter must be a string.
            - To use this function, the 'lowercase' parameter must be a boolean.
            - To use this function, the 'remove_accents' parameter must be a boolean.

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        if isinstance(language, str):
            pass
        else:
            raise TypeError(
                f"'language' parameter must be a str: got {type(language)}"
            )
        
        if isinstance(lowercase, bool):
            if lowercase == True:
                self.data = TextNet(data=self.data, column=self.column).lowercase()
            else:
                pass
        else:
            raise TypeError(
                f"'lowercase' parameter must be a bool: got {type(lowercase)}"
            )
        
        if language == "english":
            url = "https://raw.githubusercontent.com/lprtk/pyTCTK/main/ressources/stemme/english.txt"
            path = os.getcwd()
            filename = "\\english.txt"
            
            _DownloadDeleteFile(path=path, filename=filename)._download_file(url=url)
            
            with open(path+filename, "r") as file:
                list_regexs = file.read().split()
            
            _DownloadDeleteFile(path=path, filename=filename)._remove_file()
        elif language == "french":
            url = "https://raw.githubusercontent.com/lprtk/pyTCTK/main/ressources/stemme/french.txt"
            path = os.getcwd()
            filename = "\\french.txt"
            
            _DownloadDeleteFile(path=path, filename=filename)._download_file(url=url)
            
            with open(path+filename, "r") as file:
                list_regexs = file.read().split()
            
            _DownloadDeleteFile(path=path, filename=filename)._remove_file()
        
        list_regexs_len = len(list_regexs)
        
        if isinstance(remove_accents, bool):
            if remove_accents == True:
                self.data = TextNet(
                    data=self.data,
                    column=self.column
                ).remove_accent(
                    lowercase=lowercase
                )
                df_regexs = pd.DataFrame(list_regexs, columns=["Regexs"])
                df_regexs = TextNet(
                    data=df_regexs,
                    column="Regexs"
                ).remove_accent(
                    lowercase=lowercase
                )
                list_regexs = df_regexs["Regexs"].tolist()
                
                assert len(list_regexs) == list_regexs_len, "list_regexs' shape must remain the same as before deletion"
            else:
                pass
        else:
            raise TypeError(
                f"'remove_accents' parameter must be a bool: got {type(remove_accents)}"
            )
        
        for i in range(0, self.data[self.column].shape[0]):
            for regex in list_regexs:
                self.data[self.column][i] = re.sub(
                    regex,
                    "",
                    self.data[self.column][i],
                    flags=re.IGNORECASE
                )
        
        return self.data


#------------------------------------------------------------------------------


class Tokenize:
    def __init__(self, data, column: str) -> None:
        """
        Function that allows to build the Tokenize class and initialise the
        parameters.

        Parameters
        ----------
        data : pandas.core.frame.DataFrame or pandas.cores.series.Series
            Dataset to be cleaned.
        
        column : str
            - If 'data' parameter is a pandas.core.frame.DataFrame, 'column'
            parameter should be the 
            name assigned to the column to clean.
            - If 'data' parameter is a pandas.core.series.Series, 'column'
            parameter will be its output name.
        
        Raises
        ------
        TypeError
            - To use this class, the 'data' parameter must be a
            pandas.core.frame.DataFrame or pandas.cores.series.Series.
            - To use this class, the 'column' parameter must be a string.

        Returns
        -------
        None
            NoneType.

        """
        if isinstance(data, pd.core.frame.DataFrame):
            data.reset_index(drop=True, inplace=True)
            self.data = data
        elif isinstance(data, pd.core.series.Series):
            data.to_frame(
                name=column
            )
            data.reset_index(drop=True, inplace=True)
            self.data = data
        else:
            raise TypeError(
                f"'data' parameter must be a pandas.core.frame.DataFrame or pandas.cores.series.Series: got {type(data)}"
            )
        
        if isinstance(column, str):
            self.column = column
        else:
            raise TypeError(
                f"'column' parameter must be a str: got {type(column)}"
            )
    
    
    def word_tokenize(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to transform in token each word from a sentence in
        a dataset. Tokenize a word is the fact that divide single string into
        list of substring.
        
        Example of use: >>> string = "I'm a Data Scientist"
                        ... ["I'm", "a", "Data", "Scientist"]

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = self.data[self.column][i].split(" ")
        
        return self.data
    
    
    def word_detokenize(self) -> pd.core.frame.DataFrame:
        """
        Function that allows to transform in sentence a most of tokens from an
        observation in a dataset. Detokenize a list of tokens is the fact that
        join list of substring into a single string.
        
        Example of use: >>> string = ["I'm", "a", "Data", "Scientist"]
                        ... "I'm a Data Scientist"

        Returns
        -------
        self.data : pandas.core.frame.DataFrame
            Dataset cleaned.

        """
        for i in range(0, self.data[self.column].shape[0]):
            self.data[self.column][i] = " ".join(
                self.data[self.column][i]
            )
        
        return self.data


#------------------------------------------------------------------------------


class _DownloadDeleteFile:
    def __init__(self, path: str, filename: str) -> None:
        """
        Function that allows to build the _DownloadDeleteFile class and initialise
        the parameters.

        Parameters
        ----------
        path : str
            Path to where to download the file.
        
        filename : str
            Name of the file.
        
        Raises
        ------
        TypeError
            - To use this class, the 'path' parameter must be a string.
            - To use this class, the 'filename' parameter must be a string.

        Returns
        -------
        None
            NoneType.

        """
        if isinstance(path, str):
            self.path = path
        else:
            raise TypeError(
                f"'path' parameter must be a str: got {type(path)}"
            )
        
        if isinstance(filename, str):
            self.filename = filename
        else:
            raise TypeError(
                f"'filename' parameter must be a str: got {type(filename)}"
            )
    
    
    def _download_file(self, url: str) -> None:
            """
            Hidden function that allows to dowload file from an url.
    
            Parameters
            ----------
            url : str
                URL of the file to doawload.
    
            Returns
            -------
            None
                NoneType.
    
            """
            request.urlretrieve(url=url, filename=self.path+self.filename)
    
    
    def _remove_file(self) -> None:
        """
        Hidden function that allows to delete a downloaded file.

        Returns
        -------
        None
            NoneType.

        """
        os.remove(self.path+self.filename)