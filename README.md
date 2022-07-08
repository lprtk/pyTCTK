<h1 align="center">pyTCTK for Python Text Cleaning ToolKit</h1> 

<p align="center"> 
<a href="https://github.com/lprtk/pyTCTK/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/lprtk/pyTCTK"></a> 
<a href="https://github.com/lprtk/pyTCTK/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/lprtk/pyTCTK"></a> 
<a href="https://github.com/lprtk/pyTCTK/stargazers"><img alt="Github Stars" src="https://img.shields.io/github/stars/lprtk/pyTCTK "></a> 
<a href="https://github.com/lprtk/pyTCTK/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/lprtk/pyTCTK"></a> 
<a href="https://github.com/lprtk/pyTCTK/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> 
</p> 

## Table of contents 
* [Overview :loudspeaker:](#Overview)
* [Content :mag_right:](#Content)
* [Requirements :page_with_curl:](#Requirements)
* [File details :open_file_folder:](#File-details)
* [Features :computer:](#Features) 

<a id="section01"></a> 
## Overview 

<p align="justify">The objective is to give tools to prepare your text data without having to install anything. Some text cleaning libraries can't be used on professional computers because they need to download files from servers or from urls that are blocked by internet proxies. With pyTCTK, you just need Python and access to GitHub to clean your text data. So it's a library that you can use on your professional computer, that's the goal : a library usable everywhere.<p>

<a id="section02"></a> 
## Content 

For the moment, three class with several functions are available:
<ul> 
<li><p align="justify">The TextNet class implements all the general functions to clean up your text (remove punctuation, uppercase, email address, urls, html tags, etc.);</p></li> 
<li><p align="justify">The WordNet class implements all the functions to perform more precise cleaning at the word level of your text (remove stopwords or apply lemming or stemming);</p></li>
<li><p align="justify">The Tokenize class implements all two functions to tokenize and detokenize the words in your text.</p></li>
</ul> 

<a id="section03"></a> 
## Requirements
* **Python version 3.9.7** 
* **Install requirements.txt** 
```console
$ pip install -r requirements.txt 
``` 

* **Librairies used**
```python
import numpy as np
import os
import pandas as pd
import re
from urllib import request
``` 

<a id="section04"></a> 
## File details
* **requirements**
* This folder contains a .txt file with all the packages and versions needed to run the project. 
* **pyTCTK**
* This folder contains a .py file with all class, functions and methods. 
* **example**
* This folder contains an example notebook to better understand how to use the different class and functions, and their outputs.
* **ressources**
* This folder contains several subfolders in which there are .txt vocabulary files for processing and cleaning the texts.

</br> 

Here is the project pattern: 
```
- project
    > pyTCTK
        > requirements
            - requirements.txt
        > codefile 
            - pyTCTK.py
        > example 
            - pyTCTK.ipynb
        > ressources 
            >stopwords
                - english.txt
                - french.txt
            >lemme
                - english.txt
                - french.txt
            >stemme
                - english.txt
                - french.txt
            >accents
                - accents.txt
```

<a id="section05"></a> 
## Features 
<p align="center"><a href="https://github.com/lprtk/lprtk">My profil</a> • 
<a href="https://github.com/lprtk/lprtk">My GitHub</a></p>
