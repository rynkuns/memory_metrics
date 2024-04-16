import numpy as np

from openai import OpenAI

### TF-IDF relevant imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def isnotebook():
    # https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook#39662359
    try:
        shell = get_ipython().__class__.__name__
        print(shell)
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        # elif shell == "google.colab._shell":
        #     return True
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm



class Metrics():
    

    def __init__(self, target_text:str|list=None, texts:list|dict=None, openai_key:str=None) -> None:
        if (target_text != None) and (texts != None):
            self.set_texts(target_text, texts)
        elif (target_text == None) != (texts == None):
            raise Exception("Both 'target_text' and 'texts' must be specified to assign them on object initialization.")
        else:
            self.texts_scores = {}
        
        if openai_key != None:
            self.set_openai(openai_key)

    def __cos_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    
    def set_texts(self, target_text:str|list, texts:list|dict):
        """Set the texts for which metrics will be calculated.

        Args:
            target_text (str | list): The target / standard text for comparison in form of either a single string, or a list of strings for pairwise comparison.
            texts (list | dict): A list of texts to compare, or a dict of lists of texts; if 'target' arg is a list, then each of the lists in 'texts' must be of the same lenght as in 'target'.

        Raises:
            Exception: TODO
            Exception: TODO
        """

        if type(texts) not in [list, dict]:
            raise TypeError("Texts to compare must be provided either by list or dictionary of lists.")
        self.texts = texts
        self.texts_vecs = {}
        self.texts_scores = {}
        if type(target_text) not in [str, list]:
            raise TypeError("Target text(s) must be provided either by string or list.")
        self.target_text = target_text
        self.target_text_vecs = {}

    def set_openai(self, key:str, model:str="text-embedding-3-small"):
        """_summary_

        Args:
            key (str): Your private key for OpenAI API. Keep this value secret.
            model (str, optional): A string representing which OpenAI embedding model will be used. Defaults to "text-embedding-3-small".

        Raises:
            TypeError: TODO
        """
        if type(key) is not str:
            raise TypeError("OpenAI key must be a string.")
        self.openai_client = OpenAI(api_key = key)
        self.openai_model = model

        
    def calculate_openai(self, progress_bar:bool=True):
        get_openai_embedding = lambda text: self.openai_client.embeddings.create(input = [text], model=self.openai_model).data[0].embedding

        if not progress_bar:
            local_tqdm = lambda x: x
        elif progress_bar:
            local_tqdm = tqdm
            
        if type(self.target_text) == str:
            self.target_text_vecs["OpenAI"] = get_openai_embedding(self.target_text)
        elif type(self.target_text) == list:
            self.target_text_vecs["OpenAI"] = [get_openai_embedding(txt) for txt in local_tqdm(self.target_text)]#, desc="Creating target texts' embeddings")]

        if type(self.texts) == dict:
            self.texts_vecs["OpenAI"] = {}
            for key, value in self.texts.items():
                self.texts_vecs["OpenAI"][key] = [get_openai_embedding(txt) for txt in local_tqdm(value)]#, desc="Creating comparison texts' embeddings")]
            ### Scores
            self.texts_scores["OpenAI"] = {}
            for key, value in self.texts_vecs["OpenAI"].items():
                if type(self.target_text) == str:
                    self.texts_scores["OpenAI"][key] = [self.__cos_similarity(self.target_text_vecs["OpenAI"], vec) for vec in local_tqdm(value)]#, desc="Creating comparison texts' scores")]
                elif type(self.target_text) == list:
                    self.texts_scores["OpenAI"][key] = [self.__cos_similarity(self.target_text_vecs["OpenAI"][i], value[i]) for i in local_tqdm(range(len(value)))]#, desc="Creating comparison texts' scores")]

        elif type(self.texts) == list:
            self.texts_vecs["OpenAI"] = [get_openai_embedding(txt) for txt in local_tqdm(self.texts)]#, desc="Creating comparison texts' embeddings")]
            ### Scores
            if type(self.target_text) == str:
                self.texts_scores["OpenAI"] = [self.cos_similarity(self.target_text_vecs["OpenAI"], vec) for vec in local_tqdm(self.texts_vecs["OpenAI"])]#, desc="Creating comparison texts' scores")]
            elif type(self.target_text) == list:
                self.texts_scores["OpenAI"] = [self.cos_similarity(self.target_text_vecs["OpenAI"][i], self.texts_vecs["OpenAI"][i]) for i in local_tqdm(range(len(self.texts_vecs["OpenAI"])))]#, desc="Creating comparison texts' scores")]

    def calculate_tfidf(self, progress_bar:bool=True, lang:str="english"):
        nltk.download('stopwords')
        nltk.download('wordnet')
        stop_words = stopwords.words(lang)
        lemmatizer = WordNetLemmatizer()

        def prepare_tokens(text):
            tokens = word_tokenize(text)
            tokens = [token for token in tokens if token not in stop_words]
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            # tokens = [token for token in tokens if token not in stop_words] ### potentialy useful?
            return tokens
        
        def tfidf_similarity(tokens1, tokens2):
            vectorizer = TfidfVectorizer()
            corpus = [" ".join(text) for text in [tokens1,tokens2]]
            matrix_tfidf = vectorizer.fit_transform(corpus).todense()
            vec1, vec2 = [np.squeeze(np.asarray(matrix_tfidf[i,:])) for i in range(2)]
            similarity = self.__cos_similarity(vec1, vec2)
            return similarity

        if not progress_bar:
            local_tqdm = lambda x: x
        elif progress_bar:
            local_tqdm = tqdm

        if type(self.target_text) == str:
            self.target_text_vecs["TF-IDF"] = prepare_tokens(self.target_text)
        elif type(self.target_text) == list:
            self.target_text_vecs["TF-IDF"] = [prepare_tokens(txt) for txt in local_tqdm(self.target_text)]#, desc="Creating target texts' embeddings")]

        if type(self.texts) == dict:
            self.texts_vecs["TF-IDF"] = {}
            for key, value in self.texts.items():
                self.texts_vecs["TF-IDF"][key] = [prepare_tokens(txt) for txt in local_tqdm(value)]#, desc="Creating comparison texts' embeddings")]
            ### Scores
            self.texts_scores["TF-IDF"] = {}
            for key, value in self.texts_vecs["TF-IDF"].items():
                if type(self.target_text) == str:
                    self.texts_scores["TF-IDF"][key] = [tfidf_similarity(self.target_text_vecs["TF-IDF"], vec) for vec in local_tqdm(value)]#, desc="Creating comparison texts' scores")]
                elif type(self.target_text) == list:
                    self.texts_scores["TF-IDF"][key] = [tfidf_similarity(self.target_text_vecs["TF-IDF"][i], value[i]) for i in local_tqdm(range(len(value)))]#, desc="Creating comparison texts' scores")]

        elif type(self.texts) == list:
            self.texts_vecs["TF-IDF"] = [prepare_tokens(txt) for txt in local_tqdm(self.texts)]#, desc="Creating comparison texts' embeddings")]
            ### Scores
            if type(self.target_text) == str:
                self.texts_scores["TF-IDF"] = [tfidf_similarity(self.target_text_vecs["TF-IDF"], vec) for vec in local_tqdm(self.texts_vecs["TF-IDF"])]#, desc="Creating comparison texts' scores")]
            elif type(self.target_text) == list:
                self.texts_scores["TF-IDF"] = [tfidf_similarity(self.target_text_vecs["TF-IDF"][i], self.texts_vecs["TF-IDF"][i]) for i in local_tqdm(range(len(self.texts_vecs["TF-IDF"])))]#, desc="Creating comparison texts' scores")]

    def which_calculated(self):
        return self.texts_scores.keys()
        
    def scores(self, which:list=None) -> dict:
        """Return the calculated metrics.

        Args:
            which (list, optional): List os strings, which selects which kmind of metrics will be return (all by default). Defaults to None.

        Returns:
            dict: Dictionary with metrics, where keys are strings representing the type of a metric. 
        """
        possible = ["OpenAI", "TF-IDF"]
        if which == None:
            return self.texts_scores
        elif which <= possible:
            return {k: v for k, v in self.texts_scores.items() if k in which}
        else:
            raise Exception(f"Unexpected keys in 'which' arg. 'which' must be a subset of {possible}.")

