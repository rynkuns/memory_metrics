from openai import OpenAI
import numpy as np
import tqdm


class Metrics():

    def __init__(self, target_text=None, texts=None, openai_key:str=None) -> None:
        if (target_text != None) and (texts != None):
            self.set_texts(target_text, texts)
        elif (target_text == None) != (texts == None):
            raise Exception("Both 'target_text' and 'texts' must be specified to assign them on object initialization.")
        
        if openai_key != None:
            self.set_openai(openai_key)

    def __cos_similarity(vec1, vec2):
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
    
    def set_texts(self, target_text, texts):
        if type(texts) not in [list, dict]:
            raise Exception("Texts to compare must be provided either by list or dictionary of lists. TODO documentation")
        self.texts = texts
        self.texts_vecs = {}
        self.texts_scores = {}
        if type(target_text) not in [str, list]:
            raise Exception("Target text(s) must be provided either by string or list. TODO documentation")
        self.target_text = target_text
        self.target_text_vecs = {}

    def set_openai(self, key:str, model:str="text-embedding-3-small"):
        if type(key) is not str:
            raise Exception("OpenAI key must be a string.")
        self.openai_client = OpenAI(api_key = key)
        self.get_openai_embedding = lambda text: self.openai_client.embeddings.create(input = [text], model=model).data[0].embedding

        
    def calculate_openai(self, progress_bar:bool=True):
        if not progress_bar:
            tqdm = lambda x: x
        if type(self.target_text) == str:
            self.target_text_vecs["OpenAI"] = self.get_openai_embedding(self.target_text)
        elif type(self.target_text) == list:
            self.target_text_vecs["OpenAI"] = [self.get_openai_embedding(txt) for txt in tqdm(self.target_text)]

        if type(self.texts) == dict:
            self.texts_vecs["OpenAI"] = {}
            for key, value in self.texts.items():
                self.texts_vecs["OpenAI"][key] = [self.get_openai_embedding(txt) for txt in tqdm(value)]
            ### Scores
            self.texts_scores["OpenAI"] = {}
            for key, value in self.texts_vecs["OpenAI"]:
                if type(self.target_text) == str:
                    self.texts_scores["OpenAI"][key] = [self.__cos_similarity(self.target_text_vecs["OpenAI"], vec) for vec in tqdm(value)]
                elif type(self.target_text) == list:
                    self.texts_scores["OpenAI"][key] = [self.__cos_similarity(self.target_text_vecs["OpenAI"][i], value[i]) for i in tqdm(range(len(value)))]

        elif type(self.texts) == list:
            self.texts_vecs["OpenAI"] = [self.get_openai_embedding(txt) for txt in self.texts]
            ### Scores
            if type(self.target_text) == str:
                self.texts_scores["OpenAI"] = [self.cos_similarity(self.target_text_vecs["OpenAI"], vec) for vec in tqdm(self.texts_vecs["OpenAI"])]
            elif type(self.target_text) == list:
                self.texts_scores["OpenAI"] = [self.cos_similarity(self.target_text_vecs["OpenAI"][i], self.texts_vecs["OpenAI"][i]) for i in tqdm(range(len(self.texts_vecs["OpenAI"])))]
        
    def scores(self, which:list=None):
        possible = ["OpenAI", "TF-IDF"]
        if which == None:
            return self.texts_scores
        elif which < possible:
            return {k: v for k, v in self.texts_scores.items() if k in which}
        else:
            raise Exception(f"Unexpected keys in 'which'. 'which' must be a subset of {possible}.")
