from openai import OpenAI
import numpy as np


class Metrics():

    def __init__(self, target_text=None, texts=None) -> None:
        if type(texts) not in [type(None), list, dict]:
            raise Exception("Texts to compare must be provided either by list or dictionary of lists. TODO documentation")
        self.texts = texts
        self.texts_vecs = {}
        self.texts_scores = {}
        if type(target_text) not in [type(None), str, list]:
            raise Exception("Target text(s) must be provided either by string or list. TODO documentation")
        self.target_text = target_text
        # self.pairwise = False if type(target_text)==str else True if type(target_text)==list else None
        self.target_text_vecs = {}
        
        # if openai_key != None:
        #     self.set_openai(openai_key)

    def cos_similarity(vec1, vec2):
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

    def set_openai(self, key:str, model:str="text-embedding-3-small"):
        if type(key) is not str:
            raise Exception("OpenAI key must be a string.")
        self.openai_client = OpenAI(api_key = key)
        self.get_openai_embedding = lambda text: self.openai_client.embeddings.create(input = [text], model=model).data[0].embedding

        
    def calculate_openai(self):
        if type(self.target_text) == str:
            self.target_text_vecs["OpenAI"] = self.get_openai_embedding(self.target_text)
        elif type(self.target_text) == list:
            self.target_text_vecs["OpenAI"] = [self.get_openai_embedding(txt) for txt in self.target_text]

        if type(self.texts) == dict:
            self.texts_vecs["OpenAI"] = {}
            for key, value in self.texts.items():
                self.texts_vecs["OpenAI"][key] = [self.get_openai_embedding(txt) for txt in value]
                # for txt in value:
                #     self.texts_vecs["OpenAI"][key].append(self.get_openai_embedding(txt))
            self.texts_scores["OpenAI"] = {}
            for key, value in self.texts_vecs["OpenAI"]:
                if type(self.target_text) == str:
                    self.texts_scores["OpenAI"][key] = [self.cos_similarity(self.target_text_vecs["OpenAI"], vec) for vec in value]
                elif type(self.target_text) == list:
                    self.texts_scores["OpenAI"][key] = [self.cos_similarity(self.target_text_vecs["OpenAI"][i], value[i]) for i in range(len(value))]

        elif type(self.texts) == list:
            self.texts_vecs["OpenAI"] = [self.get_openai_embedding(txt) for txt in self.texts]
            # for txt in self.texts:
            #     self.texts_vecs["OpenAI"].append(self.get_openai_embedding(txt))
            if type(self.target_text) == str:
                self.texts_scores["OpenAI"] = [self.cos_similarity(self.target_text_vecs["OpenAI"], vec) for vec in self.texts_vecs["OpenAI"]]
            elif type(self.target_text) == list:
                self.texts_scores["OpenAI"] = [self.cos_similarity(self.target_text_vecs["OpenAI"][i], self.texts_vecs["OpenAI"][i]) for i in range(len(self.texts_vecs["OpenAI"]))]                

        else:
            raise Exception("TODO")
        

