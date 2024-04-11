from openai import OpenAI


class Metrics():

    def __init__(self, target_text=None, texts=None) -> None:
        self.openai_key = openai_key
        self.data = None
        if type(texts) not in [NoneType, list, dict]:
            raise Exception("Texts to compare must be provided either by list or dictionary. TODO documentation")
        self.texts = texts
        self.texts_vecs = {}
        if type(target_text) not in [NoneType, str, list]:
            raise Exception("Target text(s) must be provided either by string or list. TODO documentation")
        self.target_text = target_text
        self.target_text_ves = {}

    def set_openai(self, key:str, model:str="text-embedding-3-small"):
        if type(key) is not str:
            raise Exception("OpenAI key must be a string.")
        self.openai_client = OpenAI(api_key = key)
        self.get_openai_embedding = lambda text, model : self.openai_client.embeddings.create(input = [text], model=model).data[0].embedding

        
    def calculate_openai(self):
        if type(self.texts) == dict:
            for key, value in self.texts.items():
