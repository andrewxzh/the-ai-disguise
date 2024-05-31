import pandas as pd
import re
import spacy
import concurrent.futures
from tqdm import tqdm
import sys
nlp = spacy.load("en_core_web_lg")
from concurrent.futures import ProcessPoolExecutor
import os


def apply_in_parallel(series, func):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Apply func to each element in the series
        results = list(tqdm(executor.map(func, series), total=len(series)))
    return pd.Series(results)  # Return a Series, not a DataFrame

def tok(text):
    sentence_list=[]
    sentences = [sentence.strip() for sentence in text.split('\n') if sentence.strip()]
    for sentence in sentences:
        doc=nlp(sentence)
        for sent in doc.sents:
            words = re.findall(r'\b\w+\b', sent.text.lower())
            pp=[word for word in words if not word.isdigit()]
            #ngrams_list = pp 
            ngrams_list = [] 
            n = [2]
            for i in n: 
                ngrams_list.extend(generate_ngrams(pp, i))
            return ngrams_list
            #if len(ngrams_list)!=0:
              #  sentence_list.append(ngrams_list)
    #return sentence_list

def generate_ngrams(words, n):
    lst = []
    # no space for n-grams 
    if len(words) < n: 
        return lst
    # else create n-grams 
    for i in range(len(words) - n + 1):
        ngram = words[i:i+n]
        if not any(word.isdigit() for word in ngram): 
            lst.append(" ".join(ngram))
    return lst


if __name__ == '__main__':
    for name in ['CS', 'EESS', 'Math', 'Phys', 'Stat']:
        # GIVEN ORIGINAL ALREADY PARSED 1-GRAM INFORMATION CHANGE INTO DIFFERENT N-GRAM VERSION
        df=pd.read_parquet(f"data/training_data/{name}/human_data.parquet")
        df.reset_index(inplace=True,drop=True)
        #df['sentence'] = df['human_sentence'].apply(lambda x: ' '.join(x))
        df['human_sentence']=apply_in_parallel(df['sentence'],tok)
        df.to_parquet(f"data/training_data/{name}/human_data.parquet", index=False)

        df=pd.read_parquet(f"data/training_data/{name}/ai_data.parquet")
        df.reset_index(inplace=True,drop=True)
        #df['sentence'] = df['ai_sentence'].apply(lambda x: ' '.join(x))
        df['ai_sentence']=apply_in_parallel(df['sentence'],tok)
        df.to_parquet(f"data/training_data/{name}/ai_data.parquet")

        for i in [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]:
            df=pd.read_parquet(f"data/validation_data/{name}/ground_truth_alpha_{i}.parquet")
            df.reset_index(inplace=True,drop=True)
            #df['sentence'] = df['inference_sentence'].apply(lambda x: ' '.join(x))
            df['inference_sentence']=apply_in_parallel(df['sentence'],tok)
            df.to_parquet(f"data/validation_data/{name}/ground_truth_alpha_{i}.parquet", index=False)