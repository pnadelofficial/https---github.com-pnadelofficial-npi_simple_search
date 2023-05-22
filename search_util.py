import streamlit as st
import pandas as pd
import re
from string import punctuation
from whoosh import index, qparser
import gdown
from txtai.embeddings import Embeddings
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import subprocess

#subprocess.run(['pip', 'install', '--upgrade', '--no-cache-dir', 'gdown'])

@st.cache_resource
def get_data():
    #simple
    statements_url = 'https://drive.google.com/drive/folders/1kJLVteheTQcONEocW4gWlhgC5EOm_4qP?usp=sharing'
    gdown.download_folder(statements_url, quiet=True, use_cookies=False)
    ## transcripts
    transcript_url = 'https://drive.google.com/drive/folders/1LIl8jfzfrQQD6wkOYpvM0osYSt3dGog_?usp=sharing'
    gdown.download_folder(transcript_url, quiet=True, use_cookies=False)

    #semantic
    model_url = 'https://drive.google.com/drive/folders/1iqutHw9dJGqSnRBgrdvegmWEJ-4najWN?usp=sharing&confirm=t'
    gdown.download_folder(model_url, quiet=True, use_cookies=False)
    ss_model = SentenceTransformer('npi_ft542023_L12')
    ## transcripts
    transcript_sem_search_url = 'https://drive.google.com/drive/folders/1u5YABhLt-wv0iur-WSDR61uc7bhEkVbg?usp=sharing&confirm=t'
    gdown.download_folder(transcript_sem_search_url, quiet=True, use_cookies=False)
    t_embeddings = Embeddings()
    t_embeddings.load("npi_transcript_sem_search_index_d")
    ## statements
    ws_sem_search_url = 'https://drive.google.com/drive/folders/12Ym7Ya1C1mzIlVvvOrv42pE3rV2dsf4j?usp=sharing&confirm=t'
    gdown.download_folder(ws_sem_search_url, quiet=True, use_cookies=False)
    w_embeddings = Embeddings()
    w_embeddings.load("npi_ws_sem_search_index_d")

    #org data
    ## transcripts
    at_url = 'https://drive.google.com/file/d/1Uxqd_SKwwzzPmNdnauDmJvdzZUk2Oo0b/view?usp=sharing&confirm=t'
    t_output = 'all_transcripts_d.csv'
    gdown.download(at_url, t_output, quiet=True, fuzzy=True)
    transcripts = pd.read_csv('all_transcripts_d.csv').rename(columns={'index':'org_index'}).dropna()
    ## statements
    ast_url = 'https://drive.google.com/file/d/1MmKY3P4tUMP6IQkdBy6Eo8yGpnubTjBY/view?usp=sharing&confirm=t'
    w_output = 'all_written_statements_d.csv'
    gdown.download(ast_url, w_output, quiet=True, fuzzy=True)
    written_statements = pd.read_csv('all_written_statements_d.csv').rename(columns={'index':'org_index'}).dropna()

    #reference data
    ## transcripts
    t_ref_url = 'https://drive.google.com/file/d/1qJZR3lI95lrMCPATQ8xzkbL9_tQVmT7d/view?usp=sharing&confirm=t'
    t_ref_output = 'transcript_reference_d.csv'
    gdown.download(t_ref_url, t_ref_output, quiet=True, fuzzy=True)
    t_ref = pd.read_csv('transcript_reference_d.csv')
    ## statements
    w_ref_url = 'https://drive.google.com/file/d/1WYsv6n-2hIlvE7GdziEmmg8ARNfc9VSA/view?usp=sharing&confirm=t'
    w_ref_output = 'ws_reference_d.csv'
    gdown.download(w_ref_url, w_ref_output, quiet=True, fuzzy=True)
    w_ref = pd.read_csv('ws_reference_d.csv')

    # for sentiment analysis
#     model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
#     sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)

    return t_ref, w_ref, t_embeddings, w_embeddings, transcripts, written_statements

class Searcher:
    """
    All purpose search interface for the NPI data
    """
    def __init__(self, simple_corpus=None, sem_index=None, sem_content=None, reference=None, sentiment_task=None) -> None:     
        self.simple_corpus = simple_corpus
        self.sem_index = sem_index
        self.sem_content = sem_content
        self.reference = reference
        self.sentiment_task = sentiment_task

    def escape_markdown(self, text):
        '''Removes characters which have specific meanings in markdown'''
        MD_SPECIAL_CHARS = "\`*_{}#+"
        for char in MD_SPECIAL_CHARS:
            text = text.replace(char, '').replace('\t', '')
        return text

    def no_punct(self, word):
        '''Util for below to remove punctuation'''
        return ''.join([letter for letter in word if letter not in punctuation.replace('-', '')])

    def inject_highlights(self, text, searches):
        '''Highlights words from the search query''' # for sem search eventually want to use https://github.com/neuml/txtai/blob/master/examples/32_Model_explainability.ipynb
        esc = punctuation + '"' + '."' + '..."'
        inject = f"""
            <p>
            {' '.join([f"<span style='background-color:#fdd835'>{word}</span>" if (self.no_punct(word.lower()) in searches) and (word not in esc) else word for word in text.split()])}
            </p>
            """ 
        return inject   

    def analyze_sentiment(self, text, question=True):
        tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}
        score = self.sentiment_task(text, **tokenizer_kwargs)[0]
        if type(score) == list:
            nscore = dict()
            nscore['score'] = sum([s['score'] for s in score])/len(score)
            nscore['label'] = 'positive' if nscore['score'] >= 0 else 'negative'
            score = nscore
        if question:
            st.markdown(f"<small>Question Sentiment Score: <b>{round(score['score'], 3)}</b> {score['label']}</small>",unsafe_allow_html=True)
        else:
            st.markdown(f"<small>Answer Sentiment Score: <b>{round(score['score'], 3)}</b> {score['label']}</small>"  ,unsafe_allow_html=True)

    def show_data(self, org_id, text):
        '''Controls how the text is shown on the app'''
        st.markdown(f'<small><b>{self.reference.iloc[org_id].filename}</b></small>',unsafe_allow_html=True)
        if re.match('ANSWER:', text): 
            q_as = re.split('(?=ANSWER:)',text)
            for q_a in q_as:
                if q_a != q_as[-1]:
                    st.markdown(q_a,unsafe_allow_html=True)
                    st.markdown('<br>',unsafe_allow_html=True)
                else: st.markdown(q_a,unsafe_allow_html=True)
        else:
            st.markdown(text,unsafe_allow_html=True)
        st.markdown("<hr style='width: 75%;margin: auto;'>",unsafe_allow_html=True)
    
    def display_text(self, result, query, i):
        '''Controls vrious possibilities for displaying text as well a s increasing context''' 
        searches = re.split('AND|OR|NOT|\s', query) if self.simple_corpus else query.split()
        searches = [search.strip().lower() for search in searches]  
        text = self.escape_markdown(result['content']) if self.simple_corpus else self.escape_markdown(self.sem_content[self.sem_content.columns[-1]][result[0]])
        cur_id = int(result['path']) if self.simple_corpus else result[0]
        org_id = int(result['org']) if self.simple_corpus else self.sem_content[self.sem_content.columns[1]][result[0]]
        if not self.simple_corpus: st.markdown(f'<small>Simialrity Score: <b>{round(result[1],3)}</b></small>',unsafe_allow_html=True)
        if re.search('ANSWER:', text): 
            q_a = re.split('(?=ANSWER:)', text)
            question = self.inject_highlights(q_a[0], searches)
            answer = self.inject_highlights(q_a[1], searches) 
            text = '<br>'.join([question, answer])
            if self.sentiment_task:
                self.analyze_sentiment(question)
                self.analyze_sentiment(answer, question=False)
        else:
            if self.sentiment_task:
                self.analyze_sentiment(text, question=False)
            text = self.inject_highlights(text, searches)
        
        context = 0
        if st.button('Increase context size', key=i): # figure out more context sizes
            context += 1
        
        if context > 0:
            before, after = [], []
            for i in range(1, context+1):
                if self.sem_content[self.sem_content.columns[1]].iloc[cur_id-i] == org_id:
                    before.append(self.sem_content[self.sem_content.columns[-1]].iloc[cur_id-i])
                if self.sem_content[self.sem_content.columns[1]].iloc[cur_id+i] == org_id:
                    after.append(self.sem_content[self.sem_content.columns[-1]].iloc[cur_id+i])
            before = self.inject_highlights('\n'.join(before), searches)
            after  = self.inject_highlights('\n'.join(after ), searches)
            before = '<br><br>'.join(re.split('(?=ANSWER:)',before))
            after  = '<br><br>'.join(re.split('(?=ANSWER:)',after))
            text = '<br>'.join([before, text, after])

        self.show_data(org_id, text)

    def prep_files(self, results, start, en):
        '''Strips all text nad formats it into a file to be downloaded'''
        if self.simple_corpus:
            to_page = results[start:en]
            to_full_file = [(f"{self.reference.iloc[int(res['org'])].filename}", res['content']) for res in results if int(res['org'])<= len(self.reference)] # shouldnt need this if
            to_spec_file = [(f"{self.reference.iloc[int(res['org'])].filename}", res['content']) for res in to_page if int(res['org'])<= len(self.reference)]
        else:
            to_page = results[start:en]
            to_full_file = [(f"{self.reference.iloc[self.sem_content[self.sem_content.columns[1]][uid[0]]].filename}", self.sem_content[self.sem_content.columns[-1]][uid[0]]) for uid in results]
            to_spec_file = [(f"{self.reference.iloc[self.sem_content[self.sem_content.columns[1]][uid[0]]].filename}", self.sem_content[self.sem_content.columns[-1]][uid[0]]) for uid in to_page]
        return to_full_file, to_spec_file, to_page

    def simple_clean(self, _l):
        '''Cleans up the file ouptput from above'''
        return [(ref, doc.replace('\n', ' ').replace('  ', ' ')) for ref,doc in _l]

    def __call__(self, search_query, start, en):
        '''Function that does search and returns files'''
        if self.simple_corpus:
            ix = index.open_dir(self.simple_corpus)
            schema = ix.schema
            
            og = qparser.OrGroup.factory(0.9)
            mp = qparser.MultifieldParser(['content'], schema, group = og)

            q = mp.parse(search_query)
            with ix.searcher() as s:
                results = s.search(q,limit=None)
                to_full_file, to_spec_file, to_page = self.prep_files(results, start, en)
                for i, result in enumerate((to_page)):
                    self.display_text(result, search_query, i)
        else:
            results_to_see = st.number_input(value=10, label='Choose the amount of results you want to see')
            results = self.sem_index.search(search_query, results_to_see)
            to_full_file, to_spec_file, to_page = self.prep_files(results, start, en)
            for i, result in enumerate((to_page)):
                    self.display_text(result, search_query, i)
        return self.simple_clean(to_full_file), self.simple_clean(to_spec_file)
