import streamlit as st
import pandas as pd
import re
from string import punctuation
from whoosh.fields import Schema, TEXT, ID
from whoosh import index
import os, os.path
from whoosh import index
from whoosh import qparser
from whoosh.qparser import QueryParser
import gdown
from txtai.embeddings import Embeddings

st.title('NPI Data Simple Search')

if 'page_count' not in st.session_state:
    st.session_state['page_count'] = 1

if 'to_see' not in st.session_state:
    st.session_state['to_see'] = 10

if 'start' not in st.session_state:
    st.session_state['start'] = 0

@st.cache(allow_output_mutation=True)
def get_data():
    # sort by individual's transcript
    statements_url = 'https://drive.google.com/drive/folders/1QwLTtOuKq315_GikEPJRkvZrTIVN6WsC?usp=share_link'
    transcript_url = 'https://drive.google.com/drive/folders/1iWUCIiFLt8coLwRFzz0MCJwcoANWrk5p?usp=sharing' 
    gdown.download_folder(statements_url, quiet=True, use_cookies=False)
    gdown.download_folder(transcript_url, quiet=True, use_cookies=False)

    transcript_sem_search_url = 'https://drive.google.com/drive/folders/1ENPVSbBmc4_wOvU5GSKcYguJv0W8moDF?usp=sharing'
    gdown.download_folder(transcript_sem_search_url, quiet=True, use_cookies=False)
    embeddings = Embeddings()
    embeddings.load("npi_transcript_sem_search_index")

    transcripts = pd.read_csv('all_transcripts.csv').rename(columns={'index':'org_index'}).dropna()
    ref = pd.read_csv('reference.csv')
    ws_ref = pd.read_csv('ws_reference.csv')
    return ref, ws_ref, embeddings, transcripts
reference, ws_reference, embeddings, transcripts = get_data()

def escape_markdown(text):
    MD_SPECIAL_CHARS = "\`*_{}#+"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, '').replace('\t', '')
    return text

def no_punct(word):
    return ''.join([letter for letter in word if letter not in punctuation.replace('-', '')])

def inject_highlights(text, searches):
    inject = f"""
        <p>
        {' '.join([f"<span style='background-color:#fdd835'>{word}</span>" if no_punct(word.lower()) in searches else word for word in text.split()])}
        </p>
        """ 
    return inject   

def display_text(result, query, transcript=True, sem_search=False, index=None):
    if transcript:
        text = escape_markdown(result['content']) if not sem_search else transcripts.q_a[result[0]]
        q_a = re.split('(?=ANSWER:)', text)
        question = q_a[0]
        answer = q_a[1]
        org_id = int(result['org']) if not sem_search else transcripts.org_index[result[0]]

        if not sem_search:
            searches = re.split('AND|OR|NOT', query)
        else:
            searches = query.split()
        searches = [search.strip().lower() for search in searches]  

        context = 0
        if st.button('Increase context size', key=index):
            context += 1
            print(context)

        if (sem_search) and (context > 0):
            before, after = [], [] 
            for i in range(context+1):
                if i != 0:
                    if transcripts.org_index[result[0]-1] == transcripts.org_index[result[0]]:
                        before.append(transcripts.q_a[result[0]-1])
                    if transcripts.org_index[result[0]+1] == transcripts.org_index[result[0]]:
                        after.append(transcripts.q_a[result[0]+1] )  
            before = inject_highlights('\n'.join(before), searches)
            after  = inject_highlights('\n'.join(after ), searches)
            before = '<br><br>'.join(re.split('(?=ANSWER:)',before))
            after  = '<br><br>'.join(re.split('(?=ANSWER:)',after))

        question = inject_highlights(question, searches)
        answer = inject_highlights(answer, searches)  

        if (sem_search) and (context > 0):
            question = '<br>'.join([before, question])
            answer = '<br>'.join([answer, after])          
                                
        st.markdown(f'<small><b>{reference.iloc[org_id].filename}</b></small>',unsafe_allow_html=True)

        if sem_search:
            st.markdown(f'<small>Simialrity Score: <b>{round(result[1],3)}</b></small>',unsafe_allow_html=True)
        st.markdown(question,unsafe_allow_html=True)
        st.markdown('<br>',unsafe_allow_html=True)
        st.markdown(answer,unsafe_allow_html=True)
        st.markdown("<hr style='width: 75%;margin: auto;'>",unsafe_allow_html=True)

    else:
        text = escape_markdown(result['content'])
        org_id = int(result['org'])

        searches = re.split('AND|OR|NOT', query)
        searches = [search.strip() for search in searches]

        inject = inject_highlights(text, searches)
        
        st.markdown(f'<small><b>{ws_reference.iloc[org_id].filename}</b></small>',unsafe_allow_html=True)
        st.markdown(inject, unsafe_allow_html=True)
        st.markdown("<hr style='width: 75%;margin: auto;'>",unsafe_allow_html=True)


def index_search(dirname, search_fields, search_query, sem_option, start, en):    
    if sem_option == 'Simple Search':
        ix = index.open_dir(dirname)
        schema = ix.schema
        
        og = qparser.OrGroup.factory(0.9)
        mp = qparser.MultifieldParser(search_fields, schema, group = og)

        q = mp.parse(search_query)
        
        with ix.searcher() as s:
            if dirname == 'transcripts_index_dir':
                results = s.search(q,limit=None)
                to_full_file = [(f"{reference.iloc[int(res['org'])].filename}", res['content']) for res in results]

                to_page = results[start:en]
                to_spec_file = [(f"{reference.iloc[int(res['org'])].filename}", res['content']) for res in to_page]

                for result in to_page:
                    display_text(result, search_query)
            else:
                results = s.search(q,limit=None)
                to_full_file = [(f"{ws_reference.iloc[int(res['org'])].filename}", res['content']) for res in results]

                to_page = results[s:en]
                to_spec_file = [(f"{ws_reference.iloc[int(res['org'])].filename}", res['content']) for res in to_page]

                for result in to_page:
                    display_text(result, search_query, transcript=False)

            return to_full_file, to_spec_file
    else:
        results_to_see = st.number_input(value=10, label='Choose the amount of results you want to see')
        uids = embeddings.search(search_query, results_to_see)
        to_full_file = [(f"{reference.iloc[transcripts.org_index[uid[0]]].filename}", transcripts.q_a[uid[0]]) for uid in uids]

        to_page = uids[start:en]
        to_spec_file = [(f"{reference.iloc[transcripts.org_index[uid[0]]].filename}", transcripts.q_a[uid[0]]) for uid in uids]
       
        for i,tup in enumerate(to_page):
            display_text(tup, search_query, sem_search=True, index=i)  

        return to_full_file, to_spec_file 

search = st.text_input('Search for a word or phrase')

sem_option = st.selectbox(
    'Why type of search would you like to use?',
    ('Simple Search', 'Semantic Search')
)

option = st.selectbox(
    'What documents would you like to search in?',
    ('Transcripts', 'Written Statements')
)

if sem_option == 'Simple Search':
    if option == 'Transcripts':
        dirname = 'transcripts_index_dir'
    else:
        dirname = 'statements_index_dir'
else:
    dirname = "npi_transcript_sem_search_index"

with st.sidebar:
    if st.button('See next ten', key='next'):
        st.session_state.start = st.session_state.start + 10
        st.session_state.to_see = st.session_state.to_see + 10
        st.session_state.page_count += 1

    if st.button('See previous ten', key='prev'):
        st.session_state.to_see = st.session_state.to_see - 10
        st.session_state.start = st.session_state.start - 10
        st.session_state.page_count -= 1

if search != '':
    to_full_file, to_spec_file = index_search(dirname, ['content'], search, sem_option, st.session_state.start, st.session_state.to_see)

    st.write(f'Page: {st.session_state.page_count} of {1+ len(to_full_file)//10}')

    st.download_button(
        label = 'Download data from this search as a TXT file',
        data = ''.join([f'\n--{ref}--\n{doc}' for ref, doc in to_full_file]).encode('utf-8'),
        file_name = f'npi_data_excerpt_{search}.txt',
        mime = 'text/csv'
    )

    st.download_button(
        label = 'Download data just on page this as a TXT file',
        data = ''.join([f'\n--Result {ref}--\n{doc}' for ref, doc in to_spec_file]).encode('utf-8'),
        file_name = f'npi_data_excerpt_{search}.txt',
        mime = 'text/csv'
    )
