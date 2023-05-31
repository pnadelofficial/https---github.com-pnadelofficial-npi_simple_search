import streamlit as st
import search_util

st.title('NPI Data Search Engine')

if 'page_count' not in st.session_state:
    st.session_state['page_count'] = 1

if 'to_see' not in st.session_state:
    st.session_state['to_see'] = 10

if 'start' not in st.session_state:
    st.session_state['start'] = 0

t_ref, w_ref, transcripts, written_statements = search_util.get_data() #, sentiment_task  t_embeddings, w_embeddings,

search = st.text_input('Search for a word or phrase')

# sem_option = st.selectbox(
#     'Why type of search would you like to use?',
#     ('Simple Search', 'Semantic Search')
# )

option = st.selectbox(
    'What documents would you like to search in?',
    ('Transcripts', 'Written Statements')
)

# st.write("Sentiment Analysis is not working (memory issues). Please do not select 'Yes' for the time being. -- PN")
# sa_option = st.selectbox(
#     'Would you like to add sentiment analysis scores?',
#     ('No', 'Yes')
# )

sem_option = 'Simple Search'
sa_option = 'No'

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
    if (sem_option == 'Simple Search') and (option == 'Transcripts'):
        s = search_util.Searcher(simple_corpus='transcripts_index_dir_d', sem_content=transcripts, reference=t_ref, sentiment_task=sentiment_task if sa_option=='Yes' else None)
        to_full_file, to_spec_file = s(search, st.session_state['start'], st.session_state['to_see'])

    elif (sem_option == 'Semantic Search') and (option == 'Transcripts'):
        s = search_util.Searcher(sem_content=transcripts, sem_index=t_embeddings, reference=t_ref, sentiment_task=sentiment_task if sa_option=='Yes' else None)
        to_full_file, to_spec_file =  s(search, st.session_state['start'], st.session_state['to_see'])

    elif (sem_option == 'Simple Search') and (option == 'Written Statements'):
        s = search_util.Searcher(simple_corpus='statements_index_dir_d', sem_content=written_statements, reference=w_ref, sentiment_task=sentiment_task if sa_option=='Yes' else None)
        to_full_file, to_spec_file = s(search, st.session_state['start'], st.session_state['to_see'])

    elif (sem_option == 'Semantic Search') and (option == 'Written Statements'):
        s = search_util.Searcher(sem_content=written_statements, sem_index=w_embeddings, reference=w_ref, sentiment_task=sentiment_task if sa_option=='Yes' else None)
        to_full_file, to_spec_file =  s(search, st.session_state['start'], st.session_state['to_see'])

    st.write(f'Page: {st.session_state.page_count} of {1 + len(to_full_file)//10}')

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
