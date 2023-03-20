import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import ast

import plotly.io as pio
pio.templates.default = "plotly"

st.set_page_config(layout='wide')

st.title('Testimony Responses from the NPI')

@st.cache(allow_output_mutation=True)
def get_csvs():
    label_df = pd.read_csv('v_npi_labeled.csv')
    label_df['sents_indices'] = label_df['sents_indices'].apply(ast.literal_eval)
    df = pd.read_csv('v_npi_answers.csv')
    return label_df, df
label_df, df = get_csvs()

fig = px.scatter(label_df, x='x', y='y',size='sent_amount',color='kmeans', hover_data=['rep_words', 'label'])
with st.container():
    fig_selected = plotly_events(fig, select_event=True)

def display_text(text, title, date):
    st.write(f'**From**: {title}')
    st.write(f'**Posted on**: {date}')
    st.write(text)
    st.markdown("<hr style='width: 75%;margin: auto;'>",unsafe_allow_html=True)

if st.button('Reset'):
    fig_selected = []

tabs = []
for selected in fig_selected:
    label_subset = label_df.loc[(label_df.x == selected['x']) & (label_df.y == selected['y'])]
    tabs += label_subset.label.to_list()

if len(fig_selected) > 0:
    for tab,name in zip(st.tabs(tabs), tabs):
        with tab:
            label_subset= label_df.loc[label_df.label == name]
            idxs = label_subset.sents_indices.iloc[0]
            st.write(f"<p style='text-align: right;'><b>Selected Label</b>: {label_subset.label.iloc[0].split(': ')[1]} -- {label_subset.rep_words.iloc[0]}</p>",unsafe_allow_html=True)
            st.write(f"<p style='text-align: right;'><b>Number of responses</b>: {len(df.iloc[idxs])}</p>",unsafe_allow_html=True)
            st.markdown('<br>',unsafe_allow_html=True)
            subset = df.iloc[idxs]
            p = subset.apply(lambda x: display_text(x['text'],x['title'],x['date']), axis=1)

            st.download_button(
                label = 'Download data as CSV',
                data = subset.to_csv().encode('utf-8'),
                file_name = 'npi_data_excerpt.csv',
                mime = 'text/csv'
            )
else:
    st.write('Use the select tools in the chart above to select labeled sentences.')

st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)