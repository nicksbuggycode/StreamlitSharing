import fastapi
import httpx
import pandas as pd
import streamlit as st
from spacy import displacy
from spacy_streamlit import load_model
from services import qtc_check, renal_check


st.title("PharmDigi")
st.header("Leveraging AI to improve patient safety")
st.text(
    "PharmDigi uses the open source med7 library to detect medication names within text.\n"
    "Use it to quickly answer questions about an entire medication list"
)

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; 
margin-bottom: 2.5rem">{}</div>"""

spacy_model = "en_core_med7_lg"
nlp = load_model(spacy_model)
col_dict = {}
seven_colours = [
    "#e6194B",
    "#3cb44b",
    "#ffe119",
    "#ffd8b1",
    "#f58231",
    "#f032e6",
    "#42d4f4",
]

for label,colour in zip(nlp.pipe_labels["ner"],seven_colours):
    col_dict[label] = colour

seeit = st.checkbox("See how it works")
if seeit:
    samp = st.text_area(
        "check out our text or add your own and submit it with ctrl+enter",
        "The patient will stop sitagliptin 50mg daily and start semaglutide 0.25mg weekly",
    )
    sampdoc = nlp(samp)
    html = displacy.render(
        sampdoc,
        style="ent",
        options={"ents": nlp.pipe_labels["ner"],"colors": col_dict},
    )
    # Newlines seem to mess with the rendering
    html = html.replace("\n"," ")
    st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)
