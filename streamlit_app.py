import streamlit as st
from language_detection import load_dataset, train_model

@st.cache_data(show_spinner=False)
def get_model():
    data = load_dataset()
    return train_model(data)

st.title("Language Detection")

text = st.text_area("Enter text", "")
if st.button("Detect"):
    if text.strip():
        model = get_model()
        pred = model.predict([text])[0]
        st.write("Detected Language:", pred)
    else:
        st.warning("Please enter some text.")

