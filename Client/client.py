import streamlit as st
import requests

st.set_page_config(page_title="מערכת Naive Bayes", layout="centered")
st.title("🔍 Naive Bayes")
st.write(" מערכת למידת מכונה ")
st.write()
st.markdown("---")


url = "http://127.0.0.1:8000/"

st.header("🤖  חיזוי לפי מודל מוכן")
with st.expander("🔮 העלאת מודל וחיזוי"):
    response = requests.get(f"{url}api/models_info/")
    response = response.json()
    option = st.selectbox("מה מודל ?", [model.get("name") for model in response])

    for model in response:
        if model.get("name") == option:
            data: dict = {"model_name":model.get("name"), "input_data":{}}
            st.write(model["columns_all"])
            for col in model["columns"]:
                value = st.text_input(f"הכנס ערך עבור {col}")
                data["input_data"][col] = value
    if st.button("🔍 הפעל בדיקה"):
        response = requests.post(f"{url}api/prediction", json=data)
        st.write(response.json())



