import streamlit as st
import pandas as pd
from Model.Model_System import ModelSystem

st.set_page_config(page_title="מערכת Naive Bayes", layout="centered")
st.title("🔍 Naive Bayes")
st.write(" מערכת למידת מכונה ")
st.write()
st.markdown("---")


def print_metrics_explanation_streamlit():
    st.markdown("## 📊 הסבר על מדדי הביצועים של המודל")
    st.markdown("---")
    
    st.markdown("### 🔹 דיוק (Accuracy)")
    st.write("""
    אחוז התחזיות הנכונות (גם 'yes' וגם 'no') מתוך כלל הדגימות.
    
    > שימושי במיוחד כשיש איזון בין המקרים החיוביים והשליליים בדאטה.
    """)

    st.markdown("### 🔹 Precision (דיוק תחזיות חיוביות)")
    st.write("""
    מתוך כל המקרים שהמודל חזה כ-'yes' – כמה באמת היו 'yes'.

    > שאלה: מתוך כל מה שניבאתי כחיובי – בכמה צדקתי?
    """)

    st.markdown("### 🔹 Recall (רגישות / כיסוי)")
    st.write("""
    מתוך כל המקרים שבאמת היו 'yes' – כמה הצלחנו לגלות?

    > שאלה: מתוך כל מי שבאמת חיובי – בכמה הצלחתי לזהות?
    """)

    st.markdown("### 🔹 F1 Score (מדד מאוזן בין דיוק לרגישות)")
    st.write("""
    ממוצע הרמוני בין Precision ל-Recall. שימושי במיוחד כשיש חוסר איזון בין התוויות.

    > נותן מדד אחד שמאזן בין דיוק לזיהוי.
    """)

    st.markdown("---")

model_system = ModelSystem()

st.header("🧪 חלק א: ניתוח על קובץ נתונים")
with st.expander("📁 העלאת קובץ נתונים לאימון / בדיקה", expanded=True):
    target_variable = st.text_input("שם עמודת היעד (Target)")
    str_yes = st.text_input("הערך שמייצג 'כן'")
    str_no = st.text_input("הערך שמייצג 'לא'")
    confirmed = st.checkbox("✔️ סיימתי להגדיר משתנה יעד")

    if target_variable and str_yes and str_no and confirmed:
        uploaded_file = st.file_uploader("העלה קובץ CSV", type=["csv"])
        if uploaded_file:
            model_system.upload_data(
                file=uploaded_file,
                target_variable=target_variable,
                str_yes=str_yes,
                str_no=str_no
            )

            if model_system.upload_prepared():
                st.success("✅ קובץ הנתונים נטען בהצלחה!")

                option = st.selectbox("מה ברצונך לבצע?", ["בחר", "🧠 אימון מודל", "🔍 בדיקת מודל"])
                
                if option == "🧠 אימון מודל":
                    name_model = st.text_input("הכנס שם למודל")
                    if st.button("אמן מודל"):
                        model_system.training(name_model)
                        st.success(f"המודל '{name_model}' אומן בהצלחה!")

                elif option == "🔍 בדיקת מודל":
                    st.subheader("📦 העלאת קובץ מודל (JSON)")
                    uploaded_model_file = st.file_uploader("בחר קובץ JSON של מודל", type=["json"], key="test_model_upload")

                    if uploaded_model_file:
                        try:
                            model_system.upload_model(uploaded_model_file)
                            st.success("✅ המודל נטען בהצלחה!")
                            if st.button("🔍 הפעל בדיקה"):
                                results:dict = model_system.testing()
                                st.success("הבדיקה הושלמה בהצלחה!")
                                st.subheader("🔍 תוצאות הבדיקה")
                                st.write(f"nom yes {results["num_yes"]}")
                                st.write(f"nom no {results["num_no"]}")
                                st.write(f"**אחוז הצלחה (Accuracy):** {results["result"]:.2f}")
                                st.write(f"**Precision:** {results["precision"]:.2f}")
                                st.write(f"**Recall:** {results["recall"]:.2f}")
                                st.write(f"**F1 Score:** {results["f1"]:.2f}")

                                # לאחר התוצאות – הסבר מפורט:
                                print_metrics_explanation_streamlit()

                        except Exception as e:
                            st.error(f"שגיאה בטעינת המודל: {e}")
                    else:
                        st.info("נא להעלות קובץ JSON של המודל לביצוע בדיקה.")

            else:
                st.warning("⚠️ יש בעיה בהעלאת הקובץ. בדוק את שמות העמודות והערכים.")

st.markdown("---")

st.header("🤖 חלק ב: חיזוי לפי מודל מוכן")
with st.expander("🔮 העלאת מודל וחיזוי"):
    uploaded_file_model = st.file_uploader("העלה קובץ JSON של מודל", type=["json"])

    if uploaded_file_model:
        model_system.upload_model(uploaded_file_model)
        st.success("📦 המודל נטען בהצלחה!")

        if "columns" in model_system.nmodel:
            values = []
            st.subheader("📌 הכנס ערכים עבור החיזוי:")
            for col in model_system.nmodel["columns"]:
                if col != model_system.nmodel.get("target"):
                    value = st.text_input(f"הכנס ערך עבור {col}")
                    values.append(value)

            if all(values):
                try:
                    values = [float(v) if v.replace('.', '', 1).isdigit() else v for v in values]
                    if st.button("🔮 בצע חיזוי"):
                        result = model_system.prediction(values)
                        st.success("החיזוי בוצע בהצלחה!")
                        st.write(result)
                except Exception as e:
                    st.error(f"שגיאה בעיבוד הערכים: {e}")
            else:
                st.warning("יש למלא ערכים עבור כל המשתנים.")



