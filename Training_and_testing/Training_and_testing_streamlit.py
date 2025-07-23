import streamlit as st
import pandas as pd
from Model import ModelSystem

st.set_page_config(page_title="מערכת Naive Bayes", layout="centered")
st.title("🔍 Naive Bayes")
st.write(" מערכת למידת מכונה ")
st.write()
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
                    
                if st.checkbox("אימון מודל על הנתונים"):
                    name_model = st.text_input("הכנס שם למודל")
                    if st.button("אמן מודל"):
                        model_dict = model_system.training(name_model)
                        
                        st.success(f"המודל '{name_model}' אומן בהצלחה!")

                        try:
                            model_system.upload_model(model_dict["training_75"])
                            st.success("✅ training by 75")
                            results_all:dict = model_system.testing()
                            st.success("הבדיקה הושלמה בהצלחה!")
                            st.info("🔍 תוצאות הבדיקה")
                            st.write(results_all)
                            TP = results_all["TP"]
                            TN = results_all["TN"]
                            FP = results_all["FP"]
                            FN = results_all["FN"]
                            confusion_matrix_df = pd.DataFrame({"Predicted Positive": [TP, FP],"Predicted Negative": [FN, TN]}, index=["Actual Positive", "Actual Negative"])
                            st.write(confusion_matrix_df)
                            model_system.upload_model(model_dict["training_all"])
                            st.success("✅ training by all")
                            results_75:dict = model_system.testing()
                            st.success("הבדיקה הושלמה בהצלחה!")
                            st.info("🔍 תוצאות הבדיקה")
                            st.write(results_75)
                            TP = results_75["TP"]
                            TN = results_75["TN"]
                            FP = results_75["FP"]
                            FN = results_75["FN"]
                            confusion_matrix_df = pd.DataFrame({"Predicted Positive": [TP, FP],"Predicted Negative": [FN, TN]}, index=["Actual Positive", "Actual Negative"])
                            st.write(confusion_matrix_df)

                        except Exception as e:
                            st.error(f"שגיאה בטעינת המודל: {e}")
                    else:
                        st.info("נא להעלות קובץ JSON של המודל לביצוע בדיקה.")

            else:
                st.warning("⚠️ יש בעיה בהעלאת הקובץ. בדוק את שמות העמודות והערכים.")

st.markdown("---")
