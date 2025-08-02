import streamlit as st
import joblib
import pandas as pd
import requests
import sqlite3

spam_model=joblib.load("spam_classifier.pkl")
language_model=joblib.load("lang_det2.pkl")
news_model=joblib.load("news_cat2.pkl")
review_model=joblib.load("review.pkl")


# --- DB Setup ---
def init_db():
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comments TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_feedback(name, rating, comments):
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('INSERT INTO feedback (name, rating, comments) VALUES (?, ?, ?)', (name, rating, comments))
    conn.commit()
    conn.close()

# Initialize DB
init_db()
# Page config
st.set_page_config(page_title="Multi-Model NLP App", page_icon="🤖", layout="wide")
#css formatting
# st.markdown(
#     """
#     <style>
#     /* Sidebar background */
#     [data-testid="stSidebar"] {
#         background-color: #000080; 
#     }

#     /* Make all text in the sidebar white */
#     [data-testid="stSidebar"] * {
#         color: white !important;
#     }

#     /* Optional: change selectbox arrow color and border */
#     [data-testid="stSidebar"] .stSelectbox, 
#     [data-testid="stSidebar"] .stButton>button {
#         color: white !important;
#         border-color: white !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #000080; 
    }

    /* Make all text in the sidebar white except buttons */
    [data-testid="stSidebar"] *:not(button) {
        color: white !important;
    }

    /* Optional: change selectbox arrow color and border */
    [data-testid="stSidebar"] .stSelectbox, 
    [data-testid="stSidebar"] .stButton>button {
        color: white !important;
        border-color: white !important;
    }

    /* Custom button color inside feedback-box */
    .feedback-box .stButton>button {
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 8px;
        padding: 0.4em 1.2em;
        border: none;
        font-weight: bold;
    }

    .feedback-box .stButton>button:hover {
        background-color: #ff0000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- SIDEBAR ------------------
st.sidebar.title("🔍 More to do")
page = st.sidebar.radio("Go to:", ["🏠 Home","📂 About Project", "🙋‍♀️ About Me", "📞 Contact Info"])
st.sidebar.markdown("---")
#box

with st.sidebar:
    st.markdown('<div class="feedback-box">', unsafe_allow_html=True)

    st.markdown("### 💬 We'd love your Feedback!")

    name = st.text_input("Your Name", key="sidebar_name")
    rating = st.slider("Rate the App (1 = Bad, 5 = Excellent)", 1, 5, 3, key="sidebar_rating")
    feedback = st.text_area("Your Feedback", key="sidebar_feedback")

    if st.button("Submit Feedback", key="sidebar_submit"):
        if name and feedback:
            insert_feedback(name, rating, feedback)
            st.success("🎉 Thanks for your feedback!")
            st.balloons()
            st.snow()
        else:
            st.warning("Please fill in your name and feedback.")

    st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# ----------------- HOME PAGE ------------------

if page == "🏠 Home":
    headline = st.markdown("""
    <style>
    .headline-container {
        background: linear-gradient(to right, #ede7f6, #e0d8f7);  /* soft lavender gradient */
        padding: 25px;
        border-left: 6px solid #0f4c81;
        border-right: 6px solid #0f4c81;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        margin-bottom: 25px;
        text-align: center;
    }

    .headline-container h1 {
        font-family: 'Segoe UI', sans-serif;
        margin: 0;
        font-size: 2.4em;
        font-weight: 800;
        color: #1e1e2f;
    }
    </style>

    <div class="headline-container">
        <h1>
            Welcome to <span style="color:#7e57c2;">LENS</span>.ai e<span style="color:#7e57c2;">X</span>pert 👋
        </h1>
    </div>
""", unsafe_allow_html=True)
    

    st.markdown("""
    <style>
    .lensr-title {
        background-color: #dbeafe;  /* light blue */
        padding: 15px 20px;
        border-left: 6px solid #0f4c81;
        border-radius: 8px;
        font-size: 28px;
        font-weight: bold;
        color: #0f172a;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>

    <div class="lensr-title">
        Where NLP meets Real Life — Emotions, News, Language & More.
    </div>
""", unsafe_allow_html=True)
# -----------------tabs---------------
    tab1,tab2,tab3,tab4=st.tabs(["🍽️Food Review Sentiment","🤖 Spam Classifier","🌐Language Detection","📰News Classification"])
    with tab1:
        st.title("Food Review Sentiment")
        st.markdown("Know what customers really feel about their food — in just one click! 🍽️🔍")
        msg=st.text_input("Enter Review",placeholder="e.g: The ambience was great and the food was delicious!")
        if st.button("Analyze",key='al1'):
            pred=review_model.predict([msg])
            if pred[0]==0:
                st.error("Negative review 🙁")
                st.image("negative-vote.png", use_column_width=False)
            else:
                st.success("Positive review 😀")
                st.balloons()
                st.image("positive-vote.png", use_column_width=False)


        st.markdown("#### OR")
        uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"],key='u1')

    
        if uploaded_file:
                
            df_review=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        
            pred=review_model.predict(df_review.Msg)
            df_review.index=range(1,df_review.shape[0]+1)
            df_review["Prediction"]=pred
            df_review["Prediction"]=df_review["Prediction"].map({0:'Disliked 👎',1:'Liked 😀'})
            st.dataframe(df_review)

            # Call email result function here
            # email_result_ui(df_review, label="📩 Mail Me the Sentiment Result", default_filename="food_review_result.csv")
            
        st.markdown("""
💡 *Disclaimer: This model is trained on historical data and may not always provide accurate analysis. It's intended for educational and exploratory purposes.*
""")

# spam classifier
    with tab2:
        st.title("Spam Classifier")
        st.markdown("Not all emails are friendly... ☠️ Let our model detect spam before it clutters your inbox!")
        msg=st.text_input("Enter Msg", placeholder="e.g: Click here to receive a free car!")
        if st.button("Classify",key='cl1'):
            pred=spam_model.predict([msg])
            if pred[0]==0:
                st.error("This is a Spam Msg 🚫")
                st.image("spam.png", use_column_width=False)
            else:
                st.success("This is Not a Spam Msg 👍")
                st.image("ham.png",use_column_width=False)
                st.balloons()
        st.markdown("#### OR")
        uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"],key='u2')
    

        if uploaded_file:
                
            df_spam=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        
            pred=spam_model.predict(df_spam.Msg)
            df_spam.index=range(1,df_spam.shape[0]+1)
            df_spam["Prediction"]=pred
            df_spam["Prediction"]=df_spam["Prediction"].map({0:'Spam Msg 🚫',1:'Not a Spam Msg 👍'})
            st.dataframe(df_spam)
                #  email result function here
            #email_result_ui(df_review, label="📩 Mail Me the Spam Classification Result", default_filename="spam_classification_result.csv")
        st.markdown("""
💡 *Disclaimer: This model is trained on historical data and may not always provide accurate analysis. It's intended for educational and exploratory purposes.*
""")

    ## tab3
    with tab3:
        st.title("Language Detection")
        st.markdown("Just paste any sentence, and we’ll tell you what language it’s in! 🧠🌐")
        msg=st.text_input("Enter text", placeholder="e.g: The study of nature is a large, if not the only, part of science.")
        if st.button("Prediction",key="b2"):
            pred=language_model.predict([msg])
            st.success(f"Detected Language: **{pred[0]}**")
            # st.success(pred[0])
        st.markdown("#### OR")
        uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"],key="u3")

                
        if uploaded_file:
                
            df=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        
            pred=language_model.predict(df.Msg)
            df.index=range(1,df.shape[0]+1)
            df["Prediction"]=pred
            st.dataframe(df)
            #  email result function here
            #email_result_ui(df_review, label="📩 Mail Me the Detected Language Result", default_filename="detected_language_result.csv")
        st.markdown("""
💡🟡 *Disclaimer: This model is still under training on data and might not provide accurate analysis. It's intended for educational and exploratory purposes.*
""")

    ## tab4
    with tab4:
        st.title("News Classification")
        st.markdown("📰 Not sure what kind of news you're reading? Let our AI figure it out in seconds! 🧠✨")
        msg=st.text_input("Enter News Headline",placeholder= "e.g: Apple launches new iPhone")
        if st.button("Classify",key="b3"):
            pred=news_model.predict([msg])
            st.success(f"Detected Headline Type: **{pred[0]}**")
        st.markdown("#### OR") 
        uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"],key="u4")
    
      
        if uploaded_file:
                
            df=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        
            pred=news_model.predict(df.Msg)
            df.index=range(1,df.shape[0]+1)
            df["Prediction"]=pred
            st.dataframe(df)
                 # Call email result function here
            #email_result_ui(df_review, label="📩 Mail Me the News Category Result", default_filename="news_category_result.csv")
        st.markdown("""
💡🟡 *Disclaimer: This model is still under training so it may not always provide accurate analysis. It's intended for educational and exploratory purposes.*
""")



# ----------------- ABOUT ME SECTION ------------------
elif page == "🙋‍♀️ About Me":
    st.title("About Me")
    st.markdown("""
    **Hi, I'm Bhumika Sharma!**  
    A passionate data analyst and aspiring data scientist.  
    I love working on real-world NLP problems and building intelligent systems that help users make better decisions.

    **Skills**: Python, SQL, Excel, Power BI, Machine Learning, NLP  
    **Projects**: Banking Automation(Python, Tkinter, SQLite), IPL Analysis Report(Power BI), Superstore Analytics Dashboard(Power BI), LENSeXpert(Scikit-learn,NLP,Streamlit)
    """)

# ----------------- CONTACT SECTION ------------------
elif page == "📞 Contact Info":
    st.title("Contact Information")
    st.markdown("""
    📧 **Email**: bhumikasharma1808@gmail.com  
    💼 **LinkedIn**: [www.linkedin.com/in/098bhumika](www.linkedin.com/in/098bhumika)  
    🐱 **GitHub**: [https://github.com/Bhumika1808](https://github.com/Bhumika1808)
                """)
#----------------- about project------------------

elif page == "📂 About Project":
    st.title("About the Project")
    st.markdown("""
    **LENS eXpert: The Insight Engine** is a powerful, multi-functional **Natural Language Processing (NLP)** and **Recommendation System** platform.

    It brings together several AI models into one seamless experience:

    1. 🍽️ **Restaurant Review Sentiment Analysis** – Understand customer emotions through their reviews.
    2. ✉️ **Spam Classifier** – Detects spam messages with high accuracy.
    3. 🌐 **Language Detection** – Automatically identifies the language of any text.
    4. 📰 **News Classifier** – Categorizes news headlines into relevant topics like tech, politics, etc.

    ---

    This app demonstrates the real-world potential of **machine learning models** built using **Python**, **Scikit-learn**, and **Streamlit**, showcasing how NLP can be used to solve practical problems in media, communication, and customer experience.
    ---
    Last Updated on August 2, 2025

    """)

# st.sidebar.markdown("## Would be happy if you leave a 💬 Feedback")

# name = st.sidebar.text_input("Your Name")
# rating = st.sidebar.slider("Rate the App (1 = Bad, 5 = Excellent)", 1, 5, 3)
# comments = st.sidebar.text_area("Your Feedback")

# if st.sidebar.button("Submit Feedback"):
#     if name.strip() == "":
#         st.sidebar.warning("Please enter your name.")
#     else:
#         insert_feedback(name, rating, comments)
#         st.sidebar.success("Thank you for your feedback! 🙌")

