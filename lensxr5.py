import streamlit as st
import joblib
import pandas as pd
import requests
import streamlit.components.v1 as components
import streamlit.components.v1 as components

spam_model=joblib.load("spam_classifier.pkl")
language_model=joblib.load("lang_det.pkl")
news_model=joblib.load("news_catH.pkl")
review_model=joblib.load("review.pkl")
tv = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("knn_model.pkl")
movies = joblib.load("movies_dataframe.pkl")
# Page config
st.set_page_config(page_title="Multi-Model NLP App", page_icon="🤖", layout="wide")
# bgcolor
# st.markdown(
#     """
#     <style>
#     /* 🌈 Gradient background for the whole app */
#     .stApp {
#         background: linear-gradient(135deg, #e0f7fa, #e1bee7);
#         background-attachment: fixed;
#         background-size: cover;
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

    /* Make all text in the sidebar white */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Optional: change selectbox arrow color and border */
    [data-testid="stSidebar"] .stSelectbox, 
    [data-testid="stSidebar"] .stButton>button {
        color: white !important;
        border-color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------- SIDEBAR ------------------
st.sidebar.title("🔍 More to do")
page = st.sidebar.radio("Go to:", ["🏠 Home","📂 About Project", "🙋‍♀️ About Me", "📞 Contact Info"])

st.sidebar.markdown("---")

# ----------------- HOME PAGE ------------------
# if page == "🏠 Home":
#     st.title("LENS eXpert(NLP Suits)")
if page == "🏠 Home":
    headline=st.markdown("""
        <h1 style="font-family:sans-serif;">
            Welcome to <span style="color:#7e57c2;">LENSR</span>.ai e<span style="color:#7e57c2;">X</span>pert👋
        </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .lensr-title {
        background: linear-gradient(to right, #0f4c81, #ffffff);
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
        Where NLP meets Real Life — Movies, Emotions, News,Language & More.
    </div>
""", unsafe_allow_html=True)


    tab1,tab2,tab3,tab4,tab5=st.tabs(["🍽️Food Review Sentiment","🤖 Spam Classifier","🌐Language Detection","📰News Classification","🎥Movie Recommendation System"])
    with tab1:
        st.title("Food Review Sentiment")
        st.markdown("Know what customers really feel about their food — in just one click! 🍽️🔍")
        msg=st.text_input("Enter Review",placeholder="e.g: The ambience was great and the food was delicious!")
        if st.button("Analyze",key='al1'):
            pred=review_model.predict([msg])
            if pred[0]==0:
                st.error("The customer gave Negative review 👎")
            else:
                st.success("The customer gave Positive review 👍")
                
        st.markdown("#### OR")
        uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"],key='u1')

    
        if uploaded_file:
                
            df_review=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        
            pred=review_model.predict(df_review.Msg)
            df_review.index=range(1,df_review.shape[0]+1)
            df_review["Prediction"]=pred
            df_review["Prediction"]=df_review["Prediction"].map({0:'Disliked 👎',1:'Liked 👍'})
            st.dataframe(df_review)
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
            else:
                st.success("This is Not a Spam Msg 👍")
        st.markdown("#### OR")
        uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"],key='u2')
    

        if uploaded_file:
                
            df_spam=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        
            pred=spam_model.predict(df_spam.Msg)
            df_spam.index=range(1,df_spam.shape[0]+1)
            df_spam["Prediction"]=pred
            df_spam["Prediction"]=df_spam["Prediction"].map({0:'Spam Msg 🚫',1:'Not a Spam Msg 👍'})
            st.dataframe(df_spam)
        st.markdown("""
💡 *Disclaimer: This model is trained on historical data and may not always provide accurate analysis. It's intended for educational and exploratory purposes.*
""")

    ## tab3
    with tab3:
        st.title("Language Detection")
        st.markdown("Just paste any sentence, and we’ll tell you what language it’s in! 🧠🌐")
        msg=st.text_input("Enter text", placeholder="e.g: Hello, How are you?")
        if st.button("Prediction",key="b2"):
            pred=language_model.predict([msg])
            st.success(pred[0])
        st.markdown("#### OR")
        uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"],key="u3")

                
        if uploaded_file:
                
            df=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        
            pred=language_model.predict(df.Msg)
            df.index=range(1,df.shape[0]+1)
            df["Prediction"]=pred
            st.dataframe(df)
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
            st.success(pred[0])
        st.markdown("#### OR") 
        uploaded_file = st.file_uploader("Choose a file",type=["csv", "txt"],key="u4")
    
      
        if uploaded_file:
                
            df=pd.read_csv(uploaded_file,header=None,names=['Msg'])
        
            pred=news_model.predict(df.Msg)
            df.index=range(1,df.shape[0]+1)
            df["Prediction"]=pred
            st.dataframe(df)
        st.markdown("""
💡🟡 *Disclaimer: This model is still under training so it may not always provide accurate analysis. It's intended for educational and exploratory purposes.*
""")

    ## tab5
    with tab5:
                # --- Title ---
        st.title("🎬 Movie Recommendation System")
        st.markdown("Can’t decide what to watch next? Let our recommender be your movie buddy. 🍿🎬")

        # --- Input from user ---
        movie_list = ['-- Select your movie --'] + sorted(movies['name'].unique())

        st.markdown("""
            <style>
            /* Change cursor to pointer when hovering over the selectbox */
            div[data-baseweb="select"] > div {
                cursor: pointer;
            }
            </style>
        """, unsafe_allow_html=True)

        selected_movie = st.selectbox(
            "Pick a movie you love!",
            options=sorted(movies['name'].unique()),
            index=None,
            placeholder="Select your movie"
        )

        # --- Recommendation Function ---

        API_KEY = "85a6b972"  

        def fetch_poster(movie_id):
            url = f"http://www.omdbapi.com/?i={movie_id}&apikey={API_KEY}"
            try:
                response = requests.get(url)
                data = response.json()
                poster = data.get("Poster", "")

                # Enhanced validation check
                if not poster or poster.strip() == "" or poster == "N/A" or poster.startswith("http") is False:
                    return "https://via.placeholder.com/300x450?text=No+Image+Available"
                return poster

            except:
                return "https://via.placeholder.com/300x450?text=Error"


        num_recommendations = st.slider("How many movie recommendations do you want?", min_value=2, max_value=6, value=3)

        def recommend(movie_name, n_recommendations):
            movies_reset = movies.reset_index(drop=True)
            vectors = tv.transform(movies_reset['tag'])

            try:
                index = movies_reset[movies_reset['name'] == movie_name].index[0]
            except IndexError:
                return [], []

            distances, indices = model.kneighbors(vectors[index], n_neighbors=n_recommendations + 1)  # +1 to skip itself

            recommended_movies = movies_reset.iloc[indices[0][1:]]  # skip self
            names = recommended_movies['name'].values
            movie_ids = recommended_movies['movie_id'].values

            posters = [fetch_poster(movie_id) for movie_id in movie_ids]

            return names, posters

        # --- Show Recommendations ---
        if st.button("Show Recommendations 🎥"):
            if selected_movie != '-- Select your movie --':
                recommended_movies, recommended_posters = recommend(selected_movie, num_recommendations)

                st.subheader("📽️ Recommended Movies:")
                cols = st.columns(min(3, num_recommendations))
                for i in range(len(recommended_movies)):
                    with cols[i % len(cols)]:
                        st.markdown('<div class="poster-container">', unsafe_allow_html=True)
                        st.image(recommended_posters[i])
                        if "No+Image" in recommended_posters[i] or "Error" in recommended_posters[i]:
                            st.markdown('<p style="color: red; font-weight: bold;">Image not available</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown(f"**{recommended_movies[i]}**")

            else:
                st.warning("⚠️ Please select a movie from the dropdown.")

                st.markdown("""
                💡 *Disclaimer: This model is trained on historical data and may not always provide accurate or relevant recommendations. It's intended for educational and exploratory purposes.*
                """)



# ----------------- ABOUT ME SECTION ------------------
elif page == "🙋‍♀️ About Me":
    st.title("About Me")
    st.markdown("""
    **Hi, I'm Bhumika Sharma!**  
    A passionate data analyst and aspiring data scientist.  
    I love working on real-world NLP problems and building intelligent systems that help users make better decisions.

    **Skills**: Python, SQL, Excel, Power BI, Machine Learning, NLP  
    **Projects**: Banking Automation, IPL Analysis Report, Sentiment Analysis, News Classification, Spam Detection, Language Identification
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

    1. 🎬 **Movie Recommendation Engine** – Suggests similar movies using NLP and content-based filtering.
    2. 🍽️ **Restaurant Review Sentiment Analysis** – Understand customer emotions through their reviews.
    3. ✉️ **Spam Classifier** – Detects spam messages with high accuracy.
    4. 🌐 **Language Detection** – Automatically identifies the language of any text.
    5. 📰 **News Classifier** – Categorizes news headlines into relevant topics like tech, politics, etc.

    ---

    This app demonstrates the real-world potential of **machine learning models** built using **Python**, **Scikit-learn**, and **Streamlit**, showcasing how NLP can be used to solve practical problems in media, communication, and customer experience.

    """)

 
