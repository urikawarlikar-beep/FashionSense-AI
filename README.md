# 👗 FashionSense AI

### AI-Powered Fashion Recommendation & Virtual Stylist

FashionSense AI is an AI-powered fashion recommendation application designed to help users discover suitable outfits based on their personal preferences, style, occasion, budget, and fashion characteristics.

The application combines fashion recommendation techniques with an interactive AI stylist experience to provide personalized fashion suggestions.

---

## ✨ Features

### 👗 Fashion Recommendations

The system recommends fashion products based on user preferences such as:

- Gender
- Age
- Skin tone
- Body type
- Occasion
- Preferred style
- Budget

---

### 🤖 AI Fashion Stylist

The AI Stylist provides personalized fashion advice such as:

- Outfit suggestions
- Color combinations
- Alternative outfits
- Accessories recommendations
- Footwear suggestions
- Styling tips
- Reasons why an outfit may suit the user

---

### 📸 Outfit Analyzer

Users can:

- Upload an outfit image
- Take a webcam snapshot
- Analyze their outfit
- Receive styling suggestions

---

### 🔍 Recommendation Engine

The project uses recommendation techniques including:

- Content-based filtering
- Similarity search
- Personalized recommendations

Fashion products can be recommended based on their characteristics and similarity to the user's preferences.

---

## 🧠 Project Phases

### Phase 1 — Data Understanding & EDA

The fashion dataset is explored and analyzed to understand:

- Product categories
- Fashion attributes
- Gender
- Colors
- Seasons
- Usage
- Product types

---

### Phase 2 — Feature Engineering

Additional AI-friendly features are created for recommendation and personalization.

Examples include:

- `budget_range`
- `style`
- `comfort_level`
- `trend_score`
- `body_type_match`
- `skin_tone_match`
- `ai_styling_tip`
- `recommended_with`
- `accessory_match`
- `footwear_match`

---

### Phase 3 — Recommendation Engine

The recommendation system focuses on:

- Content-based filtering
- Similarity search
- Personalized recommendations

---

### Phase 4 — AI Stylist

The AI Stylist is designed to provide:

- Outfit explanations
- Alternative outfit ideas
- Color combinations
- Fashion advice
- Styling recommendations

---

### Phase 5 — Streamlit UI

FashionSense AI provides an interactive fashion-themed interface with:

- Luxury fashion design
- Product cards
- Fashion gallery
- User style profile
- Recommendation page
- AI Stylist
- Outfit Analyzer

---

### Phase 6 — Outfit Analysis

The application supports:

- Photo upload
- Webcam snapshot
- Outfit analysis
- Virtual stylist suggestions

---

## 🛠️ Technologies Used

### Programming

- Python

### Data Science & Machine Learning

- Pandas
- NumPy
- Scikit-learn

### Recommendation System

- Content-Based Filtering
- TF-IDF
- Cosine Similarity
- Similarity Search

### Computer Vision

- PIL
- Image Processing

### Web Application

- Streamlit

### Development Tools

- Jupyter Notebook
- Visual Studio Code
- Git
- GitHub

---
## 📸 Application Screenshots
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/866c74a4-a11e-4524-9b44-b114b66525f8" />


### 🏠 FashionSense AI Dashboard

The main dashboard allows users to create their style profile by selecting gender, age, skin tone, body type, occasion, preferred style, and budget.

![FashionSense AI Dashboard](screenshots/home.png)

---

### 👗 Fashion Recommendations

Personalized fashion products are displayed based on the user's selected preferences.

![Fashion Recommendations](screenshots/recommendations.png)

---

### 🤖 AI Stylist

The AI Stylist provides personalized outfit suggestions, styling advice, color combinations, and alternative outfit ideas.

![AI Stylist](screenshots/ai_stylist.png)

---

### 📷 Outfit Analyzer

Users can upload an image or use a webcam snapshot to analyze an outfit and receive styling suggestions.

![Outfit Analyzer](screenshots/outfit_analyzer.png)

---

## 📂 Project Structure

```text
FashionSense-AI/
│
├── app.py
├── Recommender.py
├── FashionSense AI.ipynb
│
├── cleaned_styles.csv
├── feature_matrix.npz
├── requirements.txt
│
├── data/
│   └── styles.csv
│
├── .gitignore
└── README.md
