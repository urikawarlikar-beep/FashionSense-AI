import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df = pd.read_csv("styles.csv", on_bad_lines="skip")
df.fillna("", inplace=True)

df["features"] = (
    df["gender"] + " " +
    df["masterCategory"] + " " +
    df["subCategory"] + " " +
    df["articleType"] + " " +
    df["baseColour"] + " " +
    df["season"] + " " +
    df["usage"]
)

tfidf = TfidfVectorizer(stop_words="english")
feature_matrix = tfidf.fit_transform(df["features"])
similarity = cosine_similarity(feature_matrix)

def recommend(product_name):
    index = df[df["productDisplayName"] == product_name].index
    if len(index) == 0:
        return "Product Not Found"
    idx = index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommendations = []
    for i in sim_scores[1:6]:
        recommendations.append(df.iloc[i[0]]["productDisplayName"])
    return recommendations

def personalized(gender, season, usage, color):
    result = df[
        (df["gender"] == gender) &
        (df["season"] == season) &
        (df["usage"] == usage) &
        (df["baseColour"] == color)
    ]
    return result[["productDisplayName", "articleType", "baseColour"]].head(10)

def outfit(gender, season, usage):
    shirt = df[(df["gender"] == gender) & (df["articleType"] == "Shirts")].sample(1)
    jeans = df[(df["gender"] == gender) & (df["articleType"] == "Jeans")].sample(1)
    shoes = df[(df["gender"] == gender) & (df["masterCategory"] == "Footwear")].sample(1)
    return pd.concat([shirt, jeans, shoes])

if __name__ == "__main__":
    sample_product = "Turtle Check Men Navy Blue Shirt"
    print("Recommendations for:", sample_product)
    recs = recommend(sample_product)
    for i, prod in enumerate(recs, 1):
        print(f"{i}. {prod}")

    print("\nPersonalised (Men, Summer, Casual, Blue):")
    print(personalized("Men", "Summer", "Casual", "Blue"))

    print("\nRandom outfit for Men, Summer, Casual:")
    print(outfit("Men", "Summer", "Casual"))

pickle.dump(similarity, open("similarity.pkl", "wb"))
pickle.dump(df, open("fashion_data.pkl", "wb"))