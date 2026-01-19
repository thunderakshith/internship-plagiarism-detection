import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

def check_plagiarism(texts, threshold=0.7):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    results = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] >= threshold:
                results.append({
                    "text_1": i,
                    "text_2": j,
                    "similarity": round(similarity_matrix[i][j], 2),
                    "fuzzy_score": fuzz.ratio(texts[i], texts[j])
                })
    return results

if __name__ == "__main__":
    df = pd.read_csv("data/sample_texts.csv")
    texts = df["text"].tolist()
    matches = check_plagiarism(texts)

    for match in matches:
        print(match)
