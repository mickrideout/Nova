from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_topk_similar_texts(target_text, candidate_texts, k):
    if not candidate_texts:
        return []
    # Combine the target text and candidate texts for vectorization
    texts = [target_text] + candidate_texts

    # Create the TF-IDF vectorizer and transform the texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarity between the target text and all candidate texts
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Get the indices of the top k similarities
    topk_indices = cosine_similarities.argsort()[-k:][::-1]

    # Retrieve the top k similar texts
    # topk_texts = [candidate_texts[index] for index in topk_indices]

    return topk_indices


if __name__ == '__main__':
    # Example usage:
    target_title_abstract = "Example abstract and title text"
    candidate_title_abstract_list = ["Candidate text 1", "Candidate text 2", "Candidate text 3", "Candidate text 4"]

    # Get the top 2 similar texts
    topk_indices = get_topk_similar_texts(target_title_abstract, candidate_title_abstract_list, 2)
    topk_title_abstracts = [candidate_title_abstract_list[_] for _ in topk_indices]

    print("Top-k Similar Texts:", topk_title_abstracts)