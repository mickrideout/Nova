import json
from src.academic_search_engine import PaperSearchEngine
from utils_tool import save_pickle_data_to_file


def get_doc(title, abstract):
    """Generate a document string from title and abstract."""
    return f"Title:{title}, Abstract:{abstract}"


def filter_cv_nlp_ai_paper(kaggle_arxiv_data, filter_paper_file):
    """Filter papers based on categories and publication date."""
    desired_cat = ['cs.CV', "cs.CV", "cs.CL"]  # Desired categories
    desired_update_date = '2022-01-01'  # Desired update date threshold
    cnt = 0  # Counter for filtered papers
    doc_list, paper_id_list = [], []  # Lists to store document strings and paper IDs

    with open(filter_paper_file, 'w') as fw:
        with open(kaggle_arxiv_data, 'r') as fr:
            for i, l in enumerate(fr):
                data = l.strip()  # Strip whitespace
                data = json.loads(data)  # Parse JSON data
                categories = data['categories']  # Get categories
                update_date = data['update_date']  # Get update date

                if update_date < desired_update_date:
                    continue  # Skip papers before the desired date

                hit_target_category = False  # Flag to check if paper matches desired categories
                for cat in desired_cat:
                    if cat in categories:
                        hit_target_category = True
                        break  # Break if any desired category is found

                if hit_target_category:
                    id = data['id']  # Get paper ID
                    title = data['title']  # Get paper title
                    abstract = data['abstract']  # Get paper abstract
                    paper_id_list.append(id)  # Append paper ID to list
                    doc_list.append(get_doc(title, abstract))  # Append document string to list

                    tmp = {
                        "id": id,
                        "title": title,
                        "abstract": abstract
                    }
                    s = json.dumps(tmp, ensure_ascii=False)  # Convert to JSON string
                    fw.write(s + '\n')  # Write to output file
                    cnt += 1  # Increment counter

                if i % 10000 == 0:
                    print(f'into:{i}')  # Print progress every 10000 lines

        print("cnt:", cnt)  # Print total count of filtered papers

    return paper_id_list, doc_list  # Return lists of paper IDs and document strings


if __name__ == "__main__":

    # https://www.kaggle.com/datasets/Cornell-University/arxiv
    kaggle_arxiv_data = '/home/mick/data/arxiv/arxiv-metadata-oai-snapshot.json'  # Input file path
    filter_paper_file = 'filter_arxiv-metadata-oai-snapshot.json'  # Output file path

    paper_id_list, doc_list = filter_cv_nlp_ai_paper(kaggle_arxiv_data, filter_paper_file)  # Filter papers
    save_pickle_data_to_file(paper_id_list, "faiss/paper_id_list.pkl")  # Save paper IDs to pickle file
    save_pickle_data_to_file(doc_list, "faiss/doc_list.pkl")  # Save document strings to pickle file

    retriever = PaperSearchEngine()  # Initialize paper search engine

    # First run: Persist embeddings and index
    retriever.save_embeddings(doc_list, "faiss/embeddings.pkl")  # Save embeddings to pickle file
    embeddings = retriever.load_embeddings("faiss/embeddings.pkl")  # Load embeddings from pickle file

    retriever.build_index(embeddings)  # Build index from embeddings
    retriever.save_index("faiss/index.faiss")  # Save index to file
