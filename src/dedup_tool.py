import json
import shutil
import pandas as pd
import numpy as np
import os
import glob
import logging
import random
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer
from utils_tool import load_json_from_file, save_json_data_to_file, safe_mkdir, init_logging
from src.cluster_tool import cluster_texts, select_from_cluster, get_embeddings

logger = logging.getLogger()
random.seed(2024)


def convert_to_str(s):
    if isinstance(s, str):
        return s
    if isinstance(s, dict):
        return json.dumps(s)
    if isinstance(s, List):
        return str(s)
    return s


def concatenate_idea(idea_k, idea_v):
    output = ""
    try:
        output += idea_k + "\n"
        output += "Problem: " + convert_to_str(idea_v["Problem"]) + "\n"
        output += "Existing Methods: " + convert_to_str(idea_v["Existing Methods"]) + "\n"
        output += "Motivation: " + convert_to_str(idea_v["Motivation"]) + "\n"
        output += "Proposed Method: " + convert_to_str(idea_v["Proposed Method"]) + "\n"
        output += "Experiment Plan: " + convert_to_str(idea_v["Experiment Plan"]) + "\n"
    except Exception as e:
        logger.info(f"concatenate_idea | idea_v:{json.dumps(idea_v, ensure_ascii=False, indent=4)}")
        import pdb;pdb.set_trace()
    return output

def get_dedup_info_from_our_ideas_dir(
        our_ideas_dir,
        idea_dedup_result_info_file,
        similarity_threshold=0.8,
        similarity_matrix_file=None,
    ):
    initial_proposal_files = glob.glob(os.path.join(our_ideas_dir, "idea_info_id_*.json"))
    return get_dedup_info_from_initial_proposal_files(
        initial_proposal_files,
        idea_dedup_result_info_file,
        similarity_threshold=similarity_threshold,
        similarity_matrix_file=similarity_matrix_file,
    )

def get_dedup_info_from_initial_proposal_files(
        initial_proposal_files,
        idea_dedup_result_info_file,
        similarity_threshold=0.8,
        similarity_matrix_file=None,
        embeding_url=None
    ):
    """"""
    random.shuffle(initial_proposal_files)
    all_ideas = []
    all_idea_ks = []
    all_idea_vs = []
    all_proposal_files = []
    for proposal_file in initial_proposal_files:
        data = load_json_from_file(proposal_file)
        if "proposal" in data:
            initial_proposal = data['proposal']
        elif "initial_proposal" in data:
            initial_proposal = data['initial_proposal']['proposal']
        else:
            raise NotImplemented
        if initial_proposal is None:
            logger.info(f"initial_proposal is None, please check file:{proposal_file}")
            continue
        if "Problem" in initial_proposal:
            if "Title" in initial_proposal:
                key = initial_proposal["Title"]
                initial_proposal = {key: initial_proposal}
            else:
                continue

        assert "Title" not in initial_proposal
        assert len(initial_proposal) >= 1, f"len(initial_proposal):{len(initial_proposal)}, proposal_file:{proposal_file}"
        idea_k = list(initial_proposal.keys())[0]
        idea_v = initial_proposal[idea_k]
        all_idea_ks.append(idea_k)
        all_idea_vs.append(idea_v)
        # assert "Type" in idea_v, f"file:{proposal_file},v:{idea_v}"
        assert "Problem" in idea_v, f"Problem:file:{proposal_file},v:{idea_v}, proposal_file:{proposal_file}"
        assert "Existing Methods" in idea_v, f"Existing Methods: file:{proposal_file},v:{idea_v}, proposal_file:{proposal_file}"
        assert "Motivation" in idea_v, f"Motivation: file:{proposal_file},v:{idea_v}, proposal_file:{proposal_file}"
        assert "Proposed Method" in idea_v, f"Proposed Method: file:{proposal_file},v:{idea_v}, proposal_file:{proposal_file}"
        assert "Experiment Plan" in idea_v, f"Experiment Plan: file:{proposal_file},v:{idea_v}, proposal_file:{proposal_file}"
        all_ideas.append(concatenate_idea(idea_k, idea_v))
        all_proposal_files.append(proposal_file)

    similarity_matrix_file = idea_dedup_result_info_file.replace(".json", "_similarity_matrix.npy")
    if not os.path.exists(similarity_matrix_file):
        # Use the http interface to get the embedding vector
        embeddings = get_embeddings(all_ideas, embeding_url)
        # Calculate the similarity matrix
        similarity_matrix = cosine_similarity(embeddings, embeddings)
        # Convert the similarity matrix to a numpy array
        ## setting the diagonal to 0
        np.fill_diagonal(similarity_matrix, 0)
        np.save(similarity_matrix_file, similarity_matrix)
        logger.info(f"get_dedup_info_from_our_ideas_dir | similarity_matrix_file save to: {similarity_matrix_file}")
    else:
        similarity_matrix = np.load(similarity_matrix_file)
        logger.info(f"get_dedup_info_from_our_ideas_dir | similarity_matrix_file load from: {similarity_matrix_file}")

    exist_dedup_proposal_files = []
    final_ideas = dict()
    filter_idx = [] ## ideas that should be filtered
    non_duplicate_count = []
    non_duplicate_percentage = []
    filter_file_to_exist_file_map = dict()
    for i in range(len(all_ideas)):
        if i not in filter_idx:
            ## add current idea to filtered_ideas
            final_ideas[all_idea_ks[i]] = all_idea_vs[i]
            exist_dedup_proposal_files.append(all_proposal_files[i])

            ## filter out similar ideas
            for j in range(i+1, len(all_ideas)):
                if j not in filter_idx and similarity_matrix[i][j] > similarity_threshold or all_idea_ks[j] == all_idea_ks[i]:
                    filter_idx.append(j)
                    filter_file_to_exist_file_map[all_proposal_files[j]] = all_proposal_files[i]
        non_duplicate_count.append(len(final_ideas))
        non_duplicate_percentage.append(len(final_ideas) / (i + 1) * 100)

    print ("#final ideas: ", len(final_ideas))
    result_data = {
        # "our_ideas_dir":our_ideas_dir,
        "all_proposal_files":all_proposal_files,
        "number_of_valid_ideas": len(final_ideas),
        "exist_dedup_proposal_files": exist_dedup_proposal_files,
        "filter_file_to_exist_file_map": filter_file_to_exist_file_map,
        "non_duplicate_count": non_duplicate_count,
        "non_duplicate_percentage": non_duplicate_percentage
    }
    return result_data
