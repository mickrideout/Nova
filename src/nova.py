import glob
import os
import logging
import json
import time
import random
from multiprocessing.pool import ThreadPool

# Import from local modules (assuming they are in the same directory or accessible via PYTHONPATH)
from utils_tool import init_logging, load_json_from_file, save_json_data_to_file, safe_mkdir
from args_tool import get_args
from agent.initial_proposal_agent import InitialProposalAgent
from agent.method_decom_agent import MethodDecomAgent
from agent.final_proposal_agent import FinalProposalGenerator
from vo.paper_data import Paper, load_paper_from_dict
from src.plan_and_iteration import IdeaRelatedPaperPlanThenRetrievalAgent
from src.tfidf_similarity import get_topk_similar_texts
from agent.seed_idea_agent import SeedIdeaAgent
from agent.new_idea_iteration_agent import IdeaIterationAgent
from vo.idea_data import IdeaResult, load_SeedIdeaResult_from_dict
from src.iteration_tool import get_plan_retrieval_file_name_from_idea_file_name, \
    get_next_idea_name_from_idea_file_name, \
    get_method_decom_file_from_initial_proposal_file, \
    get_dedup_file_from_initial_proposal_dir, \
    get_final_proposal_with_method_decom_file_from_initial_proposal_file, \
    get_seed_debug_info_file, \
    get_seed_result_info_file, \
    get_seed_idea_files
from src.dedup_tool import get_dedup_info_from_initial_proposal_files
from src.cluster_tool import cluster_texts, select_from_cluster, get_embeddings
from src.academic_search_tool import search_paper

# Initialize logging
init_logging(log_dir="log", log_filename="nova_demo.log")
logger = logging.getLogger(__name__)

# Initialize arguments and agents
args = get_args()
initial_proposal_agent = InitialProposalAgent(args)
methodDecomAgent = MethodDecomAgent(args)
finalProposalGenerator = FinalProposalGenerator(args)
idea_plan_then_search_tool = IdeaRelatedPaperPlanThenRetrievalAgent(args)
seed_idea_agent = SeedIdeaAgent(args)
ideaIterationAgent = IdeaIterationAgent(args)

# Configuration parameters
PROCESSES_NUM = -1  # Number of parallel processes

def get_idea_files(idea_dir) -> list:
    """Retrieves a list of idea files from a directory or a list of files."""
    if isinstance(idea_dir, str) and os.path.isdir(idea_dir):
        return get_seed_idea_files(idea_dir)
    if isinstance(idea_dir, list) and all(os.path.isfile(file) for file in idea_dir):
        return idea_dir
    return []

def idea_iteration(paper: Paper, cur_idea_dir, next_idea_dir: str):
    """Expands all ideas in a directory using the expand_single_idea function."""
    cur_idea_files = get_idea_files(cur_idea_dir)
    logger.info(f"expand_idea | cur_idea_dir:{cur_idea_dir}, cur_idea_files size:{len(cur_idea_files)}")
    safe_mkdir(next_idea_dir)

    if PROCESSES_NUM != -1:
        pool = ThreadPool(processes=PROCESSES_NUM)
        results = [pool.apply_async(single_idea_iteration, (paper, cur_idea_file, next_idea_dir)) for cur_idea_file in
                   cur_idea_files]
        pool.close()
        pool.join()
        [result.get() for result in results]
    else:
        for cur_idea_file in cur_idea_files:
            single_idea_iteration(paper, cur_idea_file, next_idea_dir)
    logger.info(f"expand_idea | finished!")

def extract_new_knowledge_from_plan_retrieval_result(search_plan_result_list):
    cur_idea_search_plan_and_retrieval_result = ""
    for search_plan_result in search_plan_result_list:
        cur_idea_search_plan_and_retrieval_result += "Search_info:" + search_plan_result["search_info"]
        cur_idea_search_plan_and_retrieval_result += "Search_result:" + search_plan_result["top1_document"]
    return cur_idea_search_plan_and_retrieval_result

def single_idea_iteration(paper: Paper, idea_file: str, next_idea_dir: str):
    """Expands a single idea using the IdeaIterationAgent, saves results, and generates new idea files."""
    logger.info(f"expand_single_idea | idea_file:{idea_file} start...")
    idea_expand_info_file = idea_file.replace(".json", "_expand_info.json")

    if os.path.exists(idea_expand_info_file):
        logger.info(
            f"expand_single_idea | idea_file:{idea_file}, idea_expand_info_file:{idea_expand_info_file} exist, skip!")
        return

    idea_info = load_json_from_file(idea_file)
    source = idea_info['source']
    idea_search_key = str(idea_info["keywords"]) + idea_info['idea'] + idea_info["thinking"]
    retrieval_result_info = load_json_from_file(get_plan_retrieval_file_name_from_idea_file_name(idea_file))
    cur_idea_search_plan_and_retrieval_result = extract_new_knowledge_from_plan_retrieval_result(
        retrieval_result_info["search_plan_result_list"])
    next_prompt, next_llm_result, next_idea_list = ideaIterationAgent.run(paper, idea_search_key,
                                                                       cur_idea_search_plan_and_retrieval_result)
    for next_idea in next_idea_list:
        next_idea.source = source  # type: ignore
    idea_expand_info = {
        "idea": idea_info,
        "next_idea_list": [_.to_dict() for _ in next_idea_list],
        "prompt": next_prompt,
        "llm_result": next_llm_result
    }
    save_json_data_to_file(idea_expand_info, idea_expand_info_file)

    for idx, next_idea in enumerate(next_idea_list):
        if isinstance(next_idea, IdeaResult):
            next_idea = next_idea.to_dict()
        next_idea["from"] = idea_file
        save_json_data_to_file(
            next_idea,
            os.path.join(next_idea_dir, os.path.basename(get_next_idea_name_from_idea_file_name(idea_file, idx)))
        )

    logger.info(f"expand_single_idea | idea_file:{idea_file}, idea_expand_info_file:{idea_expand_info_file} finished!")


def plan_search_single(idea_file: str, target_paper_title: str = ""):
    """Plans and performs a search for a single idea file."""
    plan_retrieval_result_file = get_plan_retrieval_file_name_from_idea_file_name(idea_file)
    logger.info(f"plan_search_single | idea_file:{idea_file}, start...")

    if os.path.exists(plan_retrieval_result_file):
        logger.info(f"plan_search_single | exist :{plan_retrieval_result_file}")
        return

    idea_info = load_json_from_file(idea_file)
    idea_search_key = str(idea_info["keywords"]) + idea_info['idea'] + idea_info["thinking"]
    retrieval_result_info = idea_plan_then_search_tool.run(
        idea_search_key, top_k=10, target_paper_title=target_paper_title
    )
    save_json_data_to_file(retrieval_result_info, plan_retrieval_result_file)
    logger.info(f"plan_search_single | save into:{plan_retrieval_result_file}")
    logger.info(f"plan_search_single | idea_file:{idea_file}, done!")

def plan_search(idea_dir, target_paper_title: str = ""):
    """Performs planning and searching for all idea files in a directory."""
    cur_idea_files = get_idea_files(idea_dir)
    logger.info(f"plan_search | cur_idea_files size:{len(cur_idea_files)}")

    if PROCESSES_NUM != -1:
        pool = ThreadPool(processes=PROCESSES_NUM)
        results = [pool.apply_async(plan_search_single, (cur_idea_file, target_paper_title)) for cur_idea_file in
                   cur_idea_files]
        pool.close()
        pool.join()
        [result.get() for result in results]
    else:
        for cur_idea_file in cur_idea_files:
            plan_search_single(cur_idea_file, target_paper_title)
    logger.info(f"plan_search | finished!")

def gen_single_initial_proposal(paper: Paper, idea_file: str, initial_proposal_save_path: str):
    """Generates a single initial proposal based on the given idea file."""
    logger.info(
        f"gen_single_initial_proposal | idea_file:{idea_file}, initial_proposal_save_path:{initial_proposal_save_path}, start...")

    if os.path.exists(initial_proposal_save_path):
        logger.info(
            f"gen_single_initial_proposal | idea_file:{idea_file}, initial_proposal_save_path:{initial_proposal_save_path}, exist skip!")
        return

    idea_info = load_json_from_file(idea_file)
    retrieval_result_info = load_json_from_file(get_plan_retrieval_file_name_from_idea_file_name(idea_file))
    search_plan_result_list = retrieval_result_info.get("search_plan_result_list", [])
    # idea_with_plan_retrieval_info = {
    #     "idea": idea_info['idea'],
    #     "keywords": idea_info['keywords'],
    #     "thinking": idea_info['thinking'],
    #     "retrieval_result_info": retrieval_result_info
    # }
    # initial_proposal = initial_proposal_agent.run(paper=paper, idea_with_plan_retrieval_info=idea_with_plan_retrieval_info)
    initial_proposal = initial_proposal_agent.run(paper=paper, seed_idea=idea_info, search_plan_result_list=search_plan_result_list)
    initial_proposal['from_idea'] = idea_file
    save_json_data_to_file(initial_proposal, initial_proposal_save_path)
    logger.info(
        f"gen_single_initial_proposal | idea_file:{idea_file}, initial_proposal_save_path:{initial_proposal_save_path}, done!")


def gen_initial_proposal_cluster(
        paper: Paper, idea_dir, initial_proposal_dir: str, number_of_initial_proposal: int = 100,
        url: str = "http://0.0.0.0:9918/compute_embedding"
):
    """Generates number_of_initial_proposal initial proposals based on clustered idea files."""
    st = time.time()
    cur_idea_files = get_idea_files(idea_dir)
    idea_list = [load_json_from_file(_) for _ in cur_idea_files]
    idea_list = [_['idea'] + _["thinking"] + _.get('rationale', "") for _ in idea_list]

    embeddings = get_embeddings(idea_list, url)
    cluster_dict = cluster_texts(idea_list, embeddings, k=min(len(idea_list), number_of_initial_proposal))
    cluster_n = len(cluster_dict)

    logger.info(f"gen_initial_proposal_cluster | cur_idea_dir:{idea_dir}, cur_idea_files size:{len(cur_idea_files)}")

    if PROCESSES_NUM != -1:
        # 1. select and conduct plan search
        select_idea_files = []
        meta_info_list = []
        for i in range(number_of_initial_proposal):
            select_results = select_from_cluster(cluster_dict, cluster_index=i % cluster_n, select_all=False)
            idea_text, idx = select_results[0]
            cur_idea_file = cur_idea_files[idx]
            select_idea_files.append(cur_idea_file)
            meta_info_list.append([idea_text, idx, cur_idea_file])
        plan_search(idea_dir=select_idea_files, target_paper_title=paper.title)
        # 2. generate proposal number = number_of_initial_proposal
        pool = ThreadPool(processes=PROCESSES_NUM)
        results = []
        for i in range(len(meta_info_list)):
            idea_text, idx, cur_idea_file = meta_info_list[i]
            initial_proposal_file = os.path.join(initial_proposal_dir, f"initial_proposal_{i}.json")
            logger.info(
                f"gen_initial_proposal_cluster | i:{i}, cluster_n:{cluster_n}, idx:{idx}, cur_idea_fie:{cur_idea_file}, idea_text:{idea_text}")
            result = pool.apply_async(gen_single_initial_proposal, (paper, cur_idea_file, initial_proposal_file))
            results.append(result)
        pool.close()
        pool.join()
        [result.get() for result in results]
    else:
        for i in range(number_of_initial_proposal):
            select_results = select_from_cluster(cluster_dict, cluster_index=i % cluster_n, select_all=False)
            idea_text, idx = select_results[0]
            cur_idea_file = cur_idea_files[idx]
            plan_search_single(idea_file=cur_idea_file, target_paper_title=paper.title)
            initial_proposal_file = os.path.join(initial_proposal_dir, f"initial_proposal_{i}.json")
            gen_single_initial_proposal(paper, cur_idea_file, initial_proposal_file)
    logger.info(f"gen_initial_proposal_cluster | finished! cost:{time.time() - st}")

def run_dedup(initial_proposal_dir: str, embeding_url: str):
    """Runs deduplication on initial proposal files."""
    logger.info(f"dedup | initial_proposal_dir:{initial_proposal_dir} start...")
    dedup_file = get_dedup_file_from_initial_proposal_dir(step_initial_proposal_dir=initial_proposal_dir)
    if os.path.exists(dedup_file):
        logger.info(f"dedup | initial_proposal_dir:{initial_proposal_dir}, dedup_file:{dedup_file} exist, skip!")
        return

    initial_proposal_files = glob.glob(os.path.join(initial_proposal_dir, f"initial_proposal_*.json"))
    dedup_result = get_dedup_info_from_initial_proposal_files(initial_proposal_files=initial_proposal_files,
                                                              idea_dedup_result_info_file=dedup_file,
                                                              embeding_url=embeding_url)
    save_json_data_to_file(dedup_result, dedup_file)
    logger.info(f"dedup | initial_proposal_dir:{initial_proposal_dir}, dedup_file:{dedup_file} done!")


def gen_single_final_proposal(paper: Paper, initial_proposal_file: str, final_proposal_file: str,
                              with_decom_result: bool = True):
    """Generates a single final proposal."""
    logger.info(
        f"gen_single_final_proposal | initial_proposal_file:{initial_proposal_file}, final_proposal_file:{final_proposal_file} start...")
    if os.path.exists(final_proposal_file):
        logger.info(
            f"gen_single_final_proposal | initial_proposal_file:{initial_proposal_file}, final_proposal_file:{final_proposal_file} exist!")
        return

    initial_proposal_result = load_json_from_file(initial_proposal_file)
    initial_proposal = initial_proposal_result['proposal']
    idea_file = initial_proposal_result['from_idea']

    if with_decom_result:
        method_decom_result = methodDecomAgent.run(paper=paper, initial_proposal=initial_proposal)
        method_decom_info = method_decom_result.llm_response
        method_decom_result = method_decom_result.to_dict()
    else:
        method_decom_result = {}
        method_decom_info = ""

    result = finalProposalGenerator.run(paper=paper, initial_proposal=initial_proposal, method_decom_info=method_decom_info)
    data = {
        "from_idea": initial_proposal_result['from_idea'],
        "from_initial_proposal": initial_proposal_file,
        "initial_proposal": initial_proposal,
        "method_decom_result": method_decom_result,
        "final_proposal": result
    }
    save_json_data_to_file(data, final_proposal_file)
    logger.info(
        f"gen_single_final_proposal | initial_proposal_file:{initial_proposal_file}, final_proposal_file:{final_proposal_file} done!")


def gen_final_proposal_batch(paper: Paper, step_initial_proposal_dir: str, step_final_proposal_dir: str,
                             with_decom_result: bool):
    """Generates final proposals in batch."""
    st = time.time()
    dedup_info = load_json_from_file(get_dedup_file_from_initial_proposal_dir(step_initial_proposal_dir))
    logger.info(
        f"step_initial_proposal_dir | step_initial_proposal_dir:{step_initial_proposal_dir}, step_final_proposal_dir:{step_final_proposal_dir}")
    if PROCESSES_NUM != -1:
        pool = ThreadPool(processes=PROCESSES_NUM)
        results = []
        for initial_proposal_file in dedup_info["exist_dedup_proposal_files"]:
            final_proposal_file = get_final_proposal_with_method_decom_file_from_initial_proposal_file(
                step_final_proposal_dir=step_final_proposal_dir, initial_proposal_file=initial_proposal_file
            )
            result = pool.apply_async(gen_single_final_proposal,
                                      (paper, initial_proposal_file, final_proposal_file, with_decom_result))
            results.append(result)
        pool.close()
        pool.join()
        [result.get() for result in results]
    else:
        for initial_proposal_file in dedup_info["exist_dedup_proposal_files"]:
            final_proposal_file = get_final_proposal_with_method_decom_file_from_initial_proposal_file(
                step_final_proposal_dir=step_final_proposal_dir, initial_proposal_file=initial_proposal_file
            )
            gen_single_final_proposal(paper, initial_proposal_file, final_proposal_file,
                                      with_decom_result=with_decom_result)
    logger.info(f"step_initial_proposal_dir | finished! cost:{time.time() - st}")


def valid_seed_idea(seed_idea: dict) -> bool:
    """Validates if a seed idea has necessary fields."""
    return isinstance(seed_idea, dict) and bool(seed_idea.get('keywords', "") and seed_idea.get('idea', ""))


def valid_final_proposal(final_proposal: dict) -> bool:
    """Validates if a final proposal has necessary fields."""
    if not isinstance(final_proposal, dict):
        return False

    Title = final_proposal.get('Title', "")
    Problem = final_proposal.get('Problem Statement', "")
    Motivation = final_proposal.get('Motivation', "")
    Method = final_proposal.get('Proposed Method', "")
    Experiment = final_proposal.get('Step-by-Step Experiment Plan', "")
    return bool(Title and Problem and Motivation and Method and Experiment)


def get_final_idea_result(step_idea_dir_list: str, step_final_proposal_dir: str) -> list:
    """Retrieves and validates final idea results."""
    final_proposal_files = glob.glob(os.path.join(step_final_proposal_dir, "final_*.json"))
    logger.info(f"get_final_idea_result | final_proposal_files size:{len(final_proposal_files)}")
    idea_result = []
    for final_proposal_file in final_proposal_files:
        try:
            final_proposal = load_json_from_file(final_proposal_file)
            seed_idea_file = final_proposal['from_idea']
            seed_idea = load_json_from_file(seed_idea_file)
            final_proposal = final_proposal["final_proposal"]["final_proposal"]
            if 'Experiment Plan' in final_proposal:
                final_proposal['Step-by-Step Experiment Plan'] = final_proposal['Experiment Plan']
            if valid_seed_idea(seed_idea) and valid_final_proposal(final_proposal):
                idea_result.append({
                    "seed_idea": seed_idea,
                    "final_proposal": final_proposal
                })
        except Exception as e:
            logger.info(f"get_final_idea_result | something wrong for final_proposal_file:{final_proposal_file}, e:{e}")
    logger.info(
        f"get_final_idea_result | final_proposal_files size:{len(final_proposal_files)}, idea_result size:{len(idea_result)}")
    return idea_result

def gen_idea_from_target_paper(
        paper: Paper,
        research_trending_info: str,
        high_quality_paper_list: list,
        idea_output_dir: str,
        do_iteration: bool = True,
        do_final_proposal_generation: bool = True,
) -> list:
    """Main function to generate ideas from a target paper, with or without iterations and final proposal generation."""
    safe_mkdir(idea_output_dir)
    st = time.time()

    # Select some articles that are both recent, popular, and closely related to the current paper,
    # and then use this trending knowledge to generate ideas.
    topk_index_list = get_topk_similar_texts(
        target_text=f"title:{paper.title}\nabstract:{paper.abstract}",
        candidate_texts=high_quality_paper_list,
        k=args.RELATED_HIGH_QUALITY_PAPER_NUMBER
    )
    topk_high_quality_paper_list = [high_quality_paper_list[_] for _ in topk_index_list]

    # seed idea generation
    seed_idea_list = seed_idea_agent.gen_seed_idea_if_not_exist(
        paper=paper,
        research_trending_info=research_trending_info,
        topk_high_quality_paper_list=topk_high_quality_paper_list,
        seed_idea_debug_info_file=get_seed_debug_info_file(idea_output_dir),
        seed_idea_file=get_seed_result_info_file(idea_output_dir),
        use_llm_internal_knowledge=args.SEED_IDEA_INSPIRED_BY_LLM_INTERNAL_KNOWLEDGE,
        use_popular_trending=args.SEED_IDEA_INSPIRED_BY_POPULAR_WORK,
        use_science_discovery_theory=args.SEED_IDEA_GUIDED_BY_THEORY_OF_SCIENTIFIC_INNOVATION,
        use_self_refine=args.USE_SELF_REFINE
    )
    seed0_idea_dir = os.path.join(idea_output_dir, f"step_0")
    safe_mkdir(seed0_idea_dir)
    for i, seed_idea_i in enumerate(seed_idea_list):
        save_json_data_to_file(seed_idea_i.to_dict(), os.path.join(seed0_idea_dir, f"seed_idea_{i}.json"))

    # plan-guided idea iteration
    if do_iteration:
        # iteration: step=0: 15 -> 45 seed idea
        seed0_idea_files = get_idea_files(seed0_idea_dir)
        logger.info(f"gen_idea_from_target_paper | seed0_idea_files size:{len(seed0_idea_files)}")
        # TODO: use llm or reward_model to choose idea
        choose_seed0_idea_files = random.sample(seed0_idea_files, k=min(len(seed0_idea_files), args.STEP0_SAMPLE_NUMBER))
        logger.info(f"gen_idea_from_target_paper | choose_seed0_idea_files size:{len(choose_seed0_idea_files)}")
        step1_dir = os.path.join(idea_output_dir, f"step_1")
        safe_mkdir(step1_dir)
        plan_search(idea_dir=choose_seed0_idea_files)
        idea_iteration(paper=paper, cur_idea_dir=choose_seed0_idea_files, next_idea_dir=step1_dir)
        # iteration: step=1: 45 -> 135 seed idea
        step1_seed_idea_files = get_idea_files(step1_dir)
        # TODO: use llm or reward_model to choose idea
        choose_seed1_idea_files = random.sample(step1_seed_idea_files, k=min(len(step1_seed_idea_files), args.STEP1_SAMPLE_NUMBER))
        step2_dir = os.path.join(idea_output_dir, f"step_2")
        safe_mkdir(step2_dir)
        plan_search(idea_dir=choose_seed1_idea_files)
        idea_iteration(paper=paper, cur_idea_dir=choose_seed1_idea_files, next_idea_dir=step2_dir)
        # iteration: step=2: 135 -> 405 seed idea
        step2_seed_idea_files = get_idea_files(step2_dir)
        # TODO: use llm or reward_model to choose idea
        choose_seed2_idea_files = random.sample(step2_seed_idea_files, k=min(len(step2_seed_idea_files), args.STEP2_SAMPLE_NUMBER))
        step3_dir = os.path.join(idea_output_dir, f"step_3")
        safe_mkdir(step3_dir)
        plan_search(idea_dir=choose_seed2_idea_files)
        idea_iteration(paper=paper, cur_idea_dir=choose_seed2_idea_files, next_idea_dir=step3_dir)

        # seed idea -> proposal
        if do_final_proposal_generation:
            step3_seed_idea_files = get_idea_files(step3_dir)
            initial_proposal_dir = os.path.join(idea_output_dir, f"step_3_initial_proposal")
            final_proposal_dir = os.path.join(idea_output_dir, f'step_3_final_proposal')
            safe_mkdir(initial_proposal_dir)
            safe_mkdir(final_proposal_dir)
            # initial proposal generation follow https://github.com/NoviScl/AI-Researcher
            gen_initial_proposal_cluster(
                paper=paper,
                idea_dir=step3_seed_idea_files,
                initial_proposal_dir=initial_proposal_dir,
                number_of_initial_proposal=args.NUMBER_OF_IDEA,
                url=args.EMBEDDING_URL
            )
            # gen_initial_proposal_cluster_acc(step_idea_dir_list, step_initial_proposal_dir, n=args.number_of_idea, url=args.EMBEDDING_URL)
            # dedup inital proposal
            run_dedup(initial_proposal_dir, embeding_url=args.EMBEDDING_URL)
            # gen proposal with decom first
            gen_final_proposal_batch(
                paper, initial_proposal_dir, final_proposal_dir, with_decom_result=True
            )
            logger.info(f"get_final_idea_result start...")
            final_idea_result_list = get_final_idea_result(step3_dir, final_proposal_dir)
            logger.info(f"get_final_idea_result done, size:{len(final_idea_result_list)}, cost:{time.time() - st} s")
            return final_idea_result_list
        else:
            return []
    else:
        return []
