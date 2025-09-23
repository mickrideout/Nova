import glob
import os
import logging
import json
import time
import random
from multiprocessing.pool import ThreadPool

# Import from local modules (assuming they are in the same directory or accessible via PYTHONPATH)
from utils_tool import init_logging, load_json_from_file, save_json_data_to_file, safe_mkdir, normalize_title
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
from src.nova import gen_idea_from_target_paper, args, get_final_idea_result, seed_idea_agent, run_dedup, plan_search, idea_iteration, gen_initial_proposal_cluster,gen_final_proposal_batch, get_idea_files
import logging
logger = logging.getLogger()


def get_related_papers(args, paper_info):
    """use search engine find related papers"""
    related_papers = search_paper(
        query=paper_info['title'] + paper_info['abstract_info'],
        k=50,
        url=f'{args.ACADEMIC_SEARCH_ENGINE_IP_ADDRESS}/paper_search_v2'
    )
    related_paper_titles, related_paper_abstracts = [], []
    for i, related_paper in enumerate(related_papers):
        # filter current paper
        if paper_info['title'][:10] != related_paper['title'][:10]:
            related_paper_titles.append(related_paper['title'])
            related_paper_abstracts.append(related_paper['abstract'])
            if len(related_paper_titles) >= args.RELATED_PAPER_NUMBER:
                break
    logger.info(f"generate_ideas | search paper for find related papers done! "
                f"related_paper_titles size:{len(related_paper_titles)}")
    return related_paper_titles, related_paper_abstracts


def get_paper(args, paper_info):
    logger.info(f"generate_ideas | search paper for find related papers start...")
    related_paper_titles, related_paper_abstracts = get_related_papers(args, paper_info)
    print("related papers size:", len(related_paper_titles))
    paper = Paper(
        title=paper_info['title'],
        abstract=paper_info['abstract_info'],
        related_paper_titles=paper_info.get('related_paper_titles', related_paper_titles),
        related_paper_abstract=paper_info.get('related_paper_abstracts', related_paper_abstracts),
        entities=paper_info.get('entities', None),
        categories=paper_info.get('categories', None),
        id=paper_info.get('id', None)
    )
    return paper

def iterative_research_idea_generator(
        paper: Paper,
        research_trending_info: str,
        high_quality_paper_list: list,
        idea_output_dir: str,
        max_iterations: int = 3,
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

    # 1: seed idea generation
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

    # 2: plan-guided idea iteration (The number of seed ideas triples after each iteration.)
    for current_iteration in range(max_iterations):
        pre_seed_idea_dir = os.path.join(idea_output_dir, f"step_{current_iteration}")
        pre_seed_idea_files = get_idea_files(pre_seed_idea_dir)
        logger.info(f"nova_idea_generate | plan-guided idea iteration, current_iteration:{current_iteration}")
        next_seed_idea_dir = os.path.join(idea_output_dir, f"step_{current_iteration+1}")
        safe_mkdir(next_seed_idea_dir)
        plan_search(idea_dir=pre_seed_idea_files)
        idea_iteration(paper=paper, cur_idea_dir=pre_seed_idea_files, next_idea_dir=next_seed_idea_dir)

    # 3: final proposal generation
    if do_final_proposal_generation:
        final_seed_idea_files = get_idea_files(os.path.join(idea_output_dir, f"step_{max_iterations}"))
        initial_proposal_dir = os.path.join(idea_output_dir, f"step_{max_iterations}_initial_proposal")
        final_proposal_dir = os.path.join(idea_output_dir, f'step_{max_iterations}_final_proposal')
        safe_mkdir(initial_proposal_dir)
        safe_mkdir(final_proposal_dir)
        # initial proposal follow https://github.com/NoviScl/AI-Researcher
        gen_initial_proposal_cluster(
            paper=paper,
            idea_dir=final_seed_idea_files,
            initial_proposal_dir=initial_proposal_dir,
            number_of_initial_proposal=args.NUMBER_OF_IDEA,
            url=args.EMBEDDING_URL
        )
        # dedup inital proposal follow https://github.com/NoviScl/AI-Researcher
        run_dedup(initial_proposal_dir, embeding_url=args.EMBEDDING_URL)
        # gen final proposal follow https://github.com/NoviScl/AI-Researcher
        gen_final_proposal_batch(
            paper, initial_proposal_dir, final_proposal_dir, with_decom_result=True
        )
        logger.info(f"get_final_idea_result start...")
        final_idea_result_list = get_final_idea_result(step3_dir, final_proposal_dir)
        logger.info(f"get_final_idea_result done, size:{len(final_idea_result_list)}, cost:{time.time() - st} s")
        return final_idea_result_list

    return []

def main():
    st = time.time()
    # paper_info = {
    #     "title": "ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models",
    #     "abstract_info": """Scientific Research, vital for improving human life, is hindered by its inherent complexity, slow pace, and the need for specialized experts. To enhance its productivity, we propose a ResearchAgent, a large language model-powered research idea writing agent, which automatically generates problems, methods, and experiment designs while iteratively refining them based on scientific literature. Specifically, starting with a core paper as the primary focus to generate ideas, our ResearchAgent is augmented not only with relevant publications through connecting information over an academic graph but also entities retrieved from an entity-centric knowledge store based on their underlying concepts, mined and shared across numerous papers. In addition, mirroring the human approach to iteratively improving ideas with peer discussions, we leverage multiple ReviewingAgents that provide reviews and feedback iteratively. Further, they are instantiated with human preference-aligned large language models whose criteria for evaluation are derived from actual human judgments. We experimentally validate our ResearchAgent on scientific publications across multiple disciplines, showcasing its effectiveness in generating novel, clear, and valid research ideas based on human and model-based evaluation results."""
    # }
    paper_info = {
        "title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        "abstract_info":"""We explore how generating a chain of thought -- a series of intermediate reasoning steps -- significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain of thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain of thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a 540B-parameter language model with just eight chain of thought exemplars achieves state of the art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier."""
    }
    # mock example
    research_trending_info = """
        Current Research Trends in AI: A Comprehensive Analysis\n\n### Hot Research Directions\n\n1. **Long-Context Language Models (LLMs) and Retrieval-Augmented Generation (RAG)**\n   - **Key Papers**: \"RAG in the Era of Long-Context LLMs\", \"LongCite\", \"MemLong\", \"Improved RAG with Self-Reasoning\", \"LongWriter\", \"EfficientRAG\", \"Enhanced RAG with Long-Context LLMs\", \"GraphReader\"\n   - **Highlights**: \n     - Addressing the challenge of maintaining focus and relevance in long-context LLMs.\n     - Combining RAG mechanisms with long-context capabilities to improve performance in tasks like question answering and citation generation.\n     - Innovations such as order-preserving RAG, external retrievers, and graph-based systems to enhance context handling.\n   - **Cross-Field Applications**: These advancements can be applied in fields requiring extensive document analysis, such as legal research, academic literature review, and medical records analysis.\n\n2. **Strategic Chain-of-Thought (CoT) and Self-Improvement Techniques**\n   - **Key Papers**: \"Strategic Chain-of-Thought\", \"Teaching LLM Agents to Self-Improve\", \"Self-Taught Evaluators\", \"Meta-Rewarding LLMs\", \"SelfGoal\"\n   - **Highlights**: \n     - Incorporating strategic knowledge to guide intermediate reasoning steps.\n     - Iterative self-improvement and self-evaluation to enhance model performance over multiple turns.\n     - Use of self-generated training data to refine judgment and reasoning capabilities.\n   - **Cross-Field Applications**: These methods can be beneficial in educational technologies, autonomous decision-making systems, and any domain requiring iterative problem-solving and learning.\n\n3. **Mixture-of-Experts (MoE) and Multi-Agent Systems**\n   - **Key Papers**: \"OLMoE\", \"Agentic RAG for Time Series Analysis\", \"Mixture-of-Agents\", \"MindSearch\"\n   - **Highlights**: \n     - Leveraging sparse Mixture-of-Experts to optimize model performance and efficiency.\n     - Multi-agent architectures for specialized task handling, such as time series analysis and complex web-information seeking.\n   - **Cross-Field Applications**: These approaches can be utilized in financial forecasting, climate modeling, and complex system simulations where specialized expertise is crucial.\n\n4. **Synthetic Data Generation and Utilization**\n   - **Key Papers**: \"Smaller, Weaker, Yet Better\", \"Scaling Synthetic Data Creation\", \"Improving Retrieval in LLMs through Synthetic Data\", \"Model Collapse on Synthetic Data\"\n   - **Highlights**: \n     - Using weaker models to generate high-quality synthetic data for fine-tuning stronger models.\n     - Addressing the challenges of model collapse due to recursive training on synthetic data.\n   - **Cross-Field Applications**: Synthetic data can be used in privacy-preserving data analysis, training AI models in healthcare, and augmenting datasets in low-resource languages.\n\n5. **Controllable and Robust Text Generation**\n   - **Key Papers**: \"Controllable Text Generation for LLMs\", \"Enhancing Robustness in LLMs\", \"Improving Legibility of LLM Outputs\"\n   - **Highlights**: \n     - Techniques for controlling the style, safety, and consistency of generated text.\n     - Methods to enhance robustness by filtering out irrelevant information and improving the clarity of outputs.\n   - **Cross-Field Applications**: These advancements are crucial for developing reliable AI assistants, automated content generation, and ensuring the safety of AI-generated outputs in sensitive applications.\n\n6. **AI in Scientific Discovery and Evaluation**\n   - **Key Papers**: \"The AI Scientist\", \"Automate Design of Agentic Systems\", \"Self-Taught Evaluators\"\n   - **Highlights**: \n     - AI agents capable of conducting independent research and writing scientific papers.\n     - Meta-agent frameworks for designing and evaluating agentic systems.\n   - **Cross-Field Applications**: These innovations can revolutionize scientific research, enabling faster discovery and validation of new theories across various scientific disciplines.\n\n7. **Advanced Prompt Engineering and Personalization**\n   - **Key Papers**: \"Conversational Prompt Engineering\", \"A Survey of Prompt Engineering Methods in LLMs\"\n   - **Highlights**: \n     - Techniques for creating personalized prompts through iterative user interaction.\n     - Comprehensive surveys on prompt engineering methods for various NLP tasks.\n   - **Cross-Field Applications**: Personalized prompt engineering can enhance user experience in customer service bots, personalized education platforms, and adaptive learning systems.\n\n8. **AI in Code and Software Engineering**\n   - **Key Papers**: \"LLM Compiler\", \"From LLMs to LLM-based Agents for Software Engineering\", \"DeepSeek-Coder-V2\"\n   - **Highlights**: \n     - Models designed for code optimization and generation.\n     - Surveys on the application of LLMs in software engineering tasks like requirement engineering and test generation.\n   - **Cross-Field Applications**: These advancements can improve software development workflows, automate code reviews, and enhance the capabilities of integrated development environments (IDEs).\n\n### Conclusion\n\nThe current hot research trends in AI are characterized by significant advancements in long-context LLMs, strategic reasoning, multi-agent systems, synthetic data generation, controllable text generation, AI-driven scientific discovery, prompt engineering, and AI applications in software engineering. These technologies not only push the boundaries of AI capabilities but also offer promising applications across various fields, from healthcare and finance to education and scientific research. The continuous innovation in these areas is likely to lead to more robust, efficient, and versatile AI systems in the near future.
        """
    paper = get_paper(args, paper_info)
    save_dir_name = normalize_title(paper_info['title'])
    idea_output_dir=f"{args.OUTPUT_DIR}/{save_dir_name}"
    print("idea_output_dir:", idea_output_dir)
    os.makedirs(idea_output_dir, exist_ok=True)
    save_json_data_to_file(
        paper.get_dict_result(),
        idea_output_dir+"/input_paper.json"
    )
    proposal_list = iterative_research_idea_generator(
            paper=paper ,
            research_trending_info=research_trending_info,
            high_quality_paper_list=[],
            idea_output_dir=idea_output_dir,
            max_iterations = 3,
            do_final_proposal_generation = False
        )
    for i, proposal in enumerate(proposal_list):
        print("=" * 20)
        print("i:", i)
        print("proposal:", json.dumps(proposal, ensure_ascii=False, indent=4))
    print(f"done! total time cost:{time.time() - st}")

if __name__ == "__main__":
    main()