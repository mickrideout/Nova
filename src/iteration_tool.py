import os
import glob

def get_plan_retrieval_file_name_from_idea_file_name(idea_file):
    return idea_file.replace(".json", "_plan_retrieval_result.json")

def get_without_plan_but_retrieval_file_name_from_idea_file_name(idea_file):
    return idea_file.replace(".json", "_without_plan_but_retrieval_result.json")

def get_next_idea_name_from_idea_file_name(idea_file, i):
    return idea_file.replace(".json", f"_{i}.json")

def get_method_decom_file_from_initial_proposal_file(initial_proposal_file):
    return initial_proposal_file.replace(".json", "_method_decom.json")


def get_dedup_file_from_initial_proposal_dir(step_initial_proposal_dir):
    return os.path.join(step_initial_proposal_dir, "dedup_info.json")

def get_final_proposal_with_method_decom_file_from_initial_proposal_file(
        initial_proposal_file,
        step_final_proposal_dir
):
    final_proposal_file = os.path.join(
        step_final_proposal_dir,
        os.path.basename(initial_proposal_file).replace("initial", "final")
    )
    return final_proposal_file

def get_seed_debug_info_file(output_dir):
    return os.path.join(output_dir, "seed_debug_info.json" )

def get_seed_result_info_file(output_dir):
    return os.path.join(output_dir, "seed_result_info.json" )

def get_seed_idea_files(idea_dir):
    cur_idea_files = glob.glob(os.path.join(idea_dir, "seed_idea*.json"))
    cur_idea_files = [_ for _ in cur_idea_files if "plan" not in _]
    cur_idea_files = [_ for _ in cur_idea_files if "expand_info" not in _]
    return cur_idea_files