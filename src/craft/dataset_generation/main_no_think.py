import sys
import os
import time

os.system("nvidia-smi")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

project_path = '/mnt/data/khosro/RAG-Generated-Augmented-Datasets-for-Document-Rank-Boosting-in-Information-Retrieval'
#Create data and ouput folder

sys.path.append(project_path)

from utils.utils import *
from utils.llm_agent_4_no_think import *

from pandas.io.parquet import read_parquet
import torch
import pandas as pd 
import os
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from collections import defaultdict
import torch
from tqdm import tqdm
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.translate.bleu_score import sentence_bleu
from argparse import ArgumentParser
import numpy as np
import random
import torch
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.schema import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document 
from langchain.schema import HumanMessage


def get_processed_pairs():
    try:
        existing_data = pd.read_csv(f'{project_path}/output/phase_1_chunk_{chunk_number}_{text}.csv')
        # Extract unique query_id and doc_id combinations that were already processed
        processed_pairs = set(zip(existing_data['query_id'], existing_data['target_document_rank']))
        return processed_pairs
    except:
        return set()


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ___________________________________________________________________________________________________________________________________________

chunk_number = 4
max_feedback_iteration = 2

num_max_token = 30
top_n_context = 5
n_sent = 5
text = "no_think"
# ___________________________________________________________________________________________________________________________________________

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

checkpoint = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

model_name = checkpoint  
model = CrossEncoder(checkpoint, max_length=512, device=device)

llm_initial = ChatOpenAI(model_name="unsloth/Qwen3-32B-unsloth-bnb-4bit", base_url = "http://192.168.1.8:{}/v1/".format(8000+chunk_number), temperature=0.6)
llm_feedback = ChatOpenAI(model_name="unsloth/Qwen3-32B-unsloth-bnb-4bit", base_url = "http://192.168.1.8:{}/v1/".format(8000+chunk_number),temperature=0.6)

# ___________________________________________________________________________________________________________________________________________

chunk = pd.read_csv(f'{project_path}/data/IDEM_query_documents_chunk_{chunk_number}.tsv', sep='\t')

#598 queries, for each query, 20 documents (10 context and 10 target documents)

candidate_docs = chunk
candidate_docs = (candidate_docs.rename(columns={'qid': 'query_id', 'pid': 'doc_id', 'document': 'doc_content', 'new_rank': 'rank'})
    [['query_id', 'query', 'doc_id', 'doc_content', 'score', 'rank']])

candidate_docs = convert_to_utf8(candidate_docs, ["query", "doc_content"])

# Delete this section after testing
test_candidate_docs = candidate_docs[
    candidate_docs[['query', 'doc_content']].applymap(has_non_ascii).any(axis=1) |
    candidate_docs[['query', 'doc_content']].applymap(has_quotes).any(axis=1)]

# ___________________________________________________________________________________________________________________________________________

#target_query_id_list = candidate_docs['query_id'].unique().tolist()
target_query_id_list = candidate_docs['query_id'].unique().tolist()

#target_query_id_list = target_query_id_list[:2] #FIX [target_query_id_list[0]] 
target_doc_rank_list = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009] #[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009] #FIX
# ___________________________________________________________________________________________________________________________________________

#target_query_id_list = target_query_id_list[:15] #REMOVE

# ___________________________________________________________________________________________________________________________________________

# Resume from saved processed files

# processed_pairs = get_processed_pairs()

# # Then in your main loop
# for target_query_id in target_query_id_list:
#     for target_doc_rank in target_doc_rank_list:
#         # Skip already processed pairs
#         if (target_query_id, target_doc_rank) in processed_pairs:
#             print(f"Skipping already processed pair: {target_query_id}, {target_doc_rank}")
#             continue

# ___________________________________________________________________________________________________________________________________________


count_doc = 0
count_query = 0
start_time = time.time()
df_dataset_final = pd.DataFrame()

################ Start Target Query Loop

for target_query_id in target_query_id_list:

    df_dataset_per_query = pd.DataFrame()
    df_dataset_per_query_with_feedback = pd.DataFrame()
    count_doc += 1
    print("###########################################################################")
    print(f"Count Doc -> {count_doc}")
    print("###########################################################################")
    count_query = 0

    for target_doc_rank in target_doc_rank_list:

        candidate_docs_full_query = candidate_docs[candidate_docs['query_id'] == target_query_id]
        df_dataset_per_query = pd.DataFrame()
        df_dataset_per_query_with_feedback = pd.DataFrame()
        count_query += 1
        print(f"___________________________________________________________")
        print(f"Doc -> {target_query_id} | Query -> {target_doc_rank}")
        print(f"___________________________________________________________")
        print(f"Count Doc -> {count_doc} | Count Query -> {count_query}")
        print(f"___________________________________________________________")

        # ___________________________________________________________________________________________________________________________________________

        # Target Query (query content)
        target_query = create_target_query(candidate_docs_full_query)

        # ___________________________________________________________________________________________________________________________________________

        # Validator Document
        validator_document_id, validator_document = create_validator_document_info(candidate_docs_full_query, target_doc_rank)

        # ___________________________________________________________________________________________________________________________________________


        # Sent Position List for Validator Document (Number of sentences)
        select_sent_postion = generate_sent_position_list(validator_document)

        # ___________________________________________________________________________________________________________________________________________

        # Target Rank
        target_document_rank = create_target_document_rank(candidate_docs_full_query, validator_document_id)

        #___________________________________________________________________________________________________________________________________________

        # Target Context (top 5 documents in crossencoder rank)

        target_context = create_target_context(candidate_docs_full_query, top_n_context)

        candidate_docs_full_query.loc[:, 'doc_context'] = target_context
        candidate_docs_full_query = candidate_docs_full_query[['query_id', 'query', 'doc_id', 'doc_content', 'doc_context', 'score', 'rank']]

            
        #___________________________________________________________________________________________________________________________________________

        # LLM - Create Initial Response with and without sent_position

        count_boosting_sentences = 0


        boosting_sentences, key_phrases_buffer_A, key_phrases_buffer_B = create_initial_llm_response_without_sent_position(llm_initial, target_query, 
                                                                                                                           validator_document, 
                                                                                                                        target_context, n_sent, num_max_token)
        
        count_boosting_sentences = count_boosting_sentences + len(boosting_sentences)
        
        if count_boosting_sentences == 0:
            boosting_sentences, key_phrases_buffer_A, key_phrases_buffer_B = create_initial_llm_response_without_sent_position(llm_initial, target_query, 
                                                                                                                           validator_document, 
                                                                                                                        target_context, n_sent, num_max_token)
            count_boosting_sentences = count_boosting_sentences + len(boosting_sentences)

        print(f"___________________________________________________________")
        print(f"________________________Initial LLM___________________________________| Count Doc -> {count_doc} | Count Query -> {count_query}")
        print(f"___________________________________________________________")
        print(f"Doc -> {target_query_id} | Query -> {target_doc_rank}")
        print(f"{count_boosting_sentences} sentences has been generated!| Count Doc -> {count_doc} | Count Query -> {count_query}")
        print(f"___________________________________________________________")
        print(f"{count_boosting_sentences} Rerank!")


        #sent_position = 'at the beginning'
        for sent_position in select_sent_postion:
            candidate_docs_full_query_loop = candidate_docs_full_query    
            df_dataset_per_query = create_per_query_dataset(df_dataset_per_query, validator_document_id, validator_document, target_document_rank, 
                                                            model, boosting_sentences, key_phrases_buffer_A, key_phrases_buffer_B, 
                                                            candidate_docs_full_query_loop, sent_position, target_context)

        #___________________________________________________________________________________________________________________________________________

        # Check if we do not have rank 1 in per query dataset -> second LLM Agent

        df_dataset_per_query_with_feedback = df_dataset_per_query

        feedback_counter = 0

        for sent_position in select_sent_postion: # sent_position = 'at the beginning'
            feedback_counter = 0
            if not  dataset_per_query_has_rank_below_n(df_dataset_per_query_with_feedback): # does not have rank 10 for this query
                    while not dataset_per_query_has_rank_below_n_with_sent_position(df_dataset_per_query_with_feedback, sent_position): #does not have rank 10 for this sent_position for this query
                                feedback_counter += 1
                                if feedback_counter > max_feedback_iteration:
                                    print(f"___________________________________________________________")
                                    print(f"Maximum retries reached for sent_position -> {sent_position}| Count Doc -> {count_doc} | Count Query -> {count_query}")
                                    print(f"___________________________________________________________")
                                    print(f"Doc -> {target_query_id} | Query -> {target_doc_rank}")
                                    print(f"Breaking out of the loop. Could not find rank below 10 for -> {sent_position} | Count Doc -> {count_doc} | Count Query -> {count_query}")
                                    print(f"___________________________________________________________")
                                    break  # Exit the loop after maximum retries

                                print(f"___________________________________________________________")
                                print(f"Does not have Rank 1 for sent_position -> {sent_position}| Count Doc -> {count_doc} | Count Query -> {count_query}")
                                print(f"___________________________________________________________")
                                print(f"Attempt {feedback_counter} for -> {sent_position}| Count Doc -> {count_doc} | Count Query -> {count_query}")
                                print(f"___________________________________________________________")
                                print(f"Doc -> {target_query_id} | Query -> {target_doc_rank}")
                                already_generated_new_sentences_separated = feedback_generated_sentences_per_query_rank_below_10_separated_with_sent_position(df_dataset_per_query_with_feedback, 
                                                                                                                                                            sent_position, 100) 

                                improved_sentences = feedback_llm_without_sent_position(llm_feedback, 
                                                            target_query, 
                                                            validator_document, 
                                                            target_context, 
                                                            n_sent, 
                                                            already_generated_new_sentences_separated, 
                                                            key_phrases_buffer_A, 
                                                            key_phrases_buffer_B,
                                                            num_max_token)

                                candidate_docs_full_query_loop = candidate_docs_full_query  
                                df_dataset_per_query_with_feedback = create_per_query_dataset(df_dataset_per_query_with_feedback, validator_document_id, validator_document, 
                                                                                              target_document_rank, 
                                                                                            model, improved_sentences, key_phrases_buffer_A, key_phrases_buffer_B, 
                                                                                            candidate_docs_full_query_loop, sent_position, target_context)
                                #print(f"___________________________________________________________")
                                #print(f"Remove high rank Documents")
                                df_dataset_per_query_with_feedback = remove_highest_new_rank_rows(df_dataset_per_query_with_feedback, sent_position, n_sent)
                                df_dataset_per_query_with_feedback = df_dataset_per_query_with_feedback.reset_index(drop=True)

                    if dataset_per_query_has_rank_below_n_with_sent_position(df_dataset_per_query_with_feedback, sent_position): #does have rank 1
                        print(f"___________________________________________________________")
                        print(f"Rank below 10 Found for sent_position -> {sent_position}| Count Doc -> {count_doc} | Count Query -> {count_query}")
                        print(f"___________________________________________________________")
                        print(f"Doc -> {target_query_id} | Query -> {target_doc_rank}")



        # ___________________________________________________________________________________________________________________________________________

        df_dataset_per_query_with_feedback_with_score = df_dataset_per_query_with_feedback.copy()

        # ___________________________________________________________________________________________________________________________________________

        df_dataset_final = pd.concat([df_dataset_final, df_dataset_per_query_with_feedback_with_score], ignore_index=True)
        df_dataset_final = add_best_query_doc_columns(df_dataset_final)
        df_dataset_final.to_csv(f'{project_path}/output/phase_1_chunk_{chunk_number}_{text}.csv', index=False)
        print("###########################################################################")

################ End of Target Query Loop

# ___________________________________________________________________________________________________________________________________________

df_dataset_final.to_csv(f'{project_path}/output/phase_2_chunk_{chunk_number}_{text}.csv', index=False)

end_time = time.time()
duration = end_time - start_time

print(duration) 

# ___________________________________________________________________________________________________________________________________________