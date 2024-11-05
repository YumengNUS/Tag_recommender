import os
import pandas as pd
import json
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import fasttext as ft
from multiprocessing import Pool
from tqdm import tqdm
import math

def process_chunk_for_frequency(chunk, tag_frequency, doc_tag_count):
    #update tag frequency and document count
    chunk.drop_duplicates(inplace=True)
    chunk['tags'] = chunk['tags'].str.lower().fillna('')
    chunk['root_tags'] = chunk['root_tags'].str.lower().fillna('')

    chunk['tags_split'] = chunk['tags'].str.split(',').apply(lambda x: [tag.strip() for tag in x if tag.strip()])
    chunk['root_tags_split'] = chunk['root_tags'].str.split(',').apply(lambda x: [tag.strip() for tag in x if tag.strip()])
    chunk['merged_tags'] = chunk.apply(lambda row: list(set(row['tags_split'] + row['root_tags_split'])), axis=1)

    for tags in chunk['merged_tags']:
        tag_frequency.update(tags)
        for tag in set(tags):
            doc_tag_count[tag] += 1  

def process_chunk_for_co_occurrence(chunk, co_occurrence_counter, frequent_tags):
    #update co-occurrence counts for frequent tags
    chunk.drop_duplicates(inplace=True)
    chunk['tags'] = chunk['tags'].str.lower().fillna('')
    chunk['root_tags'] = chunk['root_tags'].str.lower().fillna('')

    chunk['tags_split'] = chunk['tags'].str.split(',').apply(lambda x: [tag.strip() for tag in x if tag.strip()])
    chunk['root_tags_split'] = chunk['root_tags'].str.split(',').apply(lambda x: [tag.strip() for tag in x if tag.strip()])
    chunk['merged_tags'] = chunk.apply(lambda row: list(set(row['tags_split'] + row['root_tags_split'])), axis=1)

    for tags in chunk['merged_tags']:
        filtered_tags = [tag for tag in tags if tag in frequent_tags]
        for pair in combinations(set(filtered_tags), 2):
            sorted_pair = tuple(sorted(pair))
            co_occurrence_counter[sorted_pair] += 1

def calculate_tf_idf(tag, frequency, doc_tag_count, total_docs):
    #TF-IDF score
    tf = frequency
    idf = math.log(total_docs / (doc_tag_count[tag] + 1))
    return tf * idf

def calculate_cosine_similarity(args):
    tag, related_tag, tag_embedding, related_tag_embedding, count = args
    cos_sim = cosine_similarity([tag_embedding], [related_tag_embedding])[0][0]
    return {
        "tag": tag,
        "related_tag": related_tag,
        "co_occurrence_count": count,
        "cos_similarity": cos_sim
    }

def process_raw_data(input_file, output_file, forward_output_file, model_path, chunk_size=2000):
    # Main processing function
    model = ft.load_model(model_path)
    
    tag_frequency = Counter()
    doc_tag_count = defaultdict(int)
    doc_count = 0

    total_rows = sum(1 for _ in open(input_file, 'r', encoding='utf-8')) - 1
    chunk_progress = tqdm(total=total_rows, desc="Counting Tag Frequency", unit="rows")
    #1: Count all tags' frequencies
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        process_chunk_for_frequency(chunk, tag_frequency, doc_tag_count)
        doc_count += len(chunk)
        chunk_progress.update(len(chunk))

    chunk_progress.close()
    #Filter tags that appear less than 10 times
    frequent_tags = {tag for tag, freq in tag_frequency.items() if freq > 10}

    #2: calculate co-occurrence for frequent tags
    co_occurrence_counter = Counter()
    chunk_progress = tqdm(total=total_rows, desc="Counting Co-occurrence for Frequent Tags", unit="rows")

    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        process_chunk_for_co_occurrence(chunk, co_occurrence_counter, frequent_tags)
        chunk_progress.update(len(chunk))

    chunk_progress.close()

    #forward fields
    tag_forward = {}
    for tag in tqdm(frequent_tags):
        tag_forward[tag] = {
            "embedding": model.get_word_vector(tag).tolist(),
            "frequency": tag_frequency[tag],
            "tf_idf": calculate_tf_idf(tag, tag_frequency[tag], doc_tag_count, doc_count)
        }

    #turn co_occurrence_counter into dicts to reduce the processing time
    related_tags_dict = defaultdict(list)
    for (tag1, tag2), count in tqdm(co_occurrence_counter.items()):
        if tag1 in frequent_tags and tag2 in frequent_tags:
            related_tags_dict[tag1].append((tag2, count))
            related_tags_dict[tag2].append((tag1, count))
    
    related_tags_inputs = []
    for tag in tqdm(frequent_tags):
        related_tags = sorted(related_tags_dict[tag], key=lambda x: x[1], reverse=True)[:10]
        tag_embedding = tag_forward[tag]["embedding"]

        for related_tag, count in related_tags:
            related_tag_embedding = tag_forward[related_tag]["embedding"]
            related_tags_inputs.append((tag, related_tag, tag_embedding, related_tag_embedding, count))

    max_cpu_count = os.cpu_count()
    print(os.cpu_count())
    with Pool(processes=max_cpu_count) as pool:
        results = list(tqdm(pool.imap(calculate_cosine_similarity, related_tags_inputs, chunksize=10), 
                            total=len(related_tags_inputs), desc="Calculating Cosine Similarities"))

    tag_relationships = {}
    for result in results:
        tag = result["tag"]
        if tag not in tag_relationships:
            tag_relationships[tag] = []
        tag_relationships[tag].append(result)
    
    #save data
    with open(output_file, 'w') as json_file:
        json.dump(tag_relationships, json_file)

    tag_forward_filtered = {tag: {"frequency": value["frequency"], "tf_idf": value["tf_idf"]}
                            for tag, value in tag_forward.items()}
    with open(forward_output_file, 'w') as embed_file:
        json.dump(tag_forward_filtered, embed_file)

if __name__ == "__main__":
    input_file_path = 'full_dataset.csv'
    output_file_path = 'inverted_indexes_full_dataset.jsonl'
    forward_output_file = 'forward_indexes_full_dataset.jsonl'
    fasttext_model_path = './model/crawl-300d-2M-subword.bin'
    process_raw_data(input_file_path, output_file_path, forward_output_file, fasttext_model_path)