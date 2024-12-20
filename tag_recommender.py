import json
import time
import math
from typing import List, Dict


forward_index_data = {}
example_index_data = {}
evaluation_data = {}

def load_data():
    # load and store data into dicts
    global forward_index_data, example_index_data, evaluation_data
    
    with open('forward_indexes_full_dataset.jsonl', 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            forward_index_data.update(entry)

    with open('inverted_indexes_full_dataset.jsonl', 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            example_index_data.update(entry)
    with open('evaluation_data.json', 'r', encoding='utf-8') as f:
            entry = json.load(f)
            evaluation_data.update(entry)

def calculate_score(fields: Dict[str, float], query_tag_frequency) -> float:
    #calculate ranking score
    score_dict = {}
    co_occurrence_count = fields.get("co_occurrence_count", 0)
    frequency = fields.get("frequency", 0)
    jaccard_similarity = co_occurrence_count / (frequency + query_tag_frequency)
    score_dict["jaccard_similarity"] = jaccard_similarity
    cos_similarity = fields.get("cos_similarity", 0)
    score_dict["cos_similarity"] = cos_similarity
    tf_idf = fields.get("tf_idf", 0)
    score_dict["tf_idf"] = tf_idf
    final_score = 0.1 * cos_similarity + 15 * jaccard_similarity + 0.05 * math.log(tf_idf)
    score_dict["final_score"] = final_score
    return score_dict


    

def suggest_tags(query: str, limit: int = 5) -> List[Dict[str, float]]:
    #main recommender
    try:
        tags = [tag.strip().lower() for tag in query.split(",")]
        result_list = set()
        query_tag_frequency = 0
        
        for tag in tags:
            related_tags = example_index_data.get(tag, [])
            fields = forward_index_data.get(tag, {})
            query_tag_frequency += fields.get("frequency", 0)
        
        scored_results = []
        for tag_dict in related_tags:
            tag = tag_dict["related_tag"]
            fields = forward_index_data.get(tag, {})
            fields.update(tag_dict)
            if fields:
                score = calculate_score(fields, query_tag_frequency)
                score["tag"] = tag
                scored_results.append(score)
        
        # Sort results by final score and limit the output
        sorted_results = sorted(scored_results, key=lambda x: x["final_score"], reverse=True)[:limit]
        
        # Prepare final output
        output = []
        output_tag_list = []
        for result in sorted_results:
            output.append({
                result["tag"]: {
                    "final_score": result["final_score"],
                    "jaccard_similarity": result["jaccard_similarity"],
                    "cos_similarity": result["cos_similarity"],
                    "tf_idf": result["tf_idf"]
                }
            })
            output_tag_list.append(result["tag"])
        #print(output)
        return {query:output_tag_list}
    except Exception as e:
        print(f"Error occurred: {e}, query may not exist in the indexes")

def evaluation(evaluation_data, result_data):
    #evaluation part: hit_ratio, recall and precision
    hits = 0
    recall = 0
    precision = 0
    total_len = 0
    for key in evaluation_data:
        if result_data[key]:
            hits += any(tag in evaluation_data[key] for tag in result_data[key])
            recall += len(set(result_data[key]).intersection(evaluation_data[key]))/len(evaluation_data[key]) if evaluation_data[key] else 0
            precision += len(set(result_data[key]).intersection(evaluation_data[key])) / len(result_data[key]) if result_data[key] else 0
            total_len += 1
            if total_len < 20:
                print("query: ", key)
                print("evaluation_data: ",evaluation_data[key])
                print("result_data: ",result_data[key])
        else:
            #print(key)
            continue
    print("hit_ratio: ", hits/total_len)
    print("recall: ", recall/total_len)
    print("precision: ", precision/total_len)

if __name__ == "__main__":
    load_data()
    result_data = {}
    for key in evaluation_data:
        #query = "funny"
        limit = 20
    #start_time = time.time() 
        result_data.update(suggest_tags(key, limit))
    evaluation(evaluation_data, result_data)
    #end_time = time.time()  
    #latency = end_time - start_time  
    #print(f"Latency: {latency:.4f} seconds")