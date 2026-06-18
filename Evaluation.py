import os
import math
import pandas as pd
from scipy import stats

class DocRelScore:

    def __init__(self):
        self.topics = {}

    def add_doc(self, topic_id, doc_id, relevance_score):
        if topic_id not in self.topics:
            self.topics[topic_id] = {}
        self.topics[topic_id][doc_id] = int(relevance_score)
    
    def get_Rel(self, topic_id, doc_id):
        if topic_id not in self.topics:
            return None
        if doc_id not in self.topics[topic_id]:
            return 0
        return self.topics[topic_id][doc_id]
    
    def get_total_relevant(self, topic_id):
        if topic_id not in self.topics:
            return 0
        return sum(1 for score in self.topics[topic_id].values() if score > 0)



def load_relevance_judgements(rels_dir):

    rels = DocRelScore()

    if not os.path.exists(rels_dir):
        print(f"Error: dirc not found {rels_dir}")
        return rels

    for filename in os.listdir(rels_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(rels_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    rels.add_doc(parts[0], parts[1], parts[2])
    
    return rels

def load_model_rankings(output_dir, model_prefix):

    rankings = {}

    if not os.path.exists(output_dir):
        return rankings
    
    for filename in os.listdir(output_dir):
        if filename.startswith(model_prefix) and filename.endswith(".dat"):
            topic_id = filename.split('_')[1]
            filepath = os.path.join(output_dir, filename)

            doc_list = []
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        doc_list.append(parts[0])
            rankings[topic_id] = doc_list
    return rankings


def calc_precision_at_k( topic_id ,rankings, rels, k=10):

    relevant_count = 0
    top_k = rankings[topic_id][:k]

    for doc_id in top_k:
        if rels.get_Rel(topic_id, doc_id) > 0:
            relevant_count += 1 

    return relevant_count / k if k > 0 else 0.0

def calc_average_precision(topic_id ,rankings, rels):

    total_relevant = rels.get_total_relevant(topic_id)

    if total_relevant == 0:
        return 0.0
    
    relevant_count = 0
    sum_precision = 0.0

    
    for i, doc_id in enumerate(rankings[topic_id]):
            if rels.get_Rel(topic_id, doc_id) > 0:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1) 
                sum_precision += precision_at_i

    return sum_precision / total_relevant

def calc_dcg_at_k(topic_id ,rankings, rels, k=10):

    top_k = rankings[topic_id][:k]

    dcg = 0.0

    for i, doc_id in enumerate(top_k):
        rel_score = rels.get_Rel(topic_id, doc_id)

        if i == 0:
            dcg += rel_score
        else:
           dcg +=  rel_score / math.log2(i + 1 + 1)

    return dcg

def create_table(result_dict , all_topics, models):
        df = pd.DataFrame(index=all_topics, columns=models)
        for model in models:
            for topic in all_topics:
                df.at[topic, model] = result_dict[model].get(topic, 0.0)
        df.loc['Mean'] = df.mean()
        return df

def run_ttest(metric_dict, metric_name):
        print(f"\n--- Metric: {metric_name} ---")

        topics = sorted(list(metric_dict["ModelC"].keys()))
        scores_C = [metric_dict["ModelC"][t] for t in topics]
        scores_B1 = [metric_dict["Baseline1"][t] for t in topics]
        scores_B2 = [metric_dict["Baseline2"][t] for t in topics]

     
        t_stat_1, p_val_1 = stats.ttest_rel(scores_C, scores_B1, alternative='greater')
        t_stat_2, p_val_2 = stats.ttest_rel(scores_C, scores_B2, alternative='greater')

        def interpret(p_val):
            return "Significant (Reject H0)" if p_val < 0.05 else "Not Significant"

        result = f"\n--- Metric: {metric_name} ---\n"
        result += f"Model C vs Baseline 1: t-statistic = {t_stat_1:.4f}, p-value = {p_val_1:.4f} -> {interpret(p_val_1)}\n"
        result += f"Model C vs Baseline 2: t-statistic = {t_stat_2:.4f}, p-value = {p_val_2:.4f} -> {interpret(p_val_2)}\n"

        return result


 