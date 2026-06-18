from nltk.stem import PorterStemmer
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
import sys
import re
import xml.etree.ElementTree as ET
import math
import os
from Evaluation import *

# install depencencies
import nltk
nltk.download('stopwords')
nltk.download('punkt')


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()



# From W9 Tut
class BowDoc:
    """Bag-of-words representation of a document.

    The document has an ID, and an iterable list of terms with their
    frequencies."""

    def __init__(self, docid):
        """Constructor.

        Set the ID of the document, and initiate an empty term dictionary.
        Call add_term to add terms to the dictionary."""
        self.docid = docid
        self.terms = {}
        self.doc_len = 0

    def add_term(self, term):
        """Add a term occurrence to the BOW representation.

        This should be called each time the term occurs in the document."""
        try:
            self.terms[term] += 1
        except KeyError:  
            self.terms[term] = 1

    def get_term_count(self, term):
        """Get the term occurrence count for a term.

        Returns 0 if the term does not appear in the document."""
        try:
            return self.terms[term]
        except KeyError:
            return 0

    def get_term_freq_dict(self):
        """Return dictionary of term:freq pairs."""
        return self.terms

    def get_term_list(self):
        """Get sorted list of all terms occurring in the document."""
        return sorted(self.terms.keys())

    def get_docid(self):
        """Get the ID of the document."""
        return self.docid

    def __iter__(self):
        """Return an ordered iterator over term--frequency pairs.

        Each element is a (term, frequency) tuple.  They are iterated
        in term's frequency descending order."""
        return iter(sorted(self.terms.items(), key=lambda x: x[1],reverse=True))
        """Or in term alphabetical order:
        return iter(sorted(self.terms.iteritems()))"""
    def get_doc_len(self):
        return self.doc_len

    def set_doc_len(self, doc_len):
        self.doc_len = doc_len

class BowColl:
    """Collection of BOW documents."""

    def __init__(self):
        """Constructor.

        Creates an empty collection."""
        self.docs = {}

    def add_doc(self, doc):
        """Add a document to the collection."""
        self.docs[doc.get_docid()] = doc

    def get_doc(self, docid):
        """Return a document by docid.

        Will raise a KeyError if there is no document with that ID."""
        return self.docs[docid]

    def get_docs(self):
        """Get the full list of documents.

        Returns a dictionary, with docids as keys, and docs as values."""
        return self.docs

    def inorder_iter(self):
        """Return an ordered iterator over the documents.
        
        The iterator will traverse the collection in docid order.  Modifying
        the collection while iterating over it leads to undefined results.
        Each element is a document; to find the id, call doc.get_docid()."""
        return BowCollInorderIterator(self)

    def get_num_docs(self):
        """Get the number of documents in the collection."""
        return len(self.docs)

    def __iter__(self):
        """Iterator interface.

        See inorder_iter."""
        return self.inorder_iter()

class BowCollInorderIterator:
    """Iterator over a collection."""

    def __init__(self, coll):
        """Constructor.
        
        Takes the collection we're going to iterator over as sole argument."""
        self.coll = coll
        self.keys = sorted(coll.get_docs().keys())
        self.i = 0

    def __iter__(self):
        """Iterator interface."""
        return self

    def next(self):
        """Get next element."""
        if self.i >= len(self.keys):
            raise StopIteration
        doc = self.coll.get_doc(self.keys[self.i])
        self.i += 1
        return doc



def LoadTopic(topics_filepath):

    topics = []
    with open(topics_filepath, 'r') as f:
        content = f.read()

    blocks = content.split('<Topic>')
    
    for block in blocks:
        topic_id_re = re.search(r'<num>\s*Number:\s(R\d+)', block)
        topic_title_re = re.search(r'<title>\s*(.*?)\n', block)

        if topic_id_re and topic_title_re:
            topic_id = topic_id_re.group(1).strip()
            topic_title = topic_title_re.group(1).strip()
            topics.append((topic_id, topic_title))

    return topics


def TokenizeAndPreprocess(text):
    text = text.lower()   
    text = re.sub(r'\d+', '', text) #remove digits
    text = re.sub(r'[^\w\s]', '', text) 
    tokens = word_tokenize(text)
    clean_tokens = [stemmer.stem(w) for w in tokens if len(w) > 2 and w not in stop_words]
    return clean_tokens



def LoadDocuments(collection_dir, topic_id):

    coll = BowColl()  
    dataset_path = os.path.join(collection_dir, 'Dataset' + topic_id[-3:]) # e.g R101
    for filename in os.listdir(dataset_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(dataset_path, filename)
            try:
                # XML Element Tree
                tree = ET.parse(file_path)
                root = tree.getroot()

                doc_id = root.attrib.get('itemid', filename)
                text_body = ''
                text_node = root.find('text')
                if text_node is not None:
                    for p in text_node.findall('p'):
                        if p.text:
                            text_body += p.text + " "


                tokens = TokenizeAndPreprocess(text_body)
                doc = BowDoc(doc_id)
                for token in tokens:
                    doc.add_term(token)
                doc.set_doc_len(len(tokens))
                coll.add_doc(doc)
            
            except Exception as e:
                print(f"Error parsing {filename}: {e}")

    return coll

def SaveToFile(ranked_results, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for doc_id, score in ranked_results:
            f.write(f"{doc_id}\t{score}\n")


def Baseline1_BM25(topic_id, query_terms, coll, output_dir):
    k1 = 1.2
    b = 0.75
    k2 = 500

    N = coll.get_num_docs()
    if N == 0:
        return
    
    total_len = 0
    df_dict = {}

    for doc_id, doc in coll.get_docs().items():
        total_len += doc.get_doc_len()
        # Bow 
        for term in doc.get_term_list():
            df_dict[term] = df_dict.get(term, 0) + 1
        
    avdl = total_len / N 

    query_tf = {}

    for term in query_terms:
        query_tf[term] = query_tf.get(term, 0) + 1
    
    ranked_results = []
    # Step 4: Score Each Document
    for doc_id, doc in coll.get_docs().items():
        dl = doc.get_doc_len()
        K = k1 * ((1 - b) + b * (dl / avdl))
        bm25_score = 0.0


        for term, q_fi in query_tf.items():
            n_t = df_dict.get(term, 0)

           

            f_t = doc.get_term_count(term)
            qf_t = query_tf.get(term, 0)

            idf_weight = math.log10(1 + ((N - n_t + 0.5 ) / (n_t + 0.5) ))
            doc_tf_weight = ((k1 + 1) * f_t) / (K + f_t)
            query_tf_weight = ((k2 + 1) * qf_t) / (k2 + qf_t)

            bm25_score += (idf_weight * doc_tf_weight * query_tf_weight)

        ranked_results.append((doc_id, bm25_score))
    
    # sorting
    ranked_results.sort(key=lambda x: x[1], reverse=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f"Baseline1_{topic_id}_Ranking.dat"
    SaveToFile(ranked_results, os.path.join(output_dir, filename))


def Baseline2_JM(topic_id, query_terms, coll, output_dir):

    lam = 0.3

    N = coll.get_num_docs()
    if N == 0:
        return
    

    collection_len = 0
    cf_dict = {}

    for doc_id, doc in coll.get_docs().items():
        collection_len += doc.get_doc_len()
        for term, freq in doc.get_term_freq_dict().items():
            cf_dict[term] = cf_dict.get(term, 0) + freq
    
    if collection_len == 0:
        return

    ranked_results = []

    
    for doc_id, doc in coll.get_docs().items():
        doc_len = doc.get_doc_len() 
        if doc_len == 0 :
            doc_len = 0.5

        jm_score = 0.0

        for q_j in query_terms:

            f_qj_D = doc.get_term_count(q_j)
            c_qj = cf_dict.get(q_j , 0)

            # Cal P(term|Doc) & P(term|Collection)

            P_doc = f_qj_D / doc_len
            P_collection = c_qj / collection_len
            smoothed_prob = ((1 - lam) * P_doc) + (lam * P_collection)

            if smoothed_prob > 0:
                jm_score += math.log10(smoothed_prob)
        
        ranked_results.append((doc_id, jm_score))
    
    #  Sort and Save Output
    ranked_results.sort(key=lambda x: x[1], reverse=True)

    filename = f"Baseline2_{topic_id}_Ranking.dat"
    SaveToFile(ranked_results, os.path.join(output_dir, filename))



# Task 2 Pseudo-Feedback Algorithm

def ModelC_PRF(topic_id, query_terms, coll, max_expansion_terms, output_dir, top_k_doc ,prefix="ModelC"):

    # Hyperparameters
    k1, b, k2 = 1.2, 0.75, 500
    
    # Pass 1 (BM25) - Initial Ranking
    N = coll.get_num_docs()

    total_len = 0
    df_dict = {}

    for doc_id, doc in coll.get_docs().items():
        total_len += doc.get_doc_len()
        for term in doc.get_term_list():
            df_dict[term] = df_dict.get(term, 0) + 1

    avdl = total_len / N 
    
    query_tf = {}

    for term in query_terms:
        query_tf[term] = query_tf.get(term, 0) + 1

    ranked_results = []
    for doc_id, doc in coll.get_docs().items():
        dl = doc.get_doc_len()
        K = k1 * ((1 - b) + b * (dl / avdl))
        bm25_score = 0.0
        for term, q_fi in query_tf.items():

            n_t = df_dict.get(term, 0)
            if n_t == 0: continue

            f_t = doc.get_term_count(term)
            qf_t = query_tf.get(term, 0)
            idf_weight = math.log10(1 + ((N - n_t + 0.5 ) / (n_t + 0.5) ))
            doc_tf_weight = ((k1 + 1) * f_t) / (K + f_t)
            query_tf_weight = ((k2 + 1) * qf_t) / (k2 + qf_t)
            bm25_score += (idf_weight * doc_tf_weight * query_tf_weight)

        ranked_results.append((doc_id, bm25_score))
    # sorting
    ranked_results.sort(key=lambda x: x[1], reverse=True)


    # Pass 2: Query Expansion (Using BM25 IF w5 Formula)
    max_score = ranked_results[0][1] if ranked_results else 0
    K_DOCS = top_k_doc
    pseudo_relevant_docs = ranked_results[:K_DOCS]

    # At least top 1
    if len(pseudo_relevant_docs) == 0 and ranked_results:
        pseudo_relevant_docs = [ranked_results[0]]

    R = len(pseudo_relevant_docs)

    # calculate r(t) , D+
    r_t_dict = {}
    for doc_id, _ in pseudo_relevant_docs:
        doc = coll.get_doc(doc_id)
        for term in doc.get_term_list():
            r_t_dict[term] = r_t_dict.get(term, 0) + 1

    expansion_candidates = {}

    for term, r_t in r_t_dict.items():
        if term in query_tf:
            continue
        n_t = df_dict.get(term, 0)
        # D+
        num1 = r_t + 0.5
        den1 = R - r_t + 0.5
        part1 = num1 / den1  
        # D-
        num2 = n_t - r_t + 0.5
        den2 = N - n_t - R + r_t + 0.5
        part2 = num2 / den2 

        if part2 > 0:
            w5_weight = math.log10(part1 / part2)

            if w5_weight > 0:
                expansion_candidates[term] = w5_weight
    
    # Feature selection
    expanded_terms = []
    if len(expansion_candidates) > 0:

        sorted_candidates = sorted(
            expansion_candidates.items(), key=lambda x : x[1],
            reverse=True
        )

        expanded_terms = [
            term for term , weight in sorted_candidates[:max_expansion_terms]
        ]

    # Pass 3: Re-rank documents with adjusted weights (w5)
    final_term_weights = {}

    # Rocchio-Based Information Filtering Model (Week 7 , p 18)
    alpha = 1.0
    beta = 0.075

    for term, qf_t in query_tf.items():
        n_t = df_dict.get(term, 0)
        idf = math.log10(1 + ((N - n_t + 0.5) / (n_t + 0.5)))
        query_tf_weight = ((k2 + 1) * qf_t) / (k2 + qf_t)

         # Rocchio-Based Information Filtering Model (Week 7 , p 18)
        final_term_weights[term] = alpha * (idf * query_tf_weight)

    for term in expanded_terms:      
        # Rocchio-Based Information Filtering Model (Week 7 , p 18)
        final_term_weights[term] =  final_term_weights.get(term, 0) +  beta * expansion_candidates[term]

  

    final_scores = []
    for doc_id, doc in coll.get_docs().items():
        dl = doc.get_doc_len()
        K = k1 * ((1 - b) + b * (dl / avdl))
        score = 0.0

     
        for term, weight in final_term_weights.items():
            f_t = doc.get_term_count(term)
            if f_t == 0: 
                continue

            # (TF_Component)
            doc_tf_weight = ((k1 + 1) * f_t) / (K + f_t)

           

            score += (weight * doc_tf_weight)
            
        final_scores.append((doc_id, score)) 
        
    
    final_scores.sort(key=lambda x: x[1], reverse=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{prefix}_{topic_id}_Ranking.dat"
    SaveToFile(final_scores, os.path.join(output_dir, filename))

            


def GridSearch_ModelC(folder, topics_file, rels_dir, output_dir):
    print("=== Starting Grid Search for Model C ===")
    
  
    print("Pre-loading documents and relevance judgements...")
    topics = LoadTopic(topics_file)
    rels = load_relevance_judgements(rels_dir)
    
    preloaded_data = {}
    for topic_id, topic_title in topics:
        query_terms = TokenizeAndPreprocess(topic_title)
        coll = LoadDocuments(folder, topic_id)
        preloaded_data[topic_id] = (query_terms, coll)

    max_expansion_terms = [75, 80, 85]
    top_k_doc = [1,2,3,4]
    
    best_map = 0.0
    best_params = (max_expansion_terms[0], top_k_doc[0])
    

 
    for t in max_expansion_terms:
        prefix = f"GridC-t{t}"
        
            
        for top_k in top_k_doc:
            print(f"Testing Parameters -> max_expansion_terms: {t}, top_k_doc: {top_k}")
            for topic_id, (query_terms, coll) in preloaded_data.items():
                if coll.get_num_docs() > 0:
                    ModelC_PRF(topic_id, query_terms, coll, t, output_dir, top_k,prefix=prefix)
            
          
            rankings = load_model_rankings(output_dir, prefix)
            sum_ap = 0.0
            valid_topics = 0
                
            for topic_id in rels.topics.keys():
                if topic_id in rankings:
                    ap = calc_average_precision(topic_id, rankings, rels)
                    sum_ap += ap
                    valid_topics += 1
                
            current_map = sum_ap / valid_topics if valid_topics > 0 else 0.0
            print(f"  └─ Result MAP: {current_map:.4f}\n")
                
    
            if current_map > best_map:
                best_map = current_map
                best_params = (t,top_k )

    print("=== Grid Search Completed ===")
    print(f"Best MAP: {best_map:.4f}")
    print(f"Optimal Max_expansion_terms: {best_params[0]}")
    print(f"Optimal Top_K_doc: {best_params[1]}")
    
    return best_params


def main(folder="Doc_Collection", topics_file="Topics.txt"):
    print("--- main ---")
    topics = LoadTopic(topics_file)
    output_dir = "ModelOutputs"
    rels_dir = "Relevant_Judgements"

    max_expansion_terms, top_K_doc = GridSearch_ModelC(folder, topics_file, rels_dir, output_dir)
    
    for topic_id, topic_title in topics:
        print(f"Processing Topic: {topic_id} | {topic_title}")
        
        query_terms = TokenizeAndPreprocess(topic_title)
        
        coll = LoadDocuments(folder, topic_id)
        
        if coll.get_num_docs() > 0:
            Baseline1_BM25(topic_id, query_terms, coll, output_dir)
            Baseline2_JM(topic_id, query_terms, coll, output_dir)
            ModelC_PRF(topic_id, query_terms, coll, max_expansion_terms, output_dir, top_K_doc, prefix="ModelC")
        else:
            print(f"  └─ Dataset not found, skipping.")


    # Evaluation
    print("\n--- Running Final Evaluation ---")
    rels = load_relevance_judgements(rels_dir)
    models = ["Baseline1", "Baseline2", "ModelC"]

    results_ap = {m: {} for m in models}
    results_p10 = {m: {} for m in models}
    results_dcg10 = {m: {} for m in models}

    for model in models:
        rankings = load_model_rankings(output_dir, model)

        for topic_id in rels.topics.keys():
                if topic_id in rankings:
                    results_ap[model][topic_id] = calc_average_precision(topic_id, rankings, rels)
                    results_p10[model][topic_id] = calc_precision_at_k(topic_id, rankings, rels, k=10)
                    results_dcg10[model][topic_id] = calc_dcg_at_k(topic_id, rankings, rels, k=10)
        
    
    # table
    all_topics = sorted(list(rels.topics.keys()))
    

    table1_map = create_table(results_ap, all_topics, models)
    table2_p10 = create_table(results_p10, all_topics, models)
    table3_dcg = create_table(results_dcg10, all_topics, models)


    eval_dir = "eval"
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    report_text = ""
    
    report_text += "\n" + "="*50 + "\n"
    report_text += "Table 1. The performance of 3 models on Average Precision (MAP)\n"
    report_text += "="*50 + "\n"
    report_text += table1_map.round(4).to_string() + "\n"

    report_text += "\n" + "="*50 + "\n"
    report_text += "Table 2. The performance of 3 models on precision@10\n"
    report_text += "="*50 + "\n"
    report_text += table2_p10.round(4).to_string() + "\n"

    report_text += "\n" + "="*50 + "\n"
    report_text += "Table 3. The performance of 3 models on DCG@10\n"
    report_text += "="*50 + "\n"
    report_text += table3_dcg.round(4).to_string() + "\n"

    


    # task 5

    report_text += "\n" + "="*50 + "\n"
    report_text += "Task 5: Paired t-test Results (Model C vs Baselines)\n"
    report_text += "="*50 + "\n"
    report_text += run_ttest(results_ap, "Average Precision (MAP)")
    report_text += run_ttest(results_p10, "Precision@10")
    report_text += run_ttest(results_dcg10, "DCG@10")


    print(report_text)

    report_path = os.path.join(eval_dir, "Evaluation_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Done : {report_path}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()