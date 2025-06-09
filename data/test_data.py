# import json
# from collections import defaultdict
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# # Load test dataset
# with open('prompt_injection_dataset.json') as f:
#     dataset = json.load(f)

# # Initialize defenders
# from promptinjection_defender.domain_defense import DomainDefender


# import time

# def evaluate_all_domains(dataset_path, max_samples=None):
#     start_time = time.time()
    
#     with open(dataset_path) as f:
#         data = json.load(f)
    
#     # Initialize domain-specific counters
#     results = defaultdict(lambda: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0})
#     domain_defenders = {}  # Cache defenders per domain
    
#     samples = data['test_cases'][:max_samples] if max_samples else data['test_cases']
    
#     for idx, case in enumerate(samples, 1):
#         domain = case['domain']
        
#         # Initialize defender for domain if not exists
#         if domain not in domain_defenders:
#             domain_defenders[domain] = DomainDefender(domain)
        
#         defender = domain_defenders[domain]
#         is_malicious, _ = defender.analyze(case['text'])
        
#         if case['label'] == 'malicious':
#             results[domain]['TP' if is_malicious else 'FN'] += 1
#         else:
#             results[domain]['TN' if not is_malicious else 'FP'] += 1
        
#         # Progress tracking
#         if idx % 100 == 0:
#             elapsed = time.time() - start_time
#             print(f"Processed {idx}/{len(samples)} | {elapsed:.2f}s")

#     # Calculate overall metrics
#     total_results = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
#     for domain in results:
#         for key in total_results:
#             total_results[key] += results[domain][key]
    
#     precision = total_results['TP'] / (total_results['TP'] + total_results['FP']) if (total_results['TP'] + total_results['FP']) > 0 else 0
#     recall = total_results['TP'] / (total_results['TP'] + total_results['FN']) if (total_results['TP'] + total_results['FN']) > 0 else 0
    
#     print(f"""
#     Evaluation Completed in {time.time() - start_time:.2f} seconds
#     ===================================================
#     Overall Performance (All Domains):
#     Accuracy: {(total_results['TP'] + total_results['TN']) / len(samples):.2%}
#     Precision: {precision:.2%}
#     Recall: {recall:.2%}
#     F1-Score: {2 * (precision * recall) / (precision + recall):.2%}
    
#     Domain-Specific Recall:
#     """)
    
#     for domain in sorted(results.keys()):
#         domain_recall = results[domain]['TP'] / (results[domain]['TP'] + results[domain]['FN']) if (results[domain]['TP'] + results[domain]['FN']) > 0 else 0
#         print(f"    {domain:<12}: {domain_recall:.2%}")

# # First test with 100 samples per domain
# evaluate_all_domains("prompt_injection_dataset.json")


import json
import time
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from promptinjection_defender.domain_defense import DomainDefender  # or your import path


def evaluate_all_domains(dataset_path: str, max_samples: int = None):
    """
    Loads a JSON file of test_cases, runs each prompt through its DomainDefender,
    and then computes accuracy, precision, recall, and F1‐score overall.
    Also prints per‐domain recall.
    """

    start_time = time.time()
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # If max_samples is set, truncate the test set
    all_cases = data["test_cases"]
    if max_samples:
        all_cases = all_cases[:max_samples]

    # We'll build flat lists of true labels and predicted labels:
    #   1 => malicious, 0 => safe
    y_true = []
    y_pred = []
    # To compute per‐domain recall later, collect TP/FN per domain
    domain_counts = defaultdict(lambda: {"TP": 0, "FN": 0})

    # Cache one defender instance per domain
    defenders = {}
    latencies = []


    print("Starting evaluation on", len(all_cases), "samples...\n")
    for idx, case in enumerate(all_cases, 1):
        domain = case["domain"]
        text = case["text"]
        label_str = case["label"].lower()
        # Convert label to integer: malicious=1, safe=0
        true_label = 1 if label_str == "malicious" else 0
        y_true.append(true_label)

        # Instantiate or reuse the DomainDefender for this domain
        if domain not in defenders:
            defenders[domain] = DomainDefender(domain)
        defender = defenders[domain]

        # Run the defense check: is_malicious=True/False
        start_pred = time.time()
        is_malicious, _ = defender.analyze(text)
        end_pred = time.time()
        latencies.append(end_pred - start_pred) 
        pred_label = 1 if is_malicious else 0
        y_pred.append(pred_label)

        # Track domain‐specific TP/FN
        if true_label == 1:  # actual malicious
            if pred_label == 1:
                domain_counts[domain]["TP"] += 1
            else:
                domain_counts[domain]["FN"] += 1

        # Progress output every 100 samples
        if idx % 100 == 0 or idx == len(all_cases):
            elapsed = time.time() - start_time
            print(f"Processed {idx}/{len(all_cases)} samples ({elapsed:.1f}s elapsed)")

    # Compute overall metrics using sklearn
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n=== Overall Evaluation ===")
    print(f"Total samples     : {len(all_cases)}")
    print(f"Accuracy          : {accuracy:.2%}")
    print(f"Precision (mal.)  : {precision:.2%}")
    print(f"Recall    (mal.)  : {recall:.2%}")
    print(f"F1 Score  (mal.)  : {f1:.2%}\n")

    # Also print a full classification report (Safe vs Malicious)
    print("Detailed classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["safe", "malicious"],
            zero_division=0,
        )
    )

    # Print domain‐specific recall
    print("=== Domain‐Specific Recall ===")
    for domain in sorted(domain_counts.keys()):
        tp = domain_counts[domain]["TP"]
        fn = domain_counts[domain]["FN"]
        if tp + fn > 0:
            domain_recall = tp / (tp + fn)
            print(f"{domain:<12}: {domain_recall:.2%}  (TP={tp}, FN={fn})")
        else:
            print(f"{domain:<12}: no malicious cases in test set")

    total_time = time.time() - start_time
    print(f"\nEvaluation completed in {total_time:.2f} seconds.")

    # Calculate and print latency metrics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    print(f"=== Latency Metrics ===")
    print(f"Average prediction latency : {avg_latency * 1000:.2f} ms")
    print(f"Min prediction latency     : {min_latency * 1000:.2f} ms")
    print(f"Max prediction latency     : {max_latency * 1000:.2f} ms")

if __name__ == "__main__":
    # Example usage (adjust path as needed):
    evaluate_all_domains("prompt_injection_dataset.json", max_samples=None)
