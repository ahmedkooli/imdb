from transformers import pipeline
from sklearn.metrics import classification_report

from utils import time_it, batch_iterate, val_to_label, get_device


@time_it
def run(test_texts, test_labels, model_name, candidate_labels, batch_size):
    device = get_device()
    classifier = pipeline("zero-shot-classification", model=model_name, device=device)

    results = []
    for batch in batch_iterate(test_texts, batch_size):
        result = classifier(batch, candidate_labels)
        results.append(result)

    results_flat = [r for all_res in results for r in all_res]
    test_pred = [val_to_label(res["labels"][0]) for res in results_flat]

    report = classification_report(test_labels, test_pred, output_dict=False)
    print(report)
    return report


if __name__ == "main":
    run()
