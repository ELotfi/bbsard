import os
import json
from os.path import abspath, join

import torch
import pandas as pd

from utils.data import BSARDataset
from utils.eval import BiEncoderEvaluator
from models.trainable_dense_models import BiEncoder



if __name__ == '__main__':
    # 1. Load an already-trained BiEncoder.
    checkpoint_path = "/home/ubuntu/projects/bsard/outputs/Oct31-09-38-21/59" #"output/training/Nov10-14-11-42_bitower-camembert-base-1000-200-20-22/99"
    model = BiEncoder.load(checkpoint_path)

    # 2. Load the test set.
    test_queries_df = pd.read_csv("/home/ubuntu/projects/bsard/data/dutch/du_test_T0_filtered.csv")
    documents_df = pd.read_csv("/home/ubuntu/projects/bsard/data/dutch/du_articles_new_filtered.csv")
    test_dataset = BSARDataset(test_queries_df, documents_df)

    # 3. Initialize the Evaluator.
    evaluator = BiEncoderEvaluator(queries=test_dataset.queries, 
                                   documents=test_dataset.documents, 
                                   relevant_pairs=test_dataset.one_to_many_pairs, 
                                   score_fn=model.score_fn)

    # 4. Run trained model and compute scores.
    scores = evaluator(model=model,
                       device=torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu"),
                       batch_size=512)

    # 5. Save results.
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(join(checkpoint_path, 'test_scores_fixed.json'), 'w') as fOut:
        json.dump(scores, fOut, indent=2)