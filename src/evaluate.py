from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Dict

import anthropic
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
)
from datasets import Dataset

# Configuration
GOLDEN_DATASET_PATH = Path("data/eval/golden_dataset.json")
REPORT_PATH = Path("docs/eval_report.md")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("ragas-eval")

class RAGEvaluator:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def generate_golden_dataset(self, processed_docs_path: Path, num_pairs: int = 50):
        """
        Uses Claude 3.5 Sonnet to generate a golden dataset of (question, ground_truth, context)
        from the processed filings.
        """
        logger.info(f"Generating golden dataset with {num_pairs} pairs...")

        # Sample documents from the processed JSONL
        docs = []
        with open(processed_docs_path, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))

        # For simplicity, we take the first few documents or a random sample
        sample_docs = docs[:num_pairs] if len(docs) >= num_pairs else docs

        golden_dataset = []
        for doc in sample_docs:
            text = doc.get("text", "")[:5000] # Limit context for generation
            company = doc.get("company", "Unknown")

            prompt = (
                f"You are a financial data generator. Based on the following SEC filing excerpt "
                f"from {company}, create a challenging factual question and a precise ground-truth answer.\n\n"
                f"Excerpt:\n{text}\n\n"
                f"Return only a JSON object with the following keys:\n"
                f"{{'question': '...', 'answer': '...', 'context': '...'}}"
            )

            try:
                message = self.client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = next(
                    (
                        block.text
                        for block in message.content
                        if block.type == "text"
                    ),
                    None,
                )
                if response_text is None:
                    raise ValueError("Anthropic returned no text block.")
                pair = json.loads(response_text)
                golden_dataset.append(pair)
            except Exception as e:
                logger.error(f"Failed to generate pair for {company}: {e}")

        # Save dataset
        GOLDEN_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(GOLDEN_DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(golden_dataset, f, indent=2)

        logger.info(f"Golden dataset saved to {GOLDEN_DATASET_PATH}")

    def evaluate_system(self):
        """
        Runs the RAGAS evaluation against the existing system.
        """
        if not GOLDEN_DATASET_PATH.exists():
            raise FileNotFoundError("Golden dataset not found. Run generate_golden_dataset first.")

        with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
            dataset_list = json.load(f)

        # We need to collect the system's responses for each question
        import requests
        API_URL = "http://localhost:8000/query"

        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        logger.info("Querying system for evaluation...")
        for pair in dataset_list:
            try:
                payload = {"query": pair["question"], "agentic": True}
                resp = requests.post(API_URL, json=payload).json()

                eval_data["question"].append(pair["question"])
                eval_data["answer"].append(resp["answer"])
                # RAGAS expects contexts as a list of strings
                eval_data["contexts"].append([s["document"] for s in resp["sources"]])
                eval_data["ground_truth"].append(pair["answer"])
            except Exception as e:
                logger.error(f"Error querying system for {pair['question']}: {e}")

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_dict(eval_data)

        # Run RAGAS evaluation
        logger.info("Running RAGAS metrics...")
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevance,
                context_precision,
            ],
        )

        df = result.to_pandas()
        scores = df.mean(numeric_only=True).to_dict()

        self._write_report(scores)
        return scores

    def _write_report(self, scores: Dict[str, float]):
        """Writes the evaluation results to a markdown report."""
        report = (
            "# RAG Evaluation Report\n\n"
            "## Performance Metrics\n\n"
            "| Metric | Score |\n"
            "| :--- | :--- |\n"
            f"| Faithfulness | {scores.get('faithfulness', 0):.2%} |\n"
            f"| Answer Relevance | {scores.get('answer_relevance', 0):.2%} |\n"
            f"| Context Precision | {scores.get('context_precision', 0):.2%} |\n\n"
            "**Note**: Faithfulness measures if the answer is derived solely from the context. "
            "Answer Relevance measures if the response actually addresses the query. "
            "Context Precision evaluates the quality of the retrieved chunks.\n\n"
            f"Generated on: {Path('.').absolute()}\n"
        )

        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report written to {REPORT_PATH}")

if __name__ == "__main__":
    evaluator = RAGEvaluator()

    # 1. Generate dataset (only if it doesn't exist or forced)
    if not GOLDEN_DATASET_PATH.exists():
        evaluator.generate_golden_dataset(Path("data/processed/documents.jsonl"))

    # 2. Evaluate (Requires API to be running)
    try:
        final_scores = evaluator.evaluate_system()
        print("\nEvaluation Complete!")
        print(json.dumps(final_scores, indent=2))
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Make sure the FastAPI backend is running at http://localhost:8000")
