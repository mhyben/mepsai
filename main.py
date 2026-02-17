import json
import re
from os.path import join
from pathlib import Path
from typing import Literal

import pandas as pd

from evaluation import Evaluator
from llm_client import InternetAccessLLM


class PaperProcessor:
    """Process scientific papers and extract information"""

    def __init__(self, documents_folder: Path, prompt_file: str, model='qwen/qwen3-vl-8b'):
        self.documents_folder = Path(documents_folder)
        self.client = InternetAccessLLM(prompt_file=prompt_file, model=model)
        self.evaluator = None

    def process_all_papers(self, file_type: Literal['txt', 'md'] = 'txt', internet_access=True, skip_jsons=True):
        """Process all text/markdown files in the documents folder and save JSON outputs."""
        documents_folder = Path(join(self.documents_folder, file_type))
        documents_folder.mkdir(parents=True, exist_ok=True)

        # Find all source files
        source_files = sorted(list(documents_folder.glob(f"*.{file_type}")))

        if not source_files:
            print(f"No .{file_type} files found in {documents_folder}")
            return

        print(f"Found {len(source_files)} paper(s) to process in '{file_type}' folder\n")

        # Suffix for JSON based on internet access
        json_suffix = ".internet.json" if internet_access else ".local.json"

        for i, paper_file in enumerate(source_files, 1):
            json_file = paper_file.with_name(paper_file.stem + json_suffix)

            if skip_jsons and json_file.exists():
                print(f"[{i}/{len(source_files)}] Skipping {paper_file.name} ({json_file.name} exists)")
                continue

            # Load paper content
            with open(paper_file, "r", encoding="utf-8") as f:
                paper_content = f.read()

            # Prompt the LLM
            json_data = self.client.run(
                user_prompt=paper_content,
                internet_access=internet_access,
                # think=False,
                force_json=True,
                verbose=True
            )

            # Save JSON to corresponding subfolder
            if json_data:
                json_file.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                print(f"[{i}/{len(source_files)}] ✓ Saved results to: {json_file.name}")
            else:
                print(f"[{i}/{len(source_files)}] ✗ Failed to extract valid JSON from response")

    def classification_report(self, file_type: Literal['txt', 'md'] = 'txt', internet_access=True):
        """
        Compare extracted JSON results against ground truth CSV and
        return a Markdown table with per-variable accuracy.

        Match rule:
        A field is considered correct if ANY word from the JSON value
        appears in the ground truth value (after normalization).
        """
        if self.evaluator is None:
            self.evaluator = Evaluator(ground_truth_path='ground_truth.csv', model_name='all-MiniLM-L6-v2')

        self.evaluator.run(file_type, internet_access)


if __name__ == "__main__":
    qwen25 = 'lmstudio-community/Qwen2.5-14B-Instruct-1M-GGUF'
    qwen3_8b = 'qwen/qwen3-vl-8b'
    qwen3_4b = 'qwen/qwen3-vl-4b'
    # Initialize scientific paper processor
    agent = PaperProcessor(documents_folder=Path("documents"),
                           prompt_file="prompt.txt",
                           model=qwen3_8b)

    # Process all .txt files without the use of internet search. Skip papers with existing json.
    agent.process_all_papers(file_type='txt', internet_access=False)

    # Process all .txt files with the use of internet search. Skip papers with existing json.
    agent.process_all_papers(file_type='txt', internet_access=True)

    # Process all .md files without the use of internet search. Skip papers with existing json.
    # agent.process_all_papers(file_type='md', internet_access=False)

    # Process all .md files with the use of internet search. Skip papers with existing json.
    # agent.process_all_papers(file_type='md', internet_access=True)

    # Evaluate each group separately
    agent.classification_report(file_type='txt', internet_access=False)
    agent.classification_report(file_type='txt', internet_access=True)
    # agent.classification_report(file_type='md', internet_access=False)
    # agent.classification_report(file_type='md', internet_access=True)
