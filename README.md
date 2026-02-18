# Meta-Study Papers Analyzer
This project contains an implementation of the pipeline described in the paper 
"Automating Data Extraction in Meta-Research: A Multi-Model Benchmark and Reproducible API Pipeline".

The project consists of the following parts:
1. OCR - conversion of the PDF documents to Markdown format
2. Data Extraction - structured list of variable per paper
3. Evaluation - comparison of the extracted data against the ground truth 

## Installation
### OCR setup
Convert documents (PDF, images, etc.) to Markdown format using LightOnOcr.

```bash 
conda create -n mepsai python=3.12 && conda activate mepsai
pip install -r requirements.txt
```

### LM Studio setup
Install LM Studio Python SDK
```bash
pip install lmstudio
````

Make sure LM Studio is running
Either the GUI app or via CLI:
```bash
lms daemon up
lms server start
```

- LM Studio requires the LM Studio application to be running (GUI or daemon)
- The model name format may differ from Hugging-Face (e.g., "qwen3-vl:8b" vs "qwen/qwen3-vl-8b")
- LM Studio has better built-in support for tool calling with compatible models
- Temperature and other parameters go in a `config` file.

### Running the script
```bash
python main.py
```