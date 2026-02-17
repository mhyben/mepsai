#!/usr/bin/env python3
"""
PDF to Markdown Converter using LightOnOCR-2-1B

This script converts all PDF files in the /documents folder to Markdown format
using the LightOnOCR-2-1B model from LightOn AI.

Requirements:
    pip install torch transformers pillow pypdfium2

    Note: LightOnOCR-2 requires transformers from source:
    pip install git+https://github.com/huggingface/transformers
"""

import os
import sys
from os.path import join
from pathlib import Path
from typing import Optional
import pypdfium2 as pdfium
import torch
from transformers.models.lighton_ocr import LightOnOcrForConditionalGeneration, LightOnOcrProcessor


# from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor


class PDFToMarkdownConverter:
    """Converts PDF files to Markdown using LightOnOCR-2-1B model."""

    def __init__(self, model_name: str = "lightonai/LightOnOCR-2-1B"):
        """
        Initialize the converter with the specified model.

        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading model: {model_name}")

        # Determine device and dtype
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        print(f"Using device: {self.device} with dtype: {self.dtype}")

        # Load model and processor
        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype
        ).to(self.device)

        self.processor = LightOnOcrProcessor.from_pretrained(model_name)
        print("Model loaded successfully!\n")

    def convert_page_to_markdown(self, pil_image) -> str:
        """
        Convert a single page image to markdown text.

        Args:
            pil_image: PIL Image object of the page

        Returns:
            Markdown text extracted from the page
        """
        # Prepare conversation format
        conversation = [{
            "role": "user",
            "content": [{"type": "image", "image": pil_image}]
        }]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {
            k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point()
            else v.to(self.device)
            for k, v in inputs.items()
        }

        # Generate output
        output_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        output_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        return output_text

    def convert_pdf_to_markdown(
            self,
            pdf_path: Path,
            output_path: Optional[Path] = None,
            dpi_scale: float = 2.77
    ) -> str:
        """
        Convert a PDF file to Markdown format.

        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path for output markdown file. If None, 
                        saves alongside the PDF with .md extension
            dpi_scale: Scale factor for rendering (2.77 ≈ 200 DPI, recommended)

        Returns:
            Path to the output markdown file
        """
        print(f"Processing: {pdf_path.name}")

        # Open PDF
        pdf = pdfium.PdfDocument(pdf_path)
        total_pages = len(pdf)
        print(f"  Total pages: {total_pages}")

        markdown_content = []

        # Process each page
        for page_num in range(total_pages):
            print(f"  Converting page {page_num + 1}/{total_pages}...", end=" ")

            page = pdf[page_num]
            # Render at ~200 DPI for best results
            pil_image = page.render(scale=dpi_scale).to_pil()

            # Convert to markdown
            page_markdown = self.convert_page_to_markdown(pil_image)
            markdown_content.append(page_markdown)

            print("✓")

        # Combine all pages
        full_markdown = "\n\n---\n\n".join(markdown_content)

        # Determine output path
        if output_path is None:
            output_path = pdf_path.with_suffix('.md')

        # Save markdown file
        output_path.write_text(full_markdown, encoding='utf-8')
        print(f"  Saved: {output_path.name}\n")

        return str(output_path)


def main():
    """Main function to process all PDFs in /documents folder."""

    # Define documents folder
    documents_folder = Path("documents")

    if not documents_folder.exists():
        print(f"Error: Folder '{documents_folder}' does not exist!")
        print("Please create the folder or update the path in the script.")
        sys.exit(1)

    # Find all PDF files
    pdf_files = list(documents_folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {documents_folder}")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF file(s) to process\n")
    print("=" * 60)

    # Initialize converter
    converter = PDFToMarkdownConverter()

    # Process each PDF
    successful = 0
    failed = 0

    for pdf_file in pdf_files:
        try:
            converter.convert_pdf_to_markdown(pdf_path=pdf_file,
                                              output_path=Path(join('documents', 'md')))
            successful += 1
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {e}\n")
            failed += 1

    # Summary
    print("=" * 60)
    print(f"\nConversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(pdf_files)}")


if __name__ == "__main__":
    main()