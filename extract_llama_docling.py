import json
import re
import ast
import requests
from typing import Optional, Any
import warnings
warnings.filterwarnings("ignore")

import fitz
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.settings import settings
import pandas as pd

load_dotenv()
import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False 
    return session

configure_http_backend(backend_factory=backend_factory)

class HostedLLM(LLM):
    def __init__(self, endpoint: str, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "endpoint", endpoint)

    @property
    def _llm_type(self) -> str:
        return "Hosted LLM"

    def _call(self, prompt: str, stop=None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        try:
            prompt_template = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            payload = json.dumps({
                "provider": "tgi", 
                "deployment": "Llama 3.3 v1", 
                "spec_version": 1, 
                "input_text": prompt_template, 
                "params": {"temperature": 0.1}
            })
            headers = {'token': '0e5e7', 'Content-Type': 'application/json'}
            response = requests.request("POST", url=self.endpoint, 
                                      headers=headers, data=payload, verify=False)
            response_v = ast.literal_eval(response.text)
            resp_o = response_v['output']
            output = str(resp_o).replace(prompt_template, "")
            return output.strip()
        except Exception as e:
            return f"LLM Call Failed: {e}"

# Initialize Llama LLM
llama_client = HostedLLM(endpoint="https://llmgateway.crisil.local/api/v1/llm")

PDF_PATH = r"2024-annual-report-storebrand-asa 2.pdf"
OUTPUT_PDF_PATH = r"2024-annual-report-storebrand-asa 2_new.pdf"

def load_pdf_pages(pdf_path):
    """
    Load a PDF file and return its content as a list of strings, each representing a page.
    """
    pdf_document = fitz.open(pdf_path)
    pages = []
    for page in range(len(pdf_document)):
        text = pdf_document[page].get_text("text")
        pages.append({"page_num": page, "text": text})
    return pages, pdf_document

def keyword_prefilter(pages):
    """
    Filter pages based on the presence of specific keywords.
    """
    pattern = re.compile(r"\bcontingent\s+liabilit(y|ies)\b", re.IGNORECASE)
    return [p for p in pages if pattern.search(p['text'])]

def parse_llama_json_response(response_text):
    """
    Helper function to parse JSON from Llama response, handling potential formatting issues.
    """
    try:
        # First try direct JSON parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response if it contains extra text
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: create a basic response structure
        print(f"Warning: Could not parse JSON from response: {response_text}")
        return {"relevance": "Non Relevant", "confidence": 0.0}

def stage_1_classify(page_text):
    """
    Classify the page text using Llama model.
    """
    prompt = f"""
You are an expert Financial Analyst. You will be given the text content of a PDF page from an annual report.
Determine if the page contains the "Contingent Liabilities" table.

Rules:
1. The page must contain an actual table, not just a reference.
2. Ignore partial or incomplete mentions.

Return ONLY valid JSON in this exact format:
{{
  "relevance": "Relevant" or "Non Relevant",
  "confidence": 0.85
}}

Page Text:
{page_text}
"""
    
    response = llama_client(prompt)
    return parse_llama_json_response(response)

def stage_2_classify(page_text):
    """
    Classify the page text using Llama model for verification.
    """
    prompt = f"""
You are verifying a page for accuracy.
Confirm if the page contains the "Contingent Liabilities" table

Requirements:
- Must include the heading "Contingent Liabilities" or close variation.
- Must have tabular format with multiple rows.
- If any requirement is missing mark as false.

Return ONLY valid JSON in this exact format:
{{
  "relevance": "Relevant" or "Non Relevant",
  "confidence": 0.90
}}

Page Text:
{page_text}
"""
    
    response = llama_client(prompt)
    return parse_llama_json_response(response)

def classifyTable(table_markdown: str = ""):
    """
    Classify if the table is a contingent liabilities table using Llama.
    """
    prompt = f"""
You are a highly reliable AI assistant with deep expertise in financial reporting and analysis, especially in identifying tables related to contingent liabilities.

Task: Given the content of a financial table, determine if it represents a contingent liabilities table.

Instructions:
- Carefully analyze the table content provided below.
- Look for keywords or phrases commonly associated with contingent liabilities, such as:
  - "contingent liability", "contingent liabilities", "guarantees", "claims", "litigation", "disputed", "not acknowledged as debt", "pending cases", "bonds", "letters of credit", "bank guarantees", "legal proceedings", "tax disputes", "unascertained liabilities", "claims against the company", "arbitration", "court cases", "surety", "indemnity", "commitments", "possible obligations"
- Consider the context, column headers, and any notes or footnotes that indicate potential or possible obligations not yet recognized as actual liabilities.
- Respond ONLY with 'True' if the table is about contingent liabilities, or 'False' if it is not.
- Do not provide explanations or any additional text.

Output: True or False

Table Content:
{table_markdown}
"""
    
    response = llama_client(prompt)
    return response.strip()

def parse_page_with_docling(pdf_path, page_num):
    """
    Parse a specific page of the PDF using Docling.
    """
    converter = DocumentConverter()
    result = converter.convert(pdf_path, page_range=(page_num, page_num + 1))
    return result.document.export_to_markdown()

def extract_table_from_docling_markdown(markdown_text):
    """
    Extract the Contingent Liabilities table from the Docling markdown text using Llama.
    """
    prompt = f"""
You are a financial data extraction expert.
You are given a markdown version of a PDF page with clearly formatted tables.

Your task:
- Identify the table titled "Contingent Liabilities" or close variations.
- Extract ONLY that table into a structured JSON array where each row is a dictionary.
- Ignore all other tables.

Return ONLY valid JSON in this format:
[
    {{ "Column1": "Value1", "Column2": "Value2" }},
    {{ "Column1": "Value3", "Column2": "Value4" }}
]

If no contingent Liabilities table is found, return: []

Markdown:
{markdown_text}
"""
    
    response = llama_client(prompt)
    try:
        # Try to parse the response as JSON
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # Try to extract JSON array from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        print(f"Error parsing JSON from Llama response: {response}")
        return []

def clean_illegal_chars(df):
    return df.applymap(
        lambda x: ILLEGAL_CHARACTERS_RE.sub("", str(x)) if isinstance(x, str) else x
    )

def get_docling_pipeline():
    """
    Get docling pipeline.
    """
    try:
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            table_structure_options=dict(do_cell_matching=True, mode=TableFormerMode.ACCURATE)
        )
        
        doc_converter_global = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        settings.debug.profile_pipeline_timings = True
        return doc_converter_global
    
    except Exception as e:
        print(f"Exception occurred while getting the docling pipeline: {e}")

doc_converter_global = get_docling_pipeline()

def get_docling_results(INPUT_PDF_SOURCE):
    result = doc_converter_global.convert(INPUT_PDF_SOURCE)
    return result

def extract_cg(pdf_path):
    """
    Main function to extract Contingent Liabilities from a PDF using Llama.
    """
    pages, pdf_document = load_pdf_pages(pdf_path)
    candidates = keyword_prefilter(pages)

    print(f"[INFO] Prefiltered {len(candidates)} pages containing 'Contingent Liabilities' keyword.")

    relevant_pages = []

    for p in candidates:
        stage1_result = stage_1_classify(p['text'])
        print(f"[DEBUG] Stage 1 - Page {p['page_num'] + 1}: {stage1_result}")

        if isinstance(stage1_result, dict) and stage1_result.get("relevance") == "Relevant":
            confidence = stage1_result.get("confidence", 0)
            if confidence >= 0.85:
                print("Inside Stage 2")
                stage2_result = stage_2_classify(p['text'])
                print(f"[DEBUG] Stage 2 - Page {p['page_num'] + 1}: {stage2_result}")

                if isinstance(stage2_result, dict) and stage2_result.get("relevance") == "Relevant":
                    relevant_pages.append(p['page_num'])

    if relevant_pages:
        pdf = fitz.open(pdf_path)
        new_pdf = fitz.open()

        for page in relevant_pages:
            new_pdf.insert_pdf(pdf, from_page=page, to_page=page)
        new_pdf.save(OUTPUT_PDF_PATH)
        new_pdf.close()
        pdf.close()

        return relevant_pages
    else:
        print("No relevant pages found.")
        return None

def create_table(OUTPUT_PDF_PATH):
    """
    Create tables from the filtered PDF using Llama for classification.
    """
    result = get_docling_results(OUTPUT_PDF_PATH)
    previous_page_num = None
    current_page_table_count = 0
    
    for table_ix, table in enumerate(result.document.tables):
        current_page_num = table.dict()['prov'][0]['page_no']

        if previous_page_num is None:
            previous_page_num = current_page_num

        if previous_page_num == current_page_num:
            current_page_table_count += 1
        else:
            current_page_table_count = 1

        sheet_name = f"Page_no_{current_page_num}_table_{current_page_table_count}"

        table_df: pd.DataFrame = table.export_to_dataframe()
        # Clean column names
        table_df.columns = [
            ILLEGAL_CHARACTERS_RE.sub("", str(col)) for col in table_df.columns
        ]
        table_df = clean_illegal_chars(table_df)

        classification_result = classifyTable(table_df.to_markdown())
        print(f"Classification result for {sheet_name}: {classification_result}")

        if "true" in classification_result.lower():
            print(f"Saving contingent liabilities table: {sheet_name}")
            table_df.to_excel(sheet_name + ".xlsx", index=False)

        previous_page_num = current_page_num

# Main execution
if __name__ == "__main__":
    print("Starting PDF processing with Llama LLM...")
    relevant_pages = extract_cg(PDF_PATH)
    if relevant_pages:
        print(f"Found relevant pages: {relevant_pages}")
        create_table(OUTPUT_PDF_PATH)
        print("Processing completed!")
    else:
        print("No contingent liability tables found.")
