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
    More flexible keyword filtering for contingent liabilities
    """
    patterns = [
        r"\bcontingent\s+liabilit(y|ies)\b",
        r"\blegal\s+proceedings\b",
        r"\blitigation\b", 
        r"\bdisputed\s+claims\b",
        r"\btax\s+disputes\b",
        r"\bunacknowledged.*debt\b",
        r"\bcourt\s+cases\b"
    ]
    
    combined_pattern = "|".join(patterns)
    regex = re.compile(combined_pattern, re.IGNORECASE)
    
    return [p for p in pages if regex.search(p['text'])]

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
Determine if the page contains a "Contingent Liabilities" table (NOT guarantees or commitments tables).

Rules:
1. The page must contain a table specifically titled "Contingent Liabilities" or close variations
2. Ignore tables titled "Guarantees", "Bank Guarantees", "Commitments", etc.
3. Must be an actual table, not just a reference.

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
Confirm if the page contains a "Contingent Liabilities" table SPECIFICALLY (not guarantees or commitments).

Requirements:
- Must include the heading "Contingent Liabilities" or very close variation
- Must NOT be titled "Guarantees", "Bank Guarantees", "Commitments"
- Must have tabular format with multiple rows
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

def debug_table_with_context(OUTPUT_PDF_PATH):
    """
    Debug function to understand table context and hierarchy
    """
    result = get_docling_results(OUTPUT_PDF_PATH)
    
    # First, let's see the full document structure
    print("=== FULL DOCUMENT MARKDOWN ===")
    full_markdown = result.document.export_to_markdown()
    print(full_markdown)
    print("=== END FULL DOCUMENT ===\n")
    
    # Now let's examine each table individually
    print(f"Total tables found: {len(result.document.tables)}")
    
    for table_ix, table in enumerate(result.document.tables):
        print(f"\n=== TABLE {table_ix + 1} ANALYSIS ===")
        
        # Get table metadata
        table_dict = table.dict()
        print(f"Table position info: {table_dict.get('prov', 'No position info')}")
        
        # Get the table as DataFrame and markdown
        table_df: pd.DataFrame = table.export_to_dataframe()
        markdown_content = table_df.to_markdown()
        
        print(f"Table shape: {table_df.shape}")
        print(f"Table columns: {list(table_df.columns)}")
        print("\nTable content (markdown):")
        print(markdown_content)
        
        # Try to understand what's around this table in the full document
        print(f"\n--- CONTEXT ANALYSIS ---")
        
        # Look for this table's content in the full document to see its context
        if len(table_df) > 0:
            # Take the first cell content to search for context
            first_cell = str(table_df.iloc[0, 0]) if not table_df.empty else ""
            if first_cell and first_cell in full_markdown:
                # Find the position and show surrounding context
                pos = full_markdown.find(first_cell)
                if pos > 0:
                    context_start = max(0, pos - 500)  # 500 chars before
                    context_end = min(len(full_markdown), pos + 500)  # 500 chars after
                    context = full_markdown[context_start:context_end]
                    print("Surrounding context:")
                    print(context)
        
        print("=== END TABLE ANALYSIS ===\n")

def classifyTable_with_context_check(table_markdown: str = "", table_index: int = 0, full_document_markdown: str = ""):
    """
    Enhanced classification that considers document context
    """
    prompt = f"""
You are analyzing a financial table from an annual report.

CONTEXT: Here's the full document markdown to understand the table's position:
{full_document_markdown[:3000]}

TABLE TO CLASSIFY:
{table_markdown}

TASK: Determine if this specific table is about contingent liabilities.

ANALYSIS STEPS:
1. Look at the full document context above - is this table under a section titled:
   - "Contingent Liabilities" or "Contingent Liability"
   - "Legal Proceedings"  
   - "Litigation"
   
2. REJECT if the table appears under sections titled:
   - "Guarantees"
   - "Bank Guarantees" 
   - "Performance Guarantees"
   - "Letters of Credit"
   - "Commitments"

3. Look at the table content itself:
   - Does it contain legal disputes, court cases, tax disputes?
   - Does it show amounts "not acknowledged as debt"?
   - Does it list uncertain future obligations?

IMPORTANT: Use the document context to understand which section this table belongs to.

Respond ONLY with 'True' or 'False':
- True: If this table is specifically about contingent liabilities
- False: If this table is about guarantees, commitments, or other items

Output: True or False
"""
    
    response = llama_client(prompt)
    return response.strip()

def classifyTable_debug(table_markdown: str = ""):
    """
    Debug version to see what tables we're actually getting
    """
    print(f"\n=== DEBUGGING TABLE ===")
    print(f"Table content preview (first 500 chars):")
    print(table_markdown[:500])
    print(f"=== END TABLE PREVIEW ===\n")
    
    prompt = f"""
You are analyzing a financial table from an annual report.

Task: Analyze this table and provide detailed information about it.

Please analyze:
1. What is the exact title/heading of this table?
2. What type of financial information does it contain?
3. Does it relate to contingent liabilities (uncertain future obligations like legal cases, tax disputes, etc.)?
4. Or does it relate to guarantees/commitments (performance guarantees, bank guarantees given by company)?

Respond in this format:
TITLE: [exact table title/heading]
TYPE: [what kind of financial data]
CONTENT: [brief description of what's in the table]
IS_CONTINGENT_LIABILITY: True/False
REASON: [why you classified it this way]

Table Content:
{table_markdown}
"""
    
    response = llama_client(prompt)
    print(f"LLM Analysis Result:")
    print(response)
    print("=" * 50)
    
    # Extract the IS_CONTINGENT_LIABILITY value from the response
    if "IS_CONTINGENT_LIABILITY: True" in response:
        return "True"
    else:
        return "False"

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

    print(f"[INFO] Prefiltered {len(candidates)} pages containing relevant keywords.")

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

def create_table_with_context(OUTPUT_PDF_PATH):
    """
    Create tables with context awareness
    """
    result = get_docling_results(OUTPUT_PDF_PATH)
    full_markdown = result.document.export_to_markdown()
    
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
        table_df.columns = [
            ILLEGAL_CHARACTERS_RE.sub("", str(col)) for col in table_df.columns
        ]
        table_df = clean_illegal_chars(table_df)

        # Use context-aware classification
        classification_result = classifyTable_with_context_check(
            table_df.to_markdown(), 
            table_ix, 
            full_markdown
        )
        
        print(f"Classification result for {sheet_name}: {classification_result}")

        if "true" in classification_result.lower():
            print(f"Saving contingent liabilities table: {sheet_name}")
            table_df.to_excel(sheet_name + ".xlsx", index=False)

        previous_page_num = current_page_num

def create_table_debug(OUTPUT_PDF_PATH):
    """
    Debug version of create_table to see what's happening
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
        table_df.columns = [
            ILLEGAL_CHARACTERS_RE.sub("", str(col)) for col in table_df.columns
        ]
        table_df = clean_illegal_chars(table_df)

        classification_result = classifyTable_debug(table_df.to_markdown())
        print(f"Classification result for {sheet_name}: {classification_result}")

        if "true" in classification_result.lower():
            print(f"Saving table: {sheet_name}")
            table_df.to_excel(sheet_name + ".xlsx", index=False)

        previous_page_num = current_page_num

# Main execution with different modes
if __name__ == "__main__":
    print("Starting PDF processing with Enhanced Llama LLM...")
    
    # Step 1: Extract relevant pages
    relevant_pages = extract_cg(PDF_PATH)
    
    if relevant_pages:
        print(f"Found relevant pages: {relevant_pages}")
        
        # Choose your processing mode:
        
        # MODE 1: DEBUG - See full document structure and all tables
        # print("\n=== RUNNING DEBUG MODE ===")
        # debug_table_with_context(OUTPUT_PDF_PATH)
        
        # MODE 2: DEBUG TABLE CLASSIFICATION - See how each table is classified
        print("\n=== RUNNING DEBUG TABLE CLASSIFICATION ===")
        create_table_debug(OUTPUT_PDF_PATH)
        
        # MODE 3: PRODUCTION - Context-aware classification (uncomment to use)
        print("\n=== RUNNING PRODUCTION MODE ===")
        create_table_with_context(OUTPUT_PDF_PATH)
        
        print("Processing completed!")
    else:
        print("No contingent liability tables found.")
