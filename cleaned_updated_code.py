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

PDF_PATH = r"indus towers AR.pdf"
OUTPUT_PDF_PATH = r"indus towers AR_new.pdf"

def load_pdf_pages(pdf_path):
    """Load a PDF file and return its content as a list of strings, each representing a page."""
    pdf_document = fitz.open(pdf_path)
    pages = []
    for page in range(len(pdf_document)):
        text = pdf_document[page].get_text("text")
        pages.append({"page_num": page, "text": text})
    return pages, pdf_document

def keyword_prefilter(pages):
    """Enhanced keyword filtering for contingent liabilities"""
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
    """Helper function to parse JSON from Llama response, handling potential formatting issues."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return {"relevance": "Non Relevant", "confidence": 0.0}

def stage_1_classify(page_text):
    """Classify the page text using Llama model."""
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

def classifyTable_with_context(table_markdown: str = "", full_document_markdown: str = ""):
    """Enhanced classification that considers document context"""
    prompt = f"""
You are analyzing a financial table from an annual report.

CONTEXT: Here's the relevant document context to understand the table's position:
{full_document_markdown[:2000]}

TABLE TO CLASSIFY:
{table_markdown}

TASK: Determine if this specific table is about contingent liabilities.

ANALYSIS STEPS:
1. Look at the document context - is this table under a section titled:
   - "Contingent Liabilities" or "Contingent Liability"
   - "Legal Proceedings"  
   - "Litigation"
   
2. REJECT if the table appears under sections titled:
   - "Guarantees"
   - "Bank Guarantees" 
   - "Performance Guarantees"
   - "Letters of Credit"
   - "Commitments"

3. Look at the table content:
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

def clean_illegal_chars(df):
    return df.applymap(
        lambda x: ILLEGAL_CHARACTERS_RE.sub("", str(x)) if isinstance(x, str) else x
    )

def get_docling_pipeline():
    """Get docling pipeline."""
    try:
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            table_structure_options=dict(do_cell_matching=True, mode=TableFormerMode.ACCURATE)
        )
        
        doc_converter_global = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        settings.debug.profile_pipeline_timings = False
        return doc_converter_global
    
    except Exception as e:
        return None

doc_converter_global = get_docling_pipeline()

def get_docling_results(INPUT_PDF_SOURCE):
    result = doc_converter_global.convert(INPUT_PDF_SOURCE)
    return result

def extract_cg(pdf_path):
    """Main function to extract Contingent Liabilities from a PDF using Llama."""
    pages, pdf_document = load_pdf_pages(pdf_path)
    candidates = keyword_prefilter(pages)

    relevant_pages = []

    for p in candidates:
        stage1_result = stage_1_classify(p['text'])

        if isinstance(stage1_result, dict) and stage1_result.get("relevance") == "Relevant":
            confidence = stage1_result.get("confidence", 0)
            if confidence >= 0.85:
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
        return None

def create_table_production(OUTPUT_PDF_PATH):
    """Production version - clean output with context awareness"""
    result = get_docling_results(OUTPUT_PDF_PATH)
    full_markdown = result.document.export_to_markdown()
    
    previous_page_num = None
    current_page_table_count = 0
    tables_saved = 0
    
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

        classification_result = classifyTable_with_context(
            table_df.to_markdown(), 
            full_markdown
        )

        if "true" in classification_result.lower():
            table_df.to_excel(sheet_name + ".xlsx", index=False)
            tables_saved += 1

        previous_page_num = current_page_num
    
    return tables_saved

# Main execution
if __name__ == "__main__":
    # Step 1: Extract relevant pages
    relevant_pages = extract_cg(PDF_PATH)
    
    if relevant_pages:
        # Step 2: Process tables with context awareness
        tables_saved = create_table_production(OUTPUT_PDF_PATH)
    else:
        tables_saved = 0
