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

def enhanced_keyword_prefilter(pages):
    """Enhanced keyword filtering that includes continuation pages"""
    primary_patterns = [
        r"\bcontingent\s+liabilit(y|ies)\b",
        r"\blegal\s+proceedings\b",
        r"\blitigation\b", 
        r"\bdisputed\s+claims\b",
        r"\btax\s+disputes\b",
        r"\bunacknowledged.*debt\b",
        r"\bcourt\s+cases\b"
    ]
    
    # Secondary patterns for continuation pages
    continuation_patterns = [
        r"\b(continued|contd\.?)\b",
        r"\b(brought\s+forward|b/f)\b",
        r"\b(total|sub-total)\b.*\d+",
        r"\|\s*\w+\s*\|\s*\d+",  # Table-like patterns
        r"previous\s+year",
        r"current\s+year"
    ]
    
    primary_regex = re.compile("|".join(primary_patterns), re.IGNORECASE)
    continuation_regex = re.compile("|".join(continuation_patterns), re.IGNORECASE)
    
    relevant_pages = []
    continuation_candidates = []
    
    # First pass: Find pages with primary keywords
    for p in pages:
        if primary_regex.search(p['text']):
            relevant_pages.append(p)
    
    # Second pass: Find potential continuation pages
    for p in pages:
        if p not in relevant_pages and continuation_regex.search(p['text']):
            continuation_candidates.append(p)
    
    # Third pass: Check if continuation pages are adjacent to relevant pages
    relevant_page_nums = [p['page_num'] for p in relevant_pages]
    
    for candidate in continuation_candidates:
        page_num = candidate['page_num']
        # Check if this page is adjacent to a relevant page
        if (page_num - 1 in relevant_page_nums or 
            page_num + 1 in relevant_page_nums):
            relevant_pages.append(candidate)
            print(f"[INFO] Added continuation page {page_num + 1}")
    
    return relevant_pages

def detect_table_continuation(page_text):
    """Detect if a page contains table continuation patterns"""
    continuation_indicators = [
        r"\b(continued|contd\.?)\b",
        r"\b(brought\s+forward|b/f)\b",
        r"^\s*\|\s*\w+.*\|\s*\d+",  # Table rows without header
        r"total.*from.*previous.*page",
        r"sub-total",
        r"grand\s+total"
    ]
    
    for pattern in continuation_indicators:
        if re.search(pattern, page_text, re.IGNORECASE | re.MULTILINE):
            return True
    return False

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
        
        print(f"Warning: Could not parse JSON from response: {response_text}")
        return {"relevance": "Non Relevant", "confidence": 0.0}

def stage_1_classify_enhanced(page_text):
    """Enhanced classification that considers table continuations"""
    prompt = f"""
You are an expert Financial Analyst. You will be given the text content of a PDF page from an annual report.
Determine if the page contains a "Contingent Liabilities" table OR a continuation of such a table.

Rules:
1. ACCEPT if page contains a table titled "Contingent Liabilities" or close variations
2. ACCEPT if page appears to be a continuation of a contingent liabilities table (contains words like "continued", "brought forward", or table data without headers)
3. IGNORE tables titled "Guarantees", "Bank Guarantees", "Commitments", etc.
4. ACCEPT table fragments that appear to be part of a larger contingent liabilities table

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

def stage_2_classify_enhanced(page_text):
    """Enhanced verification that considers table continuations"""
    prompt = f"""
You are verifying a page for contingent liabilities content.
Confirm if the page contains a "Contingent Liabilities" table OR continuation of such a table.

ACCEPT IF:
- Has heading "Contingent Liabilities" with table data
- Contains continuation markers like "continued", "brought forward", "contd."
- Has table data that appears to be part of a contingent liabilities table
- Contains legal/litigation related amounts in tabular format

REJECT IF:
- Titled "Guarantees", "Bank Guarantees", "Commitments"
- Just mentions contingent liabilities without table data
- Contains unrelated tabular data

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

def classifyTable_with_context_enhanced(table_markdown: str = "", full_document_markdown: str = "", table_position: int = 0):
    """Enhanced classification that handles table fragments and continuations"""
    prompt = f"""
You are analyzing a financial table from an annual report. This might be a complete table or a fragment of a larger table.

CONTEXT: Here's the relevant document context:
{full_document_markdown[:2000]}

TABLE TO CLASSIFY (Table #{table_position}):
{table_markdown}

TASK: Determine if this table/fragment is part of contingent liabilities.

ANALYSIS:
1. Check document context for section headings:
   ✅ ACCEPT: "Contingent Liabilities", "Legal Proceedings", "Litigation"
   ❌ REJECT: "Guarantees", "Bank Guarantees", "Commitments"

2. Check table content:
   ✅ ACCEPT: Legal disputes, court cases, tax disputes, unacknowledged debts
   ❌ REJECT: Performance guarantees, letters of credit, commitments

3. Check for continuation indicators:
   ✅ ACCEPT: "continued", "brought forward", table fragments without headers
   
4. Consider table position:
   - If this is table #2 or higher on a page, it might be a continuation

IMPORTANT: 
- Table fragments without clear headers might still be contingent liabilities if context suggests it
- Use document structure to understand table purpose
- Consider this might be part of a multi-page table

Respond ONLY with 'True' or 'False':
- True: If this is contingent liabilities table or fragment
- False: If this is guarantees, commitments, or unrelated content

Output: True or False
"""
    
    response = llama_client(prompt)
    return response.strip()

def group_adjacent_pages(page_numbers):
    """Group adjacent page numbers to handle multi-page tables"""
    if not page_numbers:
        return []
    
    page_numbers = sorted(page_numbers)
    groups = []
    current_group = [page_numbers[0]]
    
    for i in range(1, len(page_numbers)):
        if page_numbers[i] - page_numbers[i-1] <= 1:  # Adjacent or same page
            current_group.append(page_numbers[i])
        else:
            groups.append(current_group)
            current_group = [page_numbers[i]]
    
    groups.append(current_group)
    return groups

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
        print(f"Exception occurred while getting the docling pipeline: {e}")

doc_converter_global = get_docling_pipeline()

def get_docling_results(INPUT_PDF_SOURCE):
    result = doc_converter_global.convert(INPUT_PDF_SOURCE)
    return result

def extract_cg_enhanced(pdf_path):
    """Enhanced function that handles multi-page tables using Code 2's successful approach"""
    pages, pdf_document = load_pdf_pages(pdf_path)
    candidates = enhanced_keyword_prefilter(pages)

    print(f"[INFO] Found {len(candidates)} pages with relevant keywords (including continuations).")

    relevant_pages = []

    for p in candidates:
        # Use enhanced classification
        stage1_result = stage_1_classify_enhanced(p['text'])
        print(f"[INFO] Page {p['page_num'] + 1}: Stage 1 - {stage1_result.get('relevance', 'Unknown')}")

        if isinstance(stage1_result, dict) and stage1_result.get("relevance") == "Relevant":
            confidence = stage1_result.get("confidence", 0)
            if confidence >= 0.75:  # Slightly lower threshold for continuations
                stage2_result = stage_2_classify_enhanced(p['text'])
                print(f"[INFO] Page {p['page_num'] + 1}: Stage 2 - {stage2_result.get('relevance', 'Unknown')}")

                if isinstance(stage2_result, dict) and stage2_result.get("relevance") == "Relevant":
                    relevant_pages.append(p['page_num'])

    if relevant_pages:
        # IMPORTANT: Add adjacent pages to handle multi-page tables
        # This is what Code 2 was doing implicitly
        extended_pages = set(relevant_pages)
        
        for page_num in relevant_pages:
            # Add previous and next pages to catch table continuations
            if page_num > 0:
                extended_pages.add(page_num - 1)
            if page_num < len(pages) - 1:
                extended_pages.add(page_num + 1)
        
        # Filter out pages that are too far from our main candidates
        final_pages = []
        for page_num in sorted(extended_pages):
            # Only add if within 1 page of an original relevant page
            if any(abs(page_num - rel_page) <= 1 for rel_page in relevant_pages):
                final_pages.append(page_num)
        
        print(f"[INFO] Extended pages to include continuations: {final_pages}")
        
        pdf = fitz.open(pdf_path)
        new_pdf = fitz.open()

        # Add all relevant pages to the new PDF (including extensions)
        for page in sorted(final_pages):
            new_pdf.insert_pdf(pdf, from_page=page, to_page=page)
        new_pdf.save(OUTPUT_PDF_PATH)
        new_pdf.close()
        pdf.close()

        print(f"[INFO] Created filtered PDF with {len(final_pages)} pages (including continuations).")
        return final_pages
    else:
        print("[INFO] No relevant pages found.")
        return None

def create_table_production_code2_style(OUTPUT_PDF_PATH):
    """
    Production version using Code 2's successful approach:
    1. Process ENTIRE PDF with Docling at once (not page by page)
    2. Let Docling handle multi-page table detection automatically
    3. Use full document context for classification
    """
    # STEP 1: Process the ENTIRE filtered PDF with Docling
    # This is the key - Docling processes all pages together and can detect
    # multi-page tables automatically
    result = get_docling_results(OUTPUT_PDF_PATH)
    
    # STEP 2: Get the COMPLETE document markdown
    # This includes all tables, even those spanning multiple pages
    full_markdown = result.document.export_to_markdown()
    
    print(f"[INFO] Docling found {len(result.document.tables)} tables across all pages")
    print(f"[INFO] Document contains {len(full_markdown)} characters of content")
    
    # STEP 3: Process each table that Docling found
    # Docling automatically handles table continuations and merges table parts
    previous_page_num = None
    current_page_table_count = 0
    tables_saved = 0
    
    for table_ix, table in enumerate(result.document.tables):
        # Get table metadata
        table_info = table.dict()
        current_page_num = table_info['prov'][0]['page_no'] if table_info.get('prov') else 0

        # Generate unique sheet name
        if previous_page_num is None:
            previous_page_num = current_page_num

        if previous_page_num == current_page_num:
            current_page_table_count += 1
        else:
            current_page_table_count = 1

        sheet_name = f"Page_no_{current_page_num}_table_{current_page_table_count}"

        # Convert table to DataFrame
        table_df: pd.DataFrame = table.export_to_dataframe()
        
        # Clean column names and data
        table_df.columns = [
            ILLEGAL_CHARACTERS_RE.sub("", str(col)) for col in table_df.columns
        ]
        table_df = clean_illegal_chars(table_df)

        # STEP 4: Use Code 2's context-aware classification
        # This is exactly what Code 2 was doing successfully
        classification_result = classifyTable_with_context(
            table_df.to_markdown(), 
            full_markdown  # Full document context - this is the key!
        )
        
        print(f"[INFO] {sheet_name}: {'✅ Contingent Liability' if 'true' in classification_result.lower() else '❌ Other Table'}")

        if "true" in classification_result.lower():
            # Save the table
            excel_filename = sheet_name + ".xlsx"
            table_df.to_excel(excel_filename, index=False)
            tables_saved += 1
            print(f"[SUCCESS] Saved: {excel_filename} ({len(table_df)} rows)")

        previous_page_num = current_page_num
    
    print(f"[COMPLETE] Saved {tables_saved} contingent liability tables.")
    return tables_saved

def classifyTable_with_context(table_markdown: str = "", full_document_markdown: str = ""):
    """
    This is the EXACT approach from Code 2 that was working correctly.
    Uses full document context to classify tables properly.
    """
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

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Enhanced Multi-Page PDF Processing (Code 2 Style)...")
    
    # Step 1: Extract relevant pages (including adjacent pages for multi-page tables)
    relevant_pages = extract_cg_enhanced(PDF_PATH)
    
    if relevant_pages:
        print(f"✅ Found relevant pages: {relevant_pages}")
        
        # Step 2: Process tables using Code 2's successful approach
        # This processes the ENTIRE filtered PDF at once, allowing Docling
        # to automatically detect and handle multi-page tables
        tables_saved = create_table_production_code2_style(OUTPUT_PDF_PATH)
        
        if tables_saved > 0:
            print("🎉 Multi-page processing completed successfully!")
        else:
            print("⚠️ No contingent liability tables were saved.")
    else:
        print("❌ No contingent liability tables found.")
