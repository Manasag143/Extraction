import json
import re
import ast
import requests
from typing import Optional, Any
import warnings
warnings.filterwarnings("ignore")
import os
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
import numpy as np

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
            headers = {'token': 'vg', 'Content-Type': 'application/json'}
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
SPLIT_PDF_PATH = r"indus towers AR_split.pdf"

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

def split_pdf_page_vertically(input_pdf_path, output_pdf_path):
    """
    HYBRID SOLUTION - PART 1
    Split each PDF page into left and right halves
    One page becomes two separate pages
    """
    print("\n" + "="*60)
    print("SPLITTING PDF PAGES VERTICALLY (LEFT & RIGHT)")
    print("="*60)
    
    pdf = fitz.open(input_pdf_path)
    new_pdf = fitz.open()
    
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        rect = page.rect
        
        # Calculate midpoint
        mid_x = rect.width / 2
        
        print(f"  Page {page_num + 1}: Splitting at x={mid_x:.2f}")
        
        # Left half
        left_rect = fitz.Rect(0, 0, mid_x, rect.height)
        left_page = new_pdf.new_page(width=mid_x, height=rect.height)
        left_page.show_pdf_page(left_page.rect, pdf, page_num, clip=left_rect)
        
        # Right half
        right_rect = fitz.Rect(mid_x, 0, rect.width, rect.height)
        right_page = new_pdf.new_page(width=mid_x, height=rect.height)
        right_page.show_pdf_page(right_page.rect, pdf, page_num, clip=right_rect)
    
    new_pdf.save(output_pdf_path)
    new_pdf.close()
    pdf.close()
    
    print(f"✓ Split PDF saved: {output_pdf_path}")
    print(f"  Original pages: {len(pdf)}")
    print(f"  New pages: {len(pdf) * 2}")

def detect_split_table(df):
    """
    HYBRID SOLUTION - PART 2
    Detect if table is incomplete/split
    Returns: True if table appears to be split
    """
    if df.empty or len(df) < 2:
        return False
    
    try:
        # Check 1: Has no total row
        has_total = any('total' in str(val).lower() for val in df.iloc[:, 0])
        
        if has_total:
            return False  # Table has total, so it's complete
        
        # Check 2: Last column has numeric values
        last_col = df.iloc[:, -1]
        numeric_last_col = pd.to_numeric(last_col, errors='coerce')
        has_numeric_data = numeric_last_col.notna().sum() > 0
        
        # Check 3: First row looks like header (has text)
        first_row_is_header = any(isinstance(val, str) and len(str(val)) > 2 
                                   for val in df.iloc[0])
        
        # Table is split if: no total + has numeric data + has header
        is_split = not has_total and has_numeric_data and first_row_is_header
        
        return is_split
        
    except Exception as e:
        print(f"    Warning in split detection: {e}")
        return False

def smart_merge_split_tables(contingent_tables):
    """
    HYBRID SOLUTION - PART 3
    Merge tables that appear to be split across pages
    """
    if not contingent_tables:
        return []
    
    print("\n" + "="*60)
    print("DETECTING AND MERGING SPLIT TABLES")
    print("="*60)
    
    merged_results = []
    skip_indices = set()
    
    for i in range(len(contingent_tables)):
        if i in skip_indices:
            continue
        
        current_table = contingent_tables[i]
        current_df = current_table['df']
        
        # Check if current table looks incomplete
        current_is_split = detect_split_table(current_df)
        
        if current_is_split and i + 1 < len(contingent_tables):
            next_table = contingent_tables[i + 1]
            next_df = next_table['df']
            
            # Check merge conditions
            same_columns = list(current_df.columns) == list(next_df.columns)
            consecutive_pages = (next_table['page_num'] - current_table['page_num']) <= 1
            
            if same_columns and consecutive_pages:
                # MERGE THEM
                print(f"  ✓ Merging: {current_table['sheet_name']} + {next_table['sheet_name']}")
                print(f"    Reason: Split table detected (no total row in first table)")
                
                merged_df = pd.concat([current_df, next_df], ignore_index=True)
                
                merged_results.append({
                    'df': merged_df,
                    'page_num': current_table['page_num'],
                    'table_num': current_table['table_num'],
                    'sheet_name': f"MERGED_{current_table['sheet_name']}_and_{next_table['sheet_name']}",
                    'filename': f"MERGED_{current_table['sheet_name']}.xlsx"
                })
                
                skip_indices.add(i + 1)
                continue
        
        # If not merged, keep as is
        merged_results.append(current_table)
    
    print(f"\n✓ Merge complete: {len(contingent_tables)} tables → {len(merged_results)} tables")
    return merged_results

def add_total_to_table(df, sheet_name):
    """
    Convert last column to integer and add total row if not present.
    """
    try:
        if df.empty or len(df.columns) == 0:
            print(f"  ⚠ Skipping totals for {sheet_name}: Empty dataframe")
            return df
        
        last_col_name = df.columns[-1]
        
        has_total = False
        if len(df) > 0 and len(df.columns) > 0:
            has_total = any('total' in str(val).lower() for val in df.iloc[:, 0])
        
        if has_total:
            print(f"  ℹ Total row already exists in {sheet_name}")
            return df
        
        df[last_col_name] = df[last_col_name].astype(str).str.replace(',', '')
        df[last_col_name] = df[last_col_name].str.replace('₹', '')
        df[last_col_name] = df[last_col_name].str.replace('$', '')
        df[last_col_name] = df[last_col_name].str.strip()
        
        df[last_col_name] = pd.to_numeric(df[last_col_name], errors='coerce')
        
        total_value = df[last_col_name].sum()
        
        total_row = pd.DataFrame([['Total'] + [''] * (len(df.columns) - 2) + [total_value]], 
                                 columns=df.columns)
        
        df = pd.concat([df, total_row], ignore_index=True)
        
        print(f"  ✓ Added total row to {sheet_name}: {total_value:,.2f}")
        
        return df
        
    except Exception as e:
        print(f"  ⚠ Error adding total to {sheet_name}: {e}")
        return df

def create_table_with_context(OUTPUT_PDF_PATH):
    """
    HYBRID SOLUTION - COMPLETE PIPELINE
    Extract tables → Merge split tables → Add totals → Save
    """
    result = get_docling_results(OUTPUT_PDF_PATH)
    full_markdown = result.document.export_to_markdown()
    
    print("\n" + "="*60)
    print("STEP 1: EXTRACTING TABLES")
    print("="*60)
    
    contingent_tables = []
    
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

        classification_result = classifyTable_with_context_check(
            table_df.to_markdown(), 
            table_ix, 
            full_markdown
        )
        
        print(f"\nClassification result for {sheet_name}: {classification_result}")

        if "true" in classification_result.lower():
            contingent_tables.append({
                'df': table_df,
                'page_num': current_page_num,
                'table_num': current_page_table_count,
                'sheet_name': sheet_name,
                'filename': f"{sheet_name}.xlsx"
            })
            print(f"  ✓ Extracted: {sheet_name}")

        previous_page_num = current_page_num
    
    if not contingent_tables:
        print("\n⚠ No contingent liability tables found")
        return
    
    # STEP 2: Smart merge split tables
    merged_tables = smart_merge_split_tables(contingent_tables)
    
    # STEP 3: Add totals and save
    print("\n" + "="*60)
    print("STEP 2: ADDING TOTALS AND SAVING")
    print("="*60)
    
    for table_info in merged_tables:
        table_df = add_total_to_table(table_info['df'], table_info['sheet_name'])
        
        filename = table_info['filename']
        table_df.to_excel(filename, index=False)
        print(f"✓ Saved: {filename}")

# Main execution
if __name__ == "__main__":
    print("Starting PDF processing with HYBRID MODEL...")
    
    # Step 1: Extract relevant pages (creates OUTPUT_PDF_PATH with fewer pages)
    relevant_pages = extract_cg(PDF_PATH)
    
    if relevant_pages:
        print(f"\n✓ Found relevant pages: {relevant_pages}")
        print(f"✓ Created filtered PDF: {OUTPUT_PDF_PATH}")
        
        # Step 2: Split the filtered PDF pages vertically (left/right separation)
        split_pdf_page_vertically(OUTPUT_PDF_PATH, SPLIT_PDF_PATH)
        
        # Step 3: Process the split PDF with smart merging and totals
        print("\n" + "="*60)
        print("PROCESSING SPLIT PDF")
        print("="*60)
        create_table_with_context(SPLIT_PDF_PATH)
        
        print("\n" + "="*60)
        print("✅ HYBRID PROCESSING COMPLETED!")
        print("="*60)
    else:
        print("No contingent liability tables found.")
