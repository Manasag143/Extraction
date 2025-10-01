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
            headers = {'token': '0e7', 'Content-Type': 'application/json'}
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
SPLIT_PDF_PATH = r"indus towers AR_new_split.pdf"

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
4. ACCEPT partial tables that appear to be cut off at page boundaries (incomplete rows/totals are OK)

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
- ACCEPT partial/incomplete tables (they may continue on next page)
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

def detect_double_page_layout(pdf_path):
    """
    Check if PDF has double-page layout by checking aspect ratio
    """
    pdf = fitz.open(pdf_path)
    page = pdf[0]
    aspect_ratio = page.rect.width / page.rect.height
    pdf.close()
    
    # Landscape book spread typically has ratio > 1.4
    print(f"[INFO] Page aspect ratio: {aspect_ratio:.2f}")
    return aspect_ratio > 1.4

def split_double_pages(pdf_path, output_path):
    """
    Split each physical page into left and right logical pages
    """
    try:
        pdf = fitz.open(pdf_path)
        new_pdf = fitz.open()
        
        print(f"[INFO] Splitting {len(pdf)} pages...")
        
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            rect = page.rect
            mid_x = rect.width / 2
            
            # Create left page
            left_rect = fitz.Rect(0, 0, mid_x, rect.height)
            left_page = new_pdf.new_page(width=mid_x, height=rect.height)
            left_page.show_pdf_page(left_page.rect, pdf, page_num, clip=left_rect)
            
            # Create right page
            right_rect = fitz.Rect(mid_x, 0, rect.width, rect.height)
            right_page = new_pdf.new_page(width=mid_x, height=rect.height)
            right_page.show_pdf_page(right_page.rect, pdf, page_num, clip=right_rect)
        
        print(f"[INFO] Created {len(new_pdf)} pages in split PDF")
        new_pdf.save(output_path)
        new_pdf.close()
        pdf.close()
        
        # Verify the file was created
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✓ Split PDF saved successfully: {output_path} ({file_size} bytes)")
            return True
        else:
            print(f"✗ ERROR: Split PDF was NOT saved to {output_path}")
            return False
            
    except Exception as e:
        print(f"✗ ERROR in split_double_pages: {e}")
        import traceback
        traceback.print_exc()
        return False

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

def clean_illegal_chars(df):
    """
    Simplified version - clean illegal characters from dataframe
    """
    for col in df.columns:
        df[col] = df[col].apply(lambda x: ILLEGAL_CHARACTERS_RE.sub("", str(x)) if isinstance(x, str) else x)
    return df

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

def create_table_with_context(processing_pdf_path):
    """
    Create tables with context awareness and merge consecutive similar tables
    """
    pdf_name = os.path.splitext(os.path.basename(processing_pdf_path))[0]
    output_folder = f"{pdf_name}_tables"
    os.makedirs(output_folder, exist_ok=True)

    result = get_docling_results(processing_pdf_path)
    full_markdown = result.document.export_to_markdown()
    
    all_tables = []
    previous_page_num = None
    current_page_table_count = 0
    
    # First pass: Extract all tables with metadata
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
        
        # Clean column names first
        table_df.columns = [ILLEGAL_CHARACTERS_RE.sub("", str(col)) for col in table_df.columns]
        
        # Make columns unique by adding suffixes to duplicates
        cols = list(table_df.columns)
        seen = {}
        for i, col in enumerate(cols):
            if col in seen:
                seen[col] += 1
                cols[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
        table_df.columns = cols
        
        # Now clean illegal characters from cell values
        table_df = clean_illegal_chars(table_df)
        
        all_tables.append({
            'df': table_df,
            'page_num': current_page_num,
            'table_num': current_page_table_count,
            'sheet_name': sheet_name,
            'original_table': table
        })
        
        previous_page_num = current_page_num
    
    # Second pass: Merge consecutive tables with similar structure
    merged_tables = []
    skip_next = False
    
    for i in range(len(all_tables)):
        if skip_next:
            skip_next = False
            continue
        
        current = all_tables[i]
        
        # Check if we should merge with next table
        if i + 1 < len(all_tables):
            next_table = all_tables[i + 1]
            
            # Check if tables should be merged
            # Condition 1: Consecutive pages or same page
            page_diff = abs(next_table['page_num'] - current['page_num'])
            
            # Condition 2: Column structure similarity
            cols1 = set(current['df'].columns)
            cols2 = set(next_table['df'].columns)
            if len(cols1) > 0 and len(cols2) > 0:
                column_similarity = len(cols1 & cols2) / max(len(cols1), len(cols2))
            else:
                column_similarity = 0
            
            # Merge if: consecutive pages AND similar columns (>70% match)
            if page_diff <= 1 and column_similarity > 0.7:
                print(f"[INFO] Merging {current['sheet_name']} with {next_table['sheet_name']}")
                
                # Merge the dataframes - reset index to avoid duplicate index issues
                merged_df = pd.concat([current['df'], next_table['df']], ignore_index=True)
                
                merged_tables.append({
                    'df': merged_df,
                    'page_num': current['page_num'],
                    'table_num': current['table_num'],
                    'sheet_name': f"{current['sheet_name']}_merged",
                    'is_merged': True
                })
                skip_next = True
                continue
        
        # No merge, add as-is
        merged_tables.append(current)
    
    print(f"[INFO] Reduced {len(all_tables)} tables to {len(merged_tables)} after merging")
    
    # Third pass: Classify and save merged tables
    for table_info in merged_tables:
        table_df = table_info['df']
        sheet_name = table_info['sheet_name']
        
        # Use context-aware classification
        classification_result = classifyTable_with_context_check(
            table_df.to_markdown(), 
            0, 
            full_markdown
        )
        
        print(f"Classification result for {sheet_name}: {classification_result}")

        if "true" in classification_result.lower():
            print(f"✓ Saving contingent liabilities table: {sheet_name}")
            filepath = os.path.join(output_folder, f"{sheet_name}.xlsx")
            table_df.to_excel(filepath, index=False)
            print(f"  Saved to: {filepath}")

def create_table_debug(processing_pdf_path):
    """
    Debug version of create_table to see what's happening with concatenation
    """
    pdf_name = os.path.splitext(os.path.basename(processing_pdf_path))[0]
    output_folder = f"{pdf_name}_debug_tables"
    os.makedirs(output_folder, exist_ok=True)

    result = get_docling_results(processing_pdf_path)
    
    all_tables = []
    previous_page_num = None
    current_page_table_count = 0
    
    # First pass: Extract all tables with metadata
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
        table_df.columns = [ILLEGAL_CHARACTERS_RE.sub("", str(col)) for col in table_df.columns]
        
        # Make columns unique
        cols = list(table_df.columns)
        seen = {}
        for i, col in enumerate(cols):
            if col in seen:
                seen[col] += 1
                cols[i] = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
        table_df.columns = cols
        
        table_df = clean_illegal_chars(table_df)
        
        all_tables.append({
            'df': table_df,
            'page_num': current_page_num,
            'table_num': current_page_table_count,
            'sheet_name': sheet_name
        })
        
        previous_page_num = current_page_num
    
    # Second pass: Merge consecutive tables
    merged_tables = []
    skip_next = False
    
    for i in range(len(all_tables)):
        if skip_next:
            skip_next = False
            continue
        
        current = all_tables[i]
        
        if i + 1 < len(all_tables):
            next_table = all_tables[i + 1]
            
            page_diff = abs(next_table['page_num'] - current['page_num'])
            cols1 = set(current['df'].columns)
            cols2 = set(next_table['df'].columns)
            
            if len(cols1) > 0 and len(cols2) > 0:
                column_similarity = len(cols1 & cols2) / max(len(cols1), len(cols2))
            else:
                column_similarity = 0
            
            if page_diff <= 1 and column_similarity > 0.7:
                print(f"[INFO] Merging {current['sheet_name']} with {next_table['sheet_name']}")
                merged_df = pd.concat([current['df'], next_table['df']], ignore_index=True)
                
                merged_tables.append({
                    'df': merged_df,
                    'sheet_name': f"{current['sheet_name']}_merged"
                })
                skip_next = True
                continue
        
        merged_tables.append(current)
    
    print(f"[INFO] Reduced {len(all_tables)} tables to {len(merged_tables)} after merging")
    
    # Classify and save
    for table_info in merged_tables:
        table_df = table_info['df']
        sheet_name = table_info['sheet_name']
        
        classification_result = classifyTable_debug(table_df.to_markdown())
        print(f"Classification result for {sheet_name}: {classification_result}")

        if "true" in classification_result.lower():
            print(f"✓ Saving table: {sheet_name}")
            filepath = os.path.join(output_folder, f"{sheet_name}.xlsx")
            table_df.to_excel(filepath, index=False)
            print(f"  Saved to: {filepath}")

# Main execution with different modes
if __name__ == "__main__":
    print("Starting PDF processing with Enhanced Llama LLM...")
    
    # Step 1: Extract relevant pages from ORIGINAL PDF
    relevant_pages = extract_cg(PDF_PATH)
    
    if relevant_pages:
        print(f"\n✓ Found relevant pages: {relevant_pages}")
        print(f"✓ Created reduced PDF: {OUTPUT_PDF_PATH}")
        
        # Step 2: Check if the REDUCED PDF has double-page layout
        print(f"\n=== CHECKING REDUCED PDF FOR DOUBLE-PAGE LAYOUT ===")
        if detect_double_page_layout(OUTPUT_PDF_PATH):
            print("[INFO] Double-page layout detected in reduced PDF. Splitting pages...")
            if split_double_pages(OUTPUT_PDF_PATH, SPLIT_PDF_PATH):
                processing_pdf = SPLIT_PDF_PATH
                print(f"✓ Will process split PDF: {processing_pdf}")
            else:
                print("[WARN] Split failed, using original reduced PDF")
                processing_pdf = OUTPUT_PDF_PATH
        else:
            print("[INFO] Normal single-page layout detected. No splitting needed.")
            processing_pdf = OUTPUT_PDF_PATH
        
        # Step 3: Process tables from the appropriate PDF
        print(f"\n=== PROCESSING TABLES FROM: {processing_pdf} ===")

        print("\n=== RUNNING DEBUG TABLE CLASSIFICATION ===")
        create_table_debug(processing_pdf)

        print("\n=== RUNNING PRODUCTION MODE ===")
        create_table_with_context(processing_pdf)
        
        print("\n✓ Processing completed!")
    else:
        print("No contingent liability tables found.")
