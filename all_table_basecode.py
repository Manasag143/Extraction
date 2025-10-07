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

def calculate_column_similarity(cols1, cols2):
    """
    Calculate similarity between two sets of columns.
    Returns similarity score between 0 and 1.
    """
    if not cols1 or not cols2:
        return 0.0
    
    # Normalize column names
    norm_cols1 = set([str(col).strip().lower() for col in cols1])
    norm_cols2 = set([str(col).strip().lower() for col in cols2])
    
    # Calculate Jaccard similarity
    intersection = len(norm_cols1.intersection(norm_cols2))
    union = len(norm_cols1.union(norm_cols2))
    
    return intersection / union if union > 0 else 0.0

def should_merge_tables(table1_info, table2_info):
    """
    Determine if two tables should be merged based on multiple criteria.
    
    Criteria:
    1. Page proximity (same page or consecutive pages)
    2. Column structure similarity (>= 70%)
    """
    # Check 1: Page proximity
    page_diff = table2_info['page_num'] - table1_info['page_num']
    if page_diff > 1:  # Not on same or consecutive pages
        return False
    
    # Check 2: Column similarity
    similarity = calculate_column_similarity(
        table1_info['df'].columns,
        table2_info['df'].columns
    )
    
    if similarity < 0.7:  # Less than 70% similar
        return False
    
    return True

def merge_tables(contingent_tables):
    """
    Identify and merge related tables.
    Returns list of merge groups.
    """
    if not contingent_tables:
        return []
    
    print("\n" + "="*60)
    print("ANALYZING TABLES FOR MERGING")
    print("="*60)
    
    merged_indices = set()
    merge_groups = []
    
    i = 0
    while i < len(contingent_tables):
        if i in merged_indices:
            i += 1
            continue
        
        current_group = [i]
        j = i + 1
        
        # Try to merge with subsequent tables
        while j < len(contingent_tables):
            if j in merged_indices:
                j += 1
                continue
            
            # Check if current group's last table can merge with table j
            last_in_group = current_group[-1]
            
            if should_merge_tables(contingent_tables[last_in_group], contingent_tables[j]):
                current_group.append(j)
                merged_indices.add(j)
                print(f"  ✓ Table on page {contingent_tables[j]['page_num']} will merge with page {contingent_tables[last_in_group]['page_num']}")
            
            j += 1
        
        if len(current_group) > 1:
            merge_groups.append(current_group)
            for idx in current_group:
                merged_indices.add(idx)
        
        i += 1
    
    return merge_groups

def save_merged_tables(merge_groups, contingent_tables):
    """
    Create and save merged Excel files.
    """
    if not merge_groups:
        print("\n✓ No tables need to be merged (all tables are independent)")
        return
    
    print(f"\n✓ Found {len(merge_groups)} group(s) of tables to merge")
    print("\n" + "="*60)
    print("CREATING MERGED FILES")
    print("="*60)
    
    for group_idx, group in enumerate(merge_groups):
        print(f"\nMerging Group {group_idx + 1}:")
        
        # Collect dataframes to merge
        dfs_to_merge = []
        page_nums = []
        sheet_names = []
        
        for table_idx in group:
            table_info = contingent_tables[table_idx]
            dfs_to_merge.append(table_info['df'])
            page_nums.append(table_info['page_num'])
            sheet_names.append(table_info['sheet_name'])
            print(f"  - {table_info['sheet_name']}: {len(table_info['df'])} rows")
        
        # Merge dataframes
        merged_df = pd.concat(dfs_to_merge, ignore_index=True)
        
        # Create filename
        first_page = min(page_nums)
        last_page = max(page_nums)
        
        if first_page == last_page:
            filename = f"MERGED_Page_{first_page}_({len(group)}_tables).xlsx"
        else:
            filename = f"MERGED_Pages_{first_page}_to_{last_page}.xlsx"
        
        merged_df.to_excel(filename, index=False)
        
        print(f"  ✓ Saved: {filename}")
        print(f"    Total rows: {len(merged_df)}")

def create_table_with_context(OUTPUT_PDF_PATH):
    """
    STEP 1: Extract and save all individual tables
    STEP 2: Analyze and create merged tables
    """
    result = get_docling_results(OUTPUT_PDF_PATH)
    full_markdown = result.document.export_to_markdown()
    
    print("\n" + "="*60)
    print("STEP 1: EXTRACTING AND SAVING INDIVIDUAL TABLES")
    print("="*60)
    
    # Store contingent liability tables for later merging
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

        # Use context-aware classification
        classification_result = classifyTable_with_context_check(
            table_df.to_markdown(), 
            table_ix, 
            full_markdown
        )
        
        print(f"\nClassification result for {sheet_name}: {classification_result}")

        if "true" in classification_result.lower():
            # Save individual table
            filename = f"{sheet_name}.xlsx"
            table_df.to_excel(filename, index=False)
            print(f"✓ Saved individual table: {filename}")
            
            # Store for merging analysis
            contingent_tables.append({
                'df': table_df,
                'page_num': current_page_num,
                'table_num': current_page_table_count,
                'sheet_name': sheet_name,
                'filename': filename
            })

        previous_page_num = current_page_num
    
    # STEP 2: Analyze and merge tables
    if contingent_tables:
        merge_groups = merge_tables(contingent_tables)
        save_merged_tables(merge_groups, contingent_tables)

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

# Main execution with different modes
if __name__ == "__main__":
    print("Starting PDF processing with Enhanced Llama LLM...")
    
    # Step 1: Extract relevant pages
    relevant_pages = extract_cg(PDF_PATH)
    
    if relevant_pages:
        print(f"Found relevant pages: {relevant_pages}")

        print("\n=== RUNNING DEBUG TABLE CLASSIFICATION ===")
        create_table_debug(OUTPUT_PDF_PATH)

        print("\n=== RUNNING PRODUCTION MODE ===")
        create_table_with_context(OUTPUT_PDF_PATH)
        
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETED!")
        print("="*60)
    else:
        print("No contingent liability tables found.")
