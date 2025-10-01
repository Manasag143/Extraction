import json
import re
import ast
import requests
from typing import Optional, Any
import warnings
warnings.filterwarnings("ignore")
import os
import fitz
import re
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
            headers = {'token': 'd2', 'Content-Type': 'application/json'}
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

PDF_PATH = r"test_pdf\\LnT-AR.pdf"
OUTPUT_PDF_PATH = r"LnT-AR_new.pdf"
SPLIT_PDF_PATH = r"LnT-AR_split.pdf"

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
        print("Step 1: Creating pipeline options....")
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            table_structure_options=dict(do_cell_matching=True, mode=TableFormerMode.ACCURATE)
        )
        print("step 2: Creating document converter...")
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        print("step 3: Setting debug flag...")
        settings.debug.profile_pipeline_timings = True
        return doc_converter
    
    except Exception as e:
        print(f"Exception occurred while getting the docling pipeline: {e}")

doc_converter_global = get_docling_pipeline()

def get_docling_results(INPUT_PDF_SOURCE):
    if doc_converter_global is None:
        raise RuntimeError(
            "Docling converts not initialized."
        )
    
    result = doc_converter_global.convert(INPUT_PDF_SOURCE)
    return result

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
    pdf = fitz.open(pdf_path)
    new_pdf = fitz.open()
    
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
    
    new_pdf.save(output_path)
    new_pdf.close()
    pdf.close()
    
    print(f"✓ Split double pages: {pdf_path} → {output_path}")

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

def extract_currency_and_unit_for_table(table_df, full_markdown: str) -> dict:
    """
    Extract both currency and unit for a table using hybrid approach.
    Returns dict like {"currency": "INR", "unit": "Crores"} or {"currency": "Unknown", "unit": "Unknown"}
    """
    
    result = {"currency": "Unknown", "unit": "Unknown"}
    
    # ===== STEP 1: Check column headers (highest priority) =====
    for col in table_df.columns:
        col_str = str(col).lower()
        
        # Check for currency in headers
        if 'inr' in col_str or '₹' in col_str or 'rupee' in col_str or 'rs.' in col_str or 'rs ' in col_str:
            result["currency"] = "INR"
        elif 'usd' in col_str or '$' in col_str or 'dollar' in col_str:
            result["currency"] = "USD"
        elif 'eur' in col_str or '€' in col_str or 'euro' in col_str:
            result["currency"] = "EUR"
        elif 'gbp' in col_str or '£' in col_str or 'pound' in col_str:
            result["currency"] = "GBP"
        elif 'jpy' in col_str or '¥' in col_str or 'yen' in col_str:
            result["currency"] = "JPY"
        
        # Check for unit in headers
        if 'crore' in col_str or 'cr.' in col_str or 'cr ' in col_str:
            result["unit"] = "Crores"
        elif 'lakh' in col_str or 'lac' in col_str:
            result["unit"] = "Lakhs"
        elif 'million' in col_str or 'mn' in col_str:
            result["unit"] = "Millions"
        elif 'billion' in col_str or 'bn' in col_str:
            result["unit"] = "Billions"
        elif 'thousand' in col_str or 'k ' in col_str:
            result["unit"] = "Thousands"
        elif 'trillion' in col_str or 'tn' in col_str:
            result["unit"] = "Trillions"
    
    # ===== STEP 2: Get text around table (look for patterns) =====
    if len(table_df) > 0 and len(table_df.columns) > 0:
        first_cell = str(table_df.iloc[0, 0])[:50]
        
        if first_cell in full_markdown:
            pos = full_markdown.find(first_cell)
            # Get 500 chars before and 200 after table
            context_before = full_markdown[max(0, pos-500):pos]
            context_after = full_markdown[pos:min(len(full_markdown), pos+200)]
            context = context_before + context_after
            
            # Currency patterns
            currency_patterns = [
                (r'(?:in|of)?\s*(?:Rs\.?|INR|₹)\s', 'INR'),
                (r'(?:in|of)?\s*(?:USD|\$|US\$)\s', 'USD'),
                (r'(?:in|of)?\s*(?:EUR|€)\s', 'EUR'),
                (r'(?:in|of)?\s*(?:GBP|£)\s', 'GBP'),
                (r'(?:in|of)?\s*(?:JPY|¥)\s', 'JPY'),
                (r'Indian\s+Rupees?', 'INR'),
                (r'US\s+Dollars?', 'USD'),
            ]
            
            # Unit patterns
            unit_patterns = [
                (r'\(.*?in.*?(?:Rs\.?|INR|₹)\s*Crores?\)', 'Crores'),
                (r'\(.*?in.*?(?:Rs\.?|INR|₹)\s*Lakhs?\)', 'Lakhs'),
                (r'\(.*?in.*?(?:Rs\.?|INR|₹)\s*Millions?\)', 'Millions'),
                (r'\(.*?in.*?(?:Rs\.?|INR|₹)\s*Billions?\)', 'Billions'),
                (r'\(.*?in.*?(?:Rs\.?|INR|₹)\s*Thousands?\)', 'Thousands'),
                (r'Amount.*?in.*?Crores?', 'Crores'),
                (r'Amount.*?in.*?Lakhs?', 'Lakhs'),
                (r'Amount.*?in.*?Millions?', 'Millions'),
                (r'Amount.*?in.*?Billions?', 'Billions'),
                (r'Amount.*?in.*?Thousands?', 'Thousands'),
                (r'(?:in|of)\s+Crores?', 'Crores'),
                (r'(?:in|of)\s+Lakhs?', 'Lakhs'),
                (r'(?:in|of)\s+Millions?', 'Millions'),
                (r'(?:in|of)\s+Billions?', 'Billions'),
            ]
            
            if result["currency"] == "Unknown":
                for pattern, currency in currency_patterns:
                    if re.search(pattern, context, re.IGNORECASE):
                        result["currency"] = currency
                        break
            
            if result["unit"] == "Unknown":
                for pattern, unit in unit_patterns:
                    if re.search(pattern, context, re.IGNORECASE):
                        result["unit"] = unit
                        break
    
    # ===== STEP 3: Check page header =====
    header = full_markdown[:1000]
    
    if result["currency"] == "Unknown":
        if re.search(r'All.*?amount.*?in.*?(?:Rs\.?|INR|₹|Indian\s+Rupees?)', header, re.IGNORECASE):
            result["currency"] = "INR"
        elif re.search(r'All.*?amount.*?in.*?(?:USD|\$|US\s+Dollars?)', header, re.IGNORECASE):
            result["currency"] = "USD"
    
    if result["unit"] == "Unknown":
        if re.search(r'All.*?amount.*?in.*?Crores?', header, re.IGNORECASE):
            result["unit"] = "Crores"
        elif re.search(r'All.*?amount.*?in.*?Lakhs?', header, re.IGNORECASE):
            result["unit"] = "Lakhs"
        elif re.search(r'All.*?amount.*?in.*?Millions?', header, re.IGNORECASE):
            result["unit"] = "Millions"
        elif re.search(r'All.*?amount.*?in.*?Billions?', header, re.IGNORECASE):
            result["unit"] = "Billions"
    
    # ===== STEP 4: If still not found, ask LLM =====
    if result["currency"] == "Unknown" or result["unit"] == "Unknown":
        prompt = f"""You are a financial document analyzer. Extract the CURRENCY and UNIT from this table.

CONTEXT (surrounding text):
{full_markdown[:1500]}

TABLE (first 3 rows):
{table_df.head(3).to_markdown(index=False)}

TASK:
1. Identify the CURRENCY used (e.g., INR, USD, EUR, GBP, JPY, etc.)
   - Look for symbols: ₹, $, €, £, ¥
   - Look for codes: INR, USD, EUR, GBP, JPY
   - Look for words: Rupees, Dollars, Euros, Pounds, Yen
   - Common in Indian reports: INR/Rs./Rupees

2. Identify the UNIT/SCALE of amounts (e.g., Crores, Lakhs, Millions, Billions, Thousands, etc.)
   - In India: Crores (10^7), Lakhs (10^5)
   - International: Millions (10^6), Billions (10^9), Thousands (10^3), Trillions (10^12)
   - Look in: table headers, column names, surrounding text, parentheses

IMPORTANT RULES:
- Return ONLY valid JSON format
- If you cannot find currency, return "Unknown"
- If you cannot find unit, return "Unknown"
- Be precise - don't guess
- Check table headers, column names, and surrounding context carefully

Return ONLY this JSON format (no extra text):
{{
  "currency": "INR",
  "unit": "Crores"
}}

Examples of valid responses:
{{"currency": "USD", "unit": "Millions"}}
{{"currency": "INR", "unit": "Lakhs"}}
{{"currency": "EUR", "unit": "Billions"}}
{{"currency": "Unknown", "unit": "Thousands"}}

Your response:"""
        
        try:
            response = llama_client(prompt).strip()
            
            # Try to parse JSON response
            llm_result = parse_llama_json_response(response)
            
            if isinstance(llm_result, dict):
                if result["currency"] == "Unknown" and "currency" in llm_result:
                    result["currency"] = llm_result["currency"]
                if result["unit"] == "Unknown" and "unit" in llm_result:
                    result["unit"] = llm_result["unit"]
        except Exception as e:
            print(f"Warning: LLM extraction failed: {e}")
    
    return result


def create_table_with_context(OUTPUT_PDF_PATH):
    """
    Create tables with context awareness and merge consecutive similar tables
    """
    pdf_name = os.path.splitext(os.path.basename(OUTPUT_PDF_PATH))[0]
    output_folder = f"{pdf_name}_tables"
    os.makedirs(output_folder, exist_ok=True)

    result = get_docling_results(OUTPUT_PDF_PATH)
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
        table_df.columns = [ILLEGAL_CHARACTERS_RE.sub("", str(col)) for col in table_df.columns]
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
                
                # Merge the dataframes
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
            # Extract currency and unit
            currency_unit = extract_currency_and_unit_for_table(table_df, full_markdown)
            print(f"  Currency: {currency_unit['currency']}, Unit: {currency_unit['unit']}")
            
            # Add both as columns
            table_df['Currency'] = currency_unit['currency']
            table_df['Unit'] = currency_unit['unit']
            
            # Save to folder with currency and unit in filename
            filename = f"{sheet_name}_{currency_unit['currency']}_{currency_unit['unit']}.xlsx"
            filepath = os.path.join(output_folder, filename)
            table_df.to_excel(filepath, index=False)
            
            print(f"  ✓ Saved: {filepath}")

if __name__ == "__main__":
    print("Starting PDF processing with Enhanced Llama LLM...")
    
    # STEP 0: Check if double-page layout and split if needed
    if detect_double_page_layout(PDF_PATH):
        print("[INFO] Double-page layout detected. Splitting pages...")
        split_double_pages(PDF_PATH, SPLIT_PDF_PATH)
        processing_pdf = SPLIT_PDF_PATH
    else:
        print("[INFO] Normal single-page layout detected.")
        processing_pdf = PDF_PATH
    
    # Step 1: Extract relevant pages from split PDF
    relevant_pages = extract_cg(processing_pdf)
    
    if relevant_pages:
        print("\n=== RUNNING PRODUCTION MODE ===")
        create_table_with_context(OUTPUT_PDF_PATH)
        
        print("Processing completed!")
    else:
        print("No contingent liability tables found.")
