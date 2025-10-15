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

# Configuration
PDF_PATH = r"test_pdf\\LnT-AR.pdf"
OUTPUT_PDF_PATH = r"LnT-AR_new.pdf"

def load_pdf_pages(pdf_path):
    """Load a PDF file and return its content as a list of strings, each representing a page."""
    pdf_document = fitz.open(pdf_path)
    pages = []
    for page in range(len(pdf_document)):
        text = pdf_document[page].get_text("text")
        pages.append({"page_num": page, "text": text})
    return pages, pdf_document

def keyword_prefilter(pages):
    """More flexible keyword filtering for contingent liabilities"""
    pattern = re.compile(r"\bcontingent\s+liabilit(y|ies)\b", re.IGNORECASE)
    return [p for p in pages if pattern.search(p['text'])]

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

def stage_2_classify(page_text):
    """Classify the page text using Llama model for verification."""
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
    """Enhanced classification that considers document context"""
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
            llm_result = parse_llama_json_response(response)
            
            if isinstance(llm_result, dict):
                if result["currency"] == "Unknown" and "currency" in llm_result:
                    result["currency"] = llm_result["currency"]
                if result["unit"] == "Unknown" and "unit" in llm_result:
                    result["unit"] = llm_result["unit"]
        except Exception as e:
            print(f"Warning: LLM extraction failed: {e}")
    
    return result

def add_total_to_table(df, sheet_name):
    """Convert last column to integer and add total row if not present."""
    try:
        if df.empty or len(df.columns) == 0:
            print(f"  ⚠ Skipping totals for {sheet_name}: Empty dataframe")
            return df
        
        # Get the last column
        last_col_name = df.columns[-1]
        
        # Check if total row already exists (check first column for 'total')
        has_total = False
        if len(df) > 0 and len(df.columns) > 0:
            has_total = any('total' in str(val).lower() for val in df.iloc[:, 0])
        
        if has_total:
            print(f"  ℹ Total row already exists in {sheet_name}")
            return df
        
        # Convert last column to numeric, handling various formats
        df[last_col_name] = df[last_col_name].astype(str).str.replace(',', '')
        df[last_col_name] = df[last_col_name].str.replace('₹', '')
        df[last_col_name] = df[last_col_name].str.replace('$', '')
        df[last_col_name] = df[last_col_name].str.strip()
        
        # Convert to numeric (coerce errors to NaN)
        df[last_col_name] = pd.to_numeric(df[last_col_name], errors='coerce')
        
        # Calculate sum (excluding NaN values)
        total_value = df[last_col_name].sum()
        
        # Create total row
        total_row = pd.DataFrame([['Total'] + [''] * (len(df.columns) - 2) + [total_value]], 
                                 columns=df.columns)
        
        # Append total row
        df = pd.concat([df, total_row], ignore_index=True)
        
        print(f"  ✓ Added total row to {sheet_name}: {total_value:,.2f}")
        
        return df
        
    except Exception as e:
        print(f"  ⚠ Error adding total to {sheet_name}: {e}")
        return df

def clean_illegal_chars(df):
    """Remove illegal characters from dataframe"""
    return df.applymap(
        lambda x: ILLEGAL_CHARACTERS_RE.sub("", str(x)) if isinstance(x, str) else x
    )

def get_docling_pipeline():
    """Get docling pipeline."""
    try:
        print("Step 1: Creating pipeline options....")
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            table_structure_options=dict(do_cell_matching=True, mode=TableFormerMode.ACCURATE)
        )
        print("Step 2: Creating document converter...")
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        print("Step 3: Setting debug flag...")
        settings.debug.profile_pipeline_timings = True
        return doc_converter
    
    except Exception as e:
        print(f"Exception occurred while getting the docling pipeline: {e}")
        return None

doc_converter_global = get_docling_pipeline()

def get_docling_results(INPUT_PDF_SOURCE):
    """Get docling results for PDF"""
    if doc_converter_global is None:
        raise RuntimeError("Docling converter not initialized.")
    
    result = doc_converter_global.convert(INPUT_PDF_SOURCE)
    return result

def extract_cg(pdf_path):
    """Main function to extract Contingent Liabilities from a PDF using Llama."""
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

def create_table_with_full_features(OUTPUT_PDF_PATH):
    """
    Extract and save tables with:
    1. Context-aware classification
    2. Currency and unit extraction
    3. Total row calculation
    """
    pdf_name = os.path.splitext(os.path.basename(OUTPUT_PDF_PATH))[0]
    output_folder = f"{pdf_name}_tables"
    os.makedirs(output_folder, exist_ok=True)

    result = get_docling_results(OUTPUT_PDF_PATH)
    full_markdown = result.document.export_to_markdown()
    
    print("\n" + "="*60)
    print("EXTRACTING TABLES WITH FULL FEATURES")
    print("="*60)
    
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
        table_df.columns = [ILLEGAL_CHARACTERS_RE.sub("", str(col)) for col in table_df.columns]
        table_df = clean_illegal_chars(table_df)

        # Use context-aware classification
        classification_result = classifyTable_with_context_check(
            table_df.to_markdown(), 
            table_ix, 
            full_markdown
        )
        
        print(f"\nClassification result for {sheet_name}: {classification_result}")

        if "true" in classification_result.lower():
            # Extract currency and unit
            currency_unit = extract_currency_and_unit_for_table(table_df, full_markdown)
            print(f"  Currency: {currency_unit['currency']}, Unit: {currency_unit['unit']}")
            
            # Add total to the table
            table_df = add_total_to_table(table_df, sheet_name)
            
            # Add currency and unit as columns
            table_df['Currency'] = currency_unit['currency']
            table_df['Unit'] = currency_unit['unit']
            
            # Save to folder with currency and unit in filename
            filename = f"{sheet_name}_{currency_unit['currency']}_{currency_unit['unit']}.xlsx"
            filepath = os.path.join(output_folder, filename)
            table_df.to_excel(filepath, index=False)
            
            print(f"  ✓ Saved: {filepath}")
        
        previous_page_num = current_page_num

if __name__ == "__main__":
    print("="*60)
    print("CONTINGENT LIABILITIES EXTRACTION - FULL FEATURED VERSION")
    print("="*60)
    
    # Step 1: Extract relevant pages
    print("\n[STEP 1] Extracting relevant pages...")
    relevant_pages = extract_cg(PDF_PATH)
    
    if relevant_pages:
        print(f"\n✓ Found {len(relevant_pages)} relevant page(s): {relevant_pages}")
        
        # Step 2: Extract tables with all features
        print("\n[STEP 2] Processing tables with full features...")
        create_table_with_full_features(OUTPUT_PDF_PATH)
        
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
    else:
        print("\n❌ No contingent liability tables found.")
