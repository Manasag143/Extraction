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
import time

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

llama_client = HostedLLM(endpoint="https://llmgateway.crisil.local/api/v1/llm")

PDF_PATH = r"test_pdf\\LnT-AR.pdf"
OUTPUT_PDF_PATH = r"result\\LnT-AR_new.pdf"

def load_pdf_pages(pdf_path):
    pdf_document = fitz.open(pdf_path)
    pages = []
    for page in range(len(pdf_document)):
        text = pdf_document[page].get_text("text")
        pages.append({"page_num": page, "text": text})
    return pages, pdf_document

def keyword_prefilter(pages):
    pattern = re.compile(r"\bcontingent\s+liabilit(y|ies)\b", re.IGNORECASE)
    return [p for p in pages if pattern.search(p['text'])]

def parse_llama_json_response(response_text):
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
    prompt = f"""
You are an expert Financial Analyst. You will be given the text content of a PDF page from an annual report.
Determine if the page contains a "Contingent Liabilities" table.

Rules:
1. ACCEPT if table title contains "Contingent Liabilities" OR "Contingent Liability"
2. ACCEPT if title is "Contingent Liabilities and Commitments" (combined table is OK)
3. REJECT if title is ONLY "Commitments" or ONLY "Guarantees" (without contingent liabilities)
4. REJECT tables titled only "Bank Guarantees", "Performance Guarantees", etc.
5. Must be an actual table, not just a reference.

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
    prompt = f"""
You are verifying a page for accuracy.
Confirm if the page contains a "Contingent Liabilities" table.

Requirements:
- Must include "Contingent Liabilities" or "Contingent Liability" in the heading
- Combined tables like "Contingent Liabilities and Commitments" are ACCEPTABLE
- Tables with ONLY "Commitments" or ONLY "Guarantees" should be rejected
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
    prompt = f"""
You are analyzing a financial table from an annual report.

CONTEXT: Here's the full document markdown to understand the table's position:
{full_document_markdown[:3000]}

TABLE TO CLASSIFY:
{table_markdown}

TASK: Determine if this specific table contains contingent liabilities information.

ANALYSIS STEPS:
1. Look at the full document context above - ACCEPT if table is under a section titled:
   - "Contingent Liabilities" or "Contingent Liability"
   - "Contingent Liabilities and Commitments" (combined table is OK)
   - "Legal Proceedings"  
   - "Litigation"
   
2. REJECT if the table appears under sections titled ONLY:
   - "Guarantees" (without contingent liabilities)
   - "Bank Guarantees" (without contingent liabilities)
   - "Performance Guarantees" (without contingent liabilities)
   - "Letters of Credit" (without contingent liabilities)
   - "Commitments" (without contingent liabilities)

3. Look at the table content itself:
   - Does it contain legal disputes, court cases, tax disputes?
   - Does it show amounts "not acknowledged as debt"?
   - Does it list uncertain future obligations?

IMPORTANT: 
- Combined tables like "Contingent Liabilities and Commitments" should be ACCEPTED
- Pure "Commitments" or "Guarantees" tables (without contingent liabilities) should be REJECTED

Respond ONLY with 'True' or 'False':
- True: If this table contains contingent liabilities (even if combined with commitments)
- False: If this table is ONLY about guarantees/commitments without contingent liabilities

Output: True or False
"""
    response = llama_client(prompt)
    return response.strip()

def fix_merged_columns(df):
    if df.empty or len(df.columns) < 2:
        return df
    
    df_fixed = df.copy()
    number_pattern = r'[\d,]+\.?\d*'
    
    for i in range(len(df_fixed)):
        first_cell = str(df_fixed.iloc[i, 0]).strip().lower()
        if 'particular' in first_cell or 'total' in first_cell:
            continue
        
        col1 = str(df_fixed.iloc[i, 0])
        col2 = str(df_fixed.iloc[i, 1]) if len(df_fixed.columns) > 1 else ''
        col3 = str(df_fixed.iloc[i, 2]) if len(df_fixed.columns) > 2 else ''
        
        numbers_in_col1 = re.findall(number_pattern, col1)
        numbers_in_col1 = [n for n in numbers_in_col1 if len(n.replace(',', '').replace('.', '')) >= 2]
        
        if numbers_in_col1:
            clean_text = col1
            for num in numbers_in_col1:
                clean_text = clean_text.replace(num, '')
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if not col2 or not any(c.isdigit() for c in col2):
                col2 = numbers_in_col1[0] if len(numbers_in_col1) > 0 else col2
            if len(numbers_in_col1) > 1 and (not col3 or not any(c.isdigit() for c in col3)):
                col3 = numbers_in_col1[1]
            
            col1 = clean_text
        
        if col2 and len(col2) > 3:
            has_letters = any(c.isalpha() for c in col2)
            has_digits = any(c.isdigit() for c in col2)
            
            if has_letters and has_digits:
                numbers_in_col2 = re.findall(number_pattern, col2)
                numbers_in_col2 = [n for n in numbers_in_col2 if len(n.replace(',', '').replace('.', '')) >= 2]
                
                if numbers_in_col2:
                    text_part = col2
                    for num in numbers_in_col2:
                        text_part = text_part.replace(num, '')
                    text_part = re.sub(r'\s+', ' ', text_part).strip()
                    
                    if col1.strip() and not col1.strip().endswith('.'):
                        col1 = col1.strip() + ' ' + text_part
                    else:
                        col1 = col1.strip() + text_part
                    
                    col2 = numbers_in_col2[0]
                    
                    if len(numbers_in_col2) > 1 and (not col3 or not any(c.isdigit() for c in col3)):
                        col3 = numbers_in_col2[1]
        
        if col3 and len(col3) > 3:
            has_letters = any(c.isalpha() for c in col3)
            has_digits = any(c.isdigit() for c in col3)
            
            if has_letters and has_digits:
                numbers_in_col3 = re.findall(number_pattern, col3)
                numbers_in_col3 = [n for n in numbers_in_col3 if len(n.replace(',', '').replace('.', '')) >= 2]
                
                if numbers_in_col3:
                    text_part = col3
                    for num in numbers_in_col3:
                        text_part = text_part.replace(num, '')
                    text_part = re.sub(r'\s+', ' ', text_part).strip()
                    
                    if col1.strip() and not col1.strip().endswith('.'):
                        col1 = col1.strip() + ' ' + text_part
                    else:
                        col1 = col1.strip() + text_part
                    
                    col3 = numbers_in_col3[0]
        
        df_fixed.iloc[i, 0] = col1
        if len(df_fixed.columns) > 1:
            df_fixed.iloc[i, 1] = col2
        if len(df_fixed.columns) > 2:
            df_fixed.iloc[i, 2] = col3
    
    return df_fixed

def extract_currency_and_unit_for_table(table_df, full_markdown: str) -> dict:
    result = {"currency": "Unknown", "unit": "Unknown"}
    
    for col in table_df.columns:
        col_str = str(col).lower()
        
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
    
    if len(table_df) > 0 and len(table_df.columns) > 0:
        first_cell = str(table_df.iloc[0, 0])[:50]
        
        if first_cell in full_markdown:
            pos = full_markdown.find(first_cell)
            context_before = full_markdown[max(0, pos-500):pos]
            context_after = full_markdown[pos:min(len(full_markdown), pos+200)]
            context = context_before + context_after
            
            currency_patterns = [
                (r'(?:in|of)?\s*(?:Rs\.?|INR|₹)\s', 'INR'),
                (r'(?:in|of)?\s*(?:USD|\$|US\$)\s', 'USD'),
                (r'(?:in|of)?\s*(?:EUR|€)\s', 'EUR'),
                (r'(?:in|of)?\s*(?:GBP|£)\s', 'GBP'),
                (r'(?:in|of)?\s*(?:JPY|¥)\s', 'JPY'),
                (r'Indian\s+Rupees?', 'INR'),
                (r'US\s+Dollars?', 'USD'),
            ]
            
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
    
    if result["currency"] == "Unknown" or result["unit"] == "Unknown":
        prompt = f"""You are a financial document analyzer. Extract the CURRENCY and UNIT from this table.

CONTEXT (surrounding text):
{full_markdown[:1500]}

TABLE (first 3 rows):
{table_df.head(3).to_markdown(index=False)}

TASK:
1. Identify the CURRENCY used (e.g., INR, USD, EUR, GBP, JPY, etc.)
2. Identify the UNIT/SCALE of amounts (e.g., Crores, Lakhs, Millions, Billions, Thousands, etc.)

Return ONLY this JSON format (no extra text):
{{
  "currency": "INR",
  "unit": "Crores"
}}

Your response:"""
        
        try:
            response = llama_client(prompt).strip()
            llm_result = parse_llama_json_response(response)
            
            if isinstance(llm_result, dict):
                if result["currency"] == "Unknown" and "currency" in llm_result:
                    result["currency"] = llm_result["currency"]
                if result["unit"] == "Unknown" and "unit" in llm_result:
                    result["unit"] = llm_result["unit"]
        except:
            pass
    
    return result

def add_total_to_table(df, sheet_name):
    try:
        if df.empty or len(df.columns) == 0:
            return df
        
        last_col_name = df.columns[-1]
        
        has_total = False
        if len(df) > 0 and len(df.columns) > 0:
            has_total = any('total' in str(val).lower() for val in df.iloc[:, 0])
        
        if has_total:
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
        
        return df
        
    except:
        return df

def clean_illegal_chars(df):
    return df.applymap(
        lambda x: ILLEGAL_CHARACTERS_RE.sub("", str(x)) if isinstance(x, str) else x
    )

def get_docling_pipeline():
    try:
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            table_structure_options=dict(do_cell_matching=True, mode=TableFormerMode.ACCURATE)
        )
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        settings.debug.profile_pipeline_timings = True
        return doc_converter
    except:
        return None

doc_converter_global = get_docling_pipeline()

def get_docling_results(INPUT_PDF_SOURCE):
    result = doc_converter_global.convert(INPUT_PDF_SOURCE)
    return result

def extract_contents_pages(pdf_path, max_pages=15):
    pages, _ = load_pdf_pages(pdf_path)
    contents_pages = []
    
    for page in pages[:max_pages]:
        text = page['text'].lower()
        
        if any(keyword in text for keyword in ['contents', 'index', 'table of contents']):
            if 'financial statement' in text or 'consolidated' in text or 'standalone' in text:
                contents_pages.append({
                    'page_num': page['page_num'],
                    'text': page['text']
                })
    
    return contents_pages

def extract_nested_page_numbers(contents_text):
    result = {"consolidated": {}, "standalone": {}}
    
    stand_pattern = r'(\d+)\s+(?:standalone|separate)\s+financial\s+statements?'
    stand_match = re.search(stand_pattern, contents_text, re.IGNORECASE)
    
    cons_pattern = r'(\d+)\s+consolidated\s+financial\s+statements?'
    cons_match = re.search(cons_pattern, contents_text, re.IGNORECASE)
    
    if stand_match:
        stand_page = int(stand_match.group(1))
        result['standalone']['start'] = stand_page
        
        if cons_match:
            cons_page = int(cons_match.group(1))
            if cons_page > stand_page:
                result['standalone']['end'] = cons_page - 1
            else:
                result['standalone']['end'] = stand_page + 70
        else:
            result['standalone']['end'] = stand_page + 70
    
    if cons_match:
        cons_page = int(cons_match.group(1))
        result['consolidated']['start'] = cons_page
        result['consolidated']['end'] = cons_page + 70
    
    return result if (result['consolidated'] or result['standalone']) else None

def parse_contents_with_llm(contents_pages):
    if not contents_pages:
        return None
    
    combined_text = "\n\n".join([p['text'] for p in contents_pages])
    
    prompt = f"""You are analyzing a Table of Contents from an annual report PDF.

YOUR TASK: Extract the page number ranges for:
1. Consolidated Financial Statements
2. Standalone Financial Statements

CONTENTS PAGE TEXT:
{combined_text}

INSTRUCTIONS:
- Look for entries mentioning financial statements. They can appear in various formats:
  
  FORMAT 1 - Nested/Indented structure:
  * "439 Financial statements"
  *     "440 Standalone Financial Statements"
  *     "576 Consolidated Financial statements"
  USE PAGE NUMBERS: 440 for Standalone, 576 for Consolidated (ignore parent 439)
  
  FORMAT 2 - Direct entries:
  * "290 standalone financial statements of PTC India limited"
  * "362 consolidated financial statements of PTC India limited"
  
  FORMAT 3 - Range format:
  * "Consolidated Financial Statements  75-131"

- If single page numbers:
  * Use as START
  * For END: If next section exists, use (next_page - 1), else add 70 pages

- Keywords (case-insensitive):
  * Consolidated: "consolidated financial statements"
  * Standalone: "standalone financial statements", "separate financial statements"

Return ONLY valid JSON:
{{
  "consolidated": {{"start": 576, "end": 650}},
  "standalone": {{"start": 440, "end": 575}}
}}

Your JSON response:"""
    
    try:
        response = llama_client(prompt).strip()
        print(f"\n[LLM Response]: {response}")
        
        result = parse_llama_json_response(response)
        
        if isinstance(result, dict) and 'consolidated' in result and 'standalone' in result:
            cons = result.get('consolidated', {})
            stand = result.get('standalone', {})
            
            if (cons.get('start') or stand.get('start')):
                print(f"\n[Success] LLM extracted:")
                print(f"  Consolidated: {cons}")
                print(f"  Standalone: {stand}")
                return result
        
        print("[Warning] LLM response invalid, trying regex fallback...")
        
    except Exception as e:
        print(f"[Error] LLM parsing failed: {e}, trying regex fallback...")
    
    regex_result = extract_nested_page_numbers(combined_text)
    if regex_result:
        print(f"\n[Success] Regex extracted:")
        print(f"  Consolidated: {regex_result.get('consolidated')}")
        print(f"  Standalone: {regex_result.get('standalone')}")
        return regex_result
    
    print("[Error] Both LLM and regex failed to extract page numbers")
    return None

def refine_page_ranges(page_ranges):
    if not page_ranges:
        return page_ranges
    
    cons = page_ranges.get('consolidated', {})
    stand = page_ranges.get('standalone', {})
    
    cons_start = cons.get('start')
    stand_start = stand.get('start')
    
    if cons_start and stand_start:
        if stand_start < cons_start:
            if stand.get('end') and stand['end'] >= cons_start:
                page_ranges['standalone']['end'] = cons_start - 1
        elif cons_start < stand_start:
            if cons.get('end') and cons['end'] >= stand_start:
                page_ranges['consolidated']['end'] = stand_start - 1
    
    return page_ranges

def determine_statement_type(page_num, page_ranges):
    if not page_ranges:
        return "Unknown"
    
    actual_page = page_num + 1
    
    cons = page_ranges.get('consolidated', {})
    stand = page_ranges.get('standalone', {})
    
    if cons.get('start') and cons.get('end'):
        if cons['start'] <= actual_page <= cons['end']:
            return "Consolidated"
    
    if stand.get('start') and stand.get('end'):
        if stand['start'] <= actual_page <= stand['end']:
            return "Standalone"
    
    return "Unknown"

def extract_cg(pdf_path):
    pages, pdf_document = load_pdf_pages(pdf_path)
    candidates = keyword_prefilter(pages)

    relevant_pages = []

    for p in candidates:
        stage1_result = stage_1_classify(p['text'])

        if isinstance(stage1_result, dict) and stage1_result.get("relevance") == "Relevant":
            confidence = stage1_result.get("confidence", 0)
            if confidence >= 0.85:
                stage2_result = stage_2_classify(p['text'])

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
    
    return None

def create_table_with_full_features(OUTPUT_PDF_PATH, page_ranges=None):
    pdf_name = os.path.splitext(os.path.basename(OUTPUT_PDF_PATH))[0]
    output_folder = f"{pdf_name}_tables"
    os.makedirs(output_folder, exist_ok=True)

    result = get_docling_results(OUTPUT_PDF_PATH)
    full_markdown = result.document.export_to_markdown()
    
    previous_page_num = None
    current_page_table_count = 0
    
    for table_ix, table in enumerate(result.document.tables):
        current_page_num = table.dict()['prov'][0]['page_no']

        statement_type = determine_statement_type(current_page_num, page_ranges)

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

        classification_result = classifyTable_with_context_check(
            table_df.to_markdown(), 
            table_ix, 
            full_markdown
        )

        if "true" in classification_result.lower():
            table_df = fix_merged_columns(table_df)
            currency_unit = extract_currency_and_unit_for_table(table_df, full_markdown)
            table_df = add_total_to_table(table_df, sheet_name)
            
            table_df['Currency'] = currency_unit['currency']
            table_df['Unit'] = currency_unit['unit']
            table_df['Statement_Type'] = statement_type
            table_df['Page_Number'] = current_page_num + 1
            
            filename = f"{sheet_name}_{statement_type}_{currency_unit['currency']}_{currency_unit['unit']}.xlsx"
            filepath = os.path.join(output_folder, filename)
            table_df.to_excel(filepath, index=False)
            
            print(f"✓ Saved: {filename}")
        
        previous_page_num = current_page_num

if __name__ == "__main__":
    start_time = time.time()

    print("\n" + "="*60)
    print("STEP 1: Extracting Contents Page")
    print("="*60)
    contents_pages = extract_contents_pages(PDF_PATH)
    
    page_ranges = None
    if contents_pages:
        print(f"Found {len(contents_pages)} contents page(s)")
        page_ranges = parse_contents_with_llm(contents_pages)
        
        if page_ranges:
            page_ranges = refine_page_ranges(page_ranges)
            print("\n✅ Page ranges identified successfully")
        else:
            print("\n⚠️  Could not extract page ranges")
    else:
        print("\n⚠️  No contents page found")

    print("\n" + "="*60)
    print("STEP 2: Extracting Contingent Liability Pages")
    print("="*60)
    relevant_pages = extract_cg(PDF_PATH)
    
    if relevant_pages:
        print(f"Found {len(relevant_pages)} relevant page(s): {relevant_pages}")
        
        print("\n" + "="*60)
        print("STEP 3: Processing Tables")
        print("="*60)
        create_table_with_full_features(OUTPUT_PDF_PATH, page_ranges)
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n✅ Processing completed in {total_time:.2f} seconds")
    else:
        print("\n❌ No relevant pages found")
