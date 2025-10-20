def extract_page_number_from_content(page_text):
    """
    Use LLM to extract the actual page number from the page content.
    PDFs often have page numbers in headers/footers.
    """
    prompt = f"""
You are analyzing a PDF page to find its page number.

TASK: Extract the page number that appears on this page (usually in header/footer).

IMPORTANT:
- Look for standalone numbers that represent page numbers (e.g., "445", "Page 445", "- 445 -")
- Ignore reference numbers, note numbers, or numbers within text content
- Focus on numbers that appear at the top or bottom of the page
- If multiple candidates exist, choose the one that looks most like a page number
- Return ONLY the number, nothing else

PAGE CONTENT (first 500 and last 500 characters):
{page_text[:500]}
...
{page_text[-500:]}

Return ONLY the page number as an integer (e.g., 445):
"""
    
    try:
        response = llama_client(prompt).strip()
        # Extract just the number from response
        page_num = int(re.search(r'\d+', response).group())
        return page_num
    except:
        return None

def create_table_with_full_features(OUTPUT_PDF_PATH, relevant_pages, page_statement_map):
    pdf_name = os.path.splitext(os.path.basename(OUTPUT_PDF_PATH))[0]
    output_folder = f"{pdf_name}_tables"
    os.makedirs(output_folder, exist_ok=True)

    result = get_docling_results(OUTPUT_PDF_PATH)
    full_markdown = result.document.export_to_markdown()
    
    # Create mapping: extracted_page_index -> original_page_num
    extracted_to_original = {}
    for idx, orig_page in enumerate(relevant_pages):
        extracted_to_original[idx] = orig_page
    
    # Load the OUTPUT_PDF to extract actual page numbers
    output_pdf = fitz.open(OUTPUT_PDF_PATH)
    actual_page_numbers = {}
    
    for idx in range(len(output_pdf)):
        page_text = output_pdf[idx].get_text("text")
        actual_page = extract_page_number_from_content(page_text)
        actual_page_numbers[idx] = actual_page
        print(f"Extracted page {idx} has actual page number: {actual_page}")
    
    output_pdf.close()
    
    # Now determine statement type based on ACTUAL page numbers
    page_to_statement_type = {}
    if page_ranges:
        cons = page_ranges.get('consolidated', {})
        stand = page_ranges.get('standalone', {})
        
        for idx, actual_page in actual_page_numbers.items():
            if actual_page:
                statement_type = "Unknown"
                
                if cons.get('start') and cons.get('end'):
                    if cons['start'] <= actual_page <= cons['end']:
                        statement_type = "Consolidated"
                
                if stand.get('start') and stand.get('end'):
                    if stand['start'] <= actual_page <= stand['end']:
                        statement_type = "Standalone"
                
                page_to_statement_type[idx] = statement_type
                print(f"Actual page {actual_page} (index {idx}): {statement_type}")
    
    previous_page_num = None
    current_page_table_count = 0
    
    for table_ix, table in enumerate(result.document.tables):
        # FIXED: Docling uses 1-indexed, convert to 0-indexed
        docling_page_num = table.dict()['prov'][0]['page_no']
        current_page_num = docling_page_num - 1  # Convert 1-indexed to 0-indexed
        
        # Get the statement type based on actual page number
        statement_type = page_to_statement_type.get(current_page_num, "Unknown")
        actual_page = actual_page_numbers.get(current_page_num, "Unknown")

        if previous_page_num is None:
            previous_page_num = current_page_num

        if previous_page_num == current_page_num:
            current_page_table_count += 1
        else:
            current_page_table_count = 1

        sheet_name = f"Page_no_{actual_page}_table_{current_page_table_count}"

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
            table_df['Statement_Type'] = statement_type  # Add statement type to the table
            table_df['Actual_Page'] = actual_page  # Add actual page number
            
            filename = f"Page_{actual_page}_{statement_type}_{currency_unit['currency']}_{currency_unit['unit']}.xlsx"
            filepath = os.path.join(output_folder, filename)
            table_df.to_excel(filepath, index=False)
            
            print(f"✓ Saved: {filename} (Statement Type: {statement_type})")
        
        previous_page_num = current_page_num

# Update the main execution to pass page_ranges to the table creation function
if __name__ == "__main__":
    start_time = time.time()

    print("\n" + "="*60)
    print("STEP 1: Extracting Contents Page")
    print("="*60)
    contents_pages = extract_contents_pages(PDF_PATH)
    
    page_ranges = None  # Make this available globally
    if contents_pages:
        print(f"Found {len(contents_pages)} contents page(s)")
        page_ranges = parse_contents_with_llm(contents_pages)
        
        if page_ranges:
            page_ranges = refine_page_ranges(page_ranges)
            print("\n✅ Page ranges identified successfully")
            print(f"  Consolidated: {page_ranges.get('consolidated')}")
            print(f"  Standalone: {page_ranges.get('standalone')}")
        else:
            print("\n⚠️  Could not extract page ranges")
    else:
        print("\n⚠️  No contents page found")

    print("\n" + "="*60)
    print("STEP 2: Extracting Contingent Liability Pages")
    print("="*60)
    relevant_pages, page_statement_map = extract_cg(PDF_PATH, page_ranges)
    
    if relevant_pages:
        print(f"Found {len(relevant_pages)} relevant page(s): {relevant_pages}")
        
        print("\nPage to Statement Type Mapping:")
        for page in relevant_pages:
            statement_type = page_statement_map.get(page, "Unknown")
            print(f"  Page {page + 1}: {statement_type}")
        
        print("\n" + "="*60)
        print("STEP 3: Processing Tables")
        print("="*60)
        # Pass page_ranges to the function
        create_table_with_full_features(OUTPUT_PDF_PATH, relevant_pages, page_statement_map)
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n✅ Processing completed in {total_time:.2f} seconds")
    else:
        print("\n❌ No relevant pages found")
