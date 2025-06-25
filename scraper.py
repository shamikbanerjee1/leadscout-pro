# scraper.py
import pandas as pd
import openai
import json
import re
import streamlit as st
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def _search_google_maps_page(domain: str, location: str, page: int, page_size: int) -> pd.DataFrame:
    """Generate a single page of business leads using GPT"""
    # Add random delay to prevent rate limiting
    time.sleep(random.uniform(0.5, 1.5))
    
    # Calculate starting point for this page
    start_index = page * page_size + 1
    
    # Generate prompt with pagination
    prompt = f"""
    Generate a list of {page_size} real companies in the {domain} industry located in {location}, 
    starting from company #{start_index}.
    
    For each company, provide:
    - Company name
    - Full street address
    - Phone number (if available, else null)
    - Website URL (if available, else null)
    - Primary business category
    - Google Maps link (search query format)
    
    Return results in JSON format:
    {{
        "companies": [
            {{
                "Name": "Example Inc",
                "Address": "123 Main St, {location}",
                "Phone": "+65 1234 5678",
                "Website": "https://example.com",
                "Category": "{domain}",
                "Map Link": "https://www.google.com/maps/search/?api=1&query=Example+Inc+{location}"
            }},
            ...
        ]
    }}
    """
    
    # Call GPT API with retry logic
    for attempt in range(3):  # Retry up to 3 times
        try:
            response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a business research assistant. Provide accurate company information in requested JSON format."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=3000
            )
            break  # Exit retry loop if successful
        except Exception as e:
            if attempt == 2:  # Final attempt failed
                raise RuntimeError(f"API request failed after 3 attempts: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    # Handle JSON parsing
    content = response.choices[0].message.content
    
    try:
        # First try to parse as JSON
        data = json.loads(content)
    except json.JSONDecodeError:
        # If fails, try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except:
                data = {"companies": []}
        else:
            # Try to fix common formatting issues
            fixed_content = re.sub(r',\s*}', '}', content)  # Fix trailing commas
            fixed_content = re.sub(r',\s*]', ']', fixed_content)
            try:
                data = json.loads(fixed_content)
            except:
                data = {"companies": []}

    companies = data.get("companies", [])

    # Add null fields expected by downstream processing
    for company in companies:
        company["Rating"] = None
        company["Review Count"] = None
        company["place_id"] = None
        
    # Remove duplicates based on name and address
    df = pd.DataFrame(companies)
    if not df.empty:
        df = df.drop_duplicates(subset=['Name', 'Address'])
        
    return df

def search_google_maps(domain: str, location: str, total_leads: int, page_size: int = 20, max_workers: int = 5) -> pd.DataFrame:
    """Generate business leads using GPT with parallel processing"""
    total_pages = (total_leads + page_size - 1) // page_size
    pages = list(range(total_pages))
    
    # Create progress elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all page requests
        futures = {executor.submit(_search_google_maps_page, domain, location, page, page_size): page 
                  for page in pages}
        
        completed = 0
        for future in as_completed(futures):
            page = futures[future]
            try:
                page_df = future.result()
                if not page_df.empty:
                    results.append(page_df)
            except Exception as e:
                st.error(f"GPT lead generation failed for page {page+1}: {str(e)}")
            
            # Update progress
            completed += 1
            progress_percent = int((completed / total_pages) * 100)
            progress_bar.progress(progress_percent)
            status_text.text(f"Scraping page {completed} of {total_pages}...")
    
    # Clear progress elements
    progress_bar.empty()
    status_text.empty()
    
    # Combine results
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()