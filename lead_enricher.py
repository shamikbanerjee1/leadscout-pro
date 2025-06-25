import openai
import json
import re
import time
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import OPENAI_API_KEY
from textblob import TextBlob

openai.api_key = OPENAI_API_KEY

class LeadEnricher:
    def __init__(self, max_workers=20):
        self.max_workers = max_workers
    
    def enrich_leads(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main enrichment pipeline with parallel processing"""
        if df.empty:
            return df
            
        # Create progress elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_container = st.empty()
        
        # Create thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            # Submit all enrichment tasks
            for index, row in df.iterrows():
                future = executor.submit(
                    self._get_gpt_enrichment, 
                    row
                )
                futures[future] = index
            
            # Process results as they complete
            enriched_data = {}
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                index = futures[future]
                try:
                    enriched_data[index] = future.result()
                except Exception as e:
                    company_name = df.loc[index, 'Name']
                    st.warning(f"GPT enrichment failed for {company_name}: {e}")
                    enriched_data[index] = self._get_default_enrichment()
                
                # Update progress
                completed += 1
                progress_percent = int((completed / total) * 100)
                progress_bar.progress(progress_percent)
                status_text.text(f"Progress: {progress_percent}%")
                status_container.text(f"Completed {completed} of {total} leads")
        
        # Clear progress elements
        progress_bar.empty()
        status_text.empty()
        status_container.empty()
        
        # Apply enrichment to DataFrame
        enriched_df = df.copy()
        for col in self._enrichment_columns():
            enriched_df[col] = enriched_df.index.map(
                lambda idx: enriched_data.get(idx, {}).get(col, None))
        
        return enriched_df
    
    def _get_gpt_enrichment(self, row: pd.Series) -> dict:
        """Get company insights from GPT with reviews and sentiment"""
        time.sleep(0.5)
        prompt = self._create_gpt_prompt(row)
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=500,  # Increased for reviews
                response_format={"type": "json_object"}
            )
            
            enrichment_data = self._parse_gpt_response(response.choices[0].message.content)
            
            # Perform sentiment analysis on reviews
            if enrichment_data.get("reviews"):
                sentiment_score = self._analyze_sentiment(enrichment_data["reviews"])
                enrichment_data["sentiment_score"] = sentiment_score
            
            return enrichment_data
            
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")
    
    def _get_default_enrichment(self) -> dict:
        return {
            "is_startup": None,
            "founded_year": None,
            "estimated_revenue": None,
            "estimated_funding": None,
            "employee_count": None,
            "industry": None,
            "business_model": None,
            "review_count": None,
            "reviews": None,
            "sentiment_score": None
        }
    
    def _enrichment_columns(self) -> list:
        return [
            "is_startup", "founded_year", "estimated_revenue",
            "estimated_funding", "employee_count", "industry", "business_model",
            "review_count", "reviews", "sentiment_score"
        ]
    
    def _create_gpt_prompt(self, row: pd.Series) -> str:
        """Create GPT prompt including reviews request"""
        return f"""
        Analyze this company and provide realistic estimates including reviews:
        Name: {row.get('Name')}
        Website: {row.get('Website') or 'Not provided'}
        Location: {row.get('Address') or 'Unknown'}
        Category: {row.get('Category') or 'Unknown'}
        
        Also provide:
        - Total number of reviews for a company
        
        Respond with this exact JSON structure:
        {{
            "is_startup": boolean|null,
            "founded_year": integer|null,
            "estimated_revenue": number|null,
            "estimated_funding": number|null,
            "employee_count": integer|null,
            "industry": string|null,
            "business_model": string|null,
            "review_count": integer|null,
            "reviews": [string]|null  # List of review texts
        }}"""

    def _get_system_prompt(self) -> str:
        """System instructions for GPT"""
        return """You are a business intelligence analyst. Provide:
        1. Reasonable estimates based on available information
        2. Industry benchmarks when exact data is unavailable
        3. Convert ranges to mid-point (e.g., "10-50 employees" → 30)
        4. Business model classification (B2B/B2C/B2B2C)
        5. Conservative estimates
        6. Employee_count should be an integer
        7. Estimated_revenue should be annual revenue in USD
        8. Estimated_funding should be total funding in USD
        9. Return values ONLY in the specified JSON format"""
    
    def _analyze_sentiment(self, reviews: list) -> float:
        """Calculate average sentiment score from reviews (0-1 scale)"""
        if not reviews:
            return 0.5  # Neutral if no reviews
        
        total_polarity = 0
        for review in reviews:
            analysis = TextBlob(review)
            # Convert polarity from (-1 to 1) to (0 to 1)
            total_polarity += (analysis.sentiment.polarity + 1) / 2
        
        return total_polarity / len(reviews)
    
    def _parse_gpt_response(self, response: str) -> dict:
        """Parse and validate GPT response"""
        try:
            # Clean response - remove markdown code blocks if present
            clean_response = re.sub(r'```(?:json)?', '', response).strip()
            
            # Try to find JSON structure
            json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = json.loads(clean_response)
            
            # Parse values
            parsed = {
                "is_startup": self._parse_bool(data.get("is_startup")),
                "founded_year": self._parse_int(data.get("founded_year")),
                "estimated_revenue": self._parse_float(data.get("estimated_revenue")),
                "estimated_funding": self._parse_float(data.get("estimated_funding")),
                "employee_count": self._parse_int(data.get("employee_count")),
                "industry": self._parse_str(data.get("industry")),
                "business_model": self._parse_business_model(data.get("business_model")),
                "review_count": self._parse_int(data.get("review_count")),
                "reviews": self._parse_reviews(data.get("reviews")),
                "sentiment_score": None 
            }
            
            # Validate revenue estimates
            revenue = parsed["estimated_revenue"]
            if revenue and revenue > 1e12:  # Sanity check ($1 trillion)
                parsed["estimated_revenue"] = None
            
            # Validate employee counts
            employees = parsed["employee_count"]
            if employees and employees > 1000000:  # Sanity check
                parsed["employee_count"] = None
                
            # Validate funding
            funding = parsed["estimated_funding"]
            if funding and funding > 1e11:  # $100 billion
                parsed["estimated_funding"] = None
            
            return parsed
            
        except Exception as e:
            st.warning(f"Failed to parse GPT response: {e}")
            st.warning(f"Original response: {response[:500]}...")
            return {
                "is_startup": None,
                "founded_year": None,
                "estimated_revenue": None,
                "estimated_funding": None,
                "employee_count": None,
                "industry": None,
                "business_model": None,
                "review_count" : None,
                "reviews" : None,
                "sentiment_score" : None
            }
        
    def _parse_reviews(self, value):
        """Parse reviews field into list of strings"""
        if isinstance(value, list):
            return [str(item) for item in value]
        return None
    
    def _parse_bool(self, value):
        """Convert to boolean or return None"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ["true", "yes"]:
                return True
            if value.lower() in ["false", "no"]:
                return False
        return None
    
    def _parse_int(self, value):
        """Convert to integer or return None"""
        try:
            # Handle ranges (e.g., "10-50" → 30)
            if isinstance(value, str) and '-' in value:
                parts = value.split('-')
                low = self._convert_numeric(parts[0].strip())
                high = self._convert_numeric(parts[1].strip())
                return int((low + high) / 2)
            return int(self._convert_numeric(value))
        except (TypeError, ValueError):
            return None
    
    def _parse_float(self, value):
        """Convert to float or return None"""
        try:
            # Handle ranges (e.g., "1M-5M" → 3M)
            if isinstance(value, str) and '-' in value:
                parts = value.split('-')
                low = self._convert_numeric(parts[0].strip())
                high = self._convert_numeric(parts[1].strip())
                return (low + high) / 2
            return self._convert_numeric(value)
        except (TypeError, ValueError):
            return None
            
    def _convert_numeric(self, value):
        """Convert string representations to numbers"""
        if isinstance(value, (int, float)):
            return value
            
        if value is None:
            return None
            
        # Handle currency strings and ranges
        if isinstance(value, str):
            value = value.replace('$', '').replace(',', '').strip().upper()
            
            # Handle multipliers (K, M, B)
            multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}
            
            if value.endswith(tuple(multipliers.keys())):
                num_part = value[:-1]
                multiplier = multipliers[value[-1]]
                try:
                    return float(num_part) * multiplier
                except ValueError:
                    pass
                    
            # Handle "million", "billion" text
            if "million" in value.lower():
                num_part = value.lower().replace("million", "").strip()
                try:
                    return float(num_part) * 1e6
                except ValueError:
                    pass
                    
            if "billion" in value.lower():
                num_part = value.lower().replace("billion", "").strip()
                try:
                    return float(num_part) * 1e9
                except ValueError:
                    pass
                    
            # Try direct conversion
            try:
                return float(value)
            except ValueError:
                pass
                
        return None
    
    def _parse_str(self, value):
        """Return string or None"""
        return str(value) if value is not None else None
    
    def _parse_business_model(self, value):
        """Validate business model value"""
        valid_models = ["B2B", "B2C", "B2B2C"]
        if value in valid_models:
            return value
        return None