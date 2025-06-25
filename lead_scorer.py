import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

class LeadScorer:
    def __init__(self, explicit_weights=None):
        # Updated weights with funding and dynamic adjustment
        self.base_weights = explicit_weights or {
            'employee_count': 0.10,
            'estimated_revenue': 0.25,
            'founded_year': 0.10,
            'estimated_funding': 0.20,
            'business_model_score': 0.10,
            'sentiment_score': 0.25  
        }
        self.scaler = MinMaxScaler()
    
    def calculate_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate business model scores
        df['business_model_score'] = df['business_model'].apply(
            self._calculate_business_model_score
        )
        
        # Fill missing values
        df = self._fill_missing_values(df)
        
        # Calculate dynamic weights
        dyn_weights = self._calculate_dynamic_weights(df)
        
        # Apply dynamic weighting
        final_weights = {
            f: self.base_weights[f] * dyn_weights.get(f, 1)
            for f in self.base_weights
        }
        total = sum(final_weights.values())
        final_weights = {f: w/total for f, w in final_weights.items()}
        
        # Calculate score
        df['explicit_score'] = self._calculate_component_score(df, final_weights)
        df['lead_score'] = df['explicit_score']  # Now 100% of score
        
        # Add priority clusters
        df = self._add_priority_clusters(df)
        
        return df.sort_values('lead_score', ascending=False)
    
    def _calculate_business_model_score(self, model):
        model_scores = {'B2B': 0.9, 'B2B2C': 0.7, 'B2C': 0.5}
        return model_scores.get(model, 0.5)
    
    def _calculate_dynamic_weights(self, df):
        variance_weights = {}
        total_variance = 0
        
        for feature in ['employee_count', 'estimated_revenue', 
                       'founded_year', 'estimated_funding',
                       'sentiment_score']:
            if feature in df.columns:
                std = df[feature].std()
                mean = abs(df[feature].mean()) or 1  # Avoid division by zero
                cv = std / mean
                variance_weights[feature] = max(0.1, cv)
                total_variance += variance_weights[feature]
        
        if total_variance > 0:
            return {f: w/total_variance for f, w in variance_weights.items()}
        return {f: 1/len(variance_weights) for f in variance_weights}
    
    def _fill_missing_values(self, df):
        current_year = pd.Timestamp.now().year
        
        fill_values = {
            'employee_count': 10,  # Small team
            'estimated_revenue': 0,
            'estimated_funding': 0,
            'founded_year': current_year - 5,  # Avg startup age
            'business_model_score': 0.5,
            'sentiment_score': 0.5
        }
        
        for col, default in fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
                
        return df
    
    def _calculate_component_score(self, df, weights):
        scores = []
        for feature, weight in weights.items():
            if feature not in df.columns:
                continue
                
            values = df[feature].values.reshape(-1, 1)
            normalized = self.scaler.fit_transform(values).flatten()
            weighted = normalized * weight
            scores.append(weighted)
        
        if scores:
            total = np.sum(scores, axis=0)
            return np.interp(total, (total.min(), total.max()), (0, 100))
        return np.zeros(len(df))
    
    def _add_priority_clusters(self, df):
        if not df.empty:
            bins = [0, 30, 80, 100]
            labels = ['Low', 'Medium', 'High']
            df['priority_level'] = pd.cut(
                df['lead_score'], 
                bins=bins, 
                labels=labels,
                include_lowest=True
            )
        else:
            df['priority_level'] = 'High'
        return df
