import streamlit as st
import pandas as pd
import plotly.express as px
import time
import base64
from io import BytesIO, StringIO
from scraper import search_google_maps
from lead_enricher import LeadEnricher
from lead_scorer import LeadScorer
from fpdf import FPDF
from datetime import datetime


# ================ APP CONFIG ================
st.set_page_config(
    page_title="LeadScout Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================ STYLING ================
# Inject custom CSS
st.markdown("""
<style>
    /* Main styling */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(90deg, #1a2980 0%, #26d0ce 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e6ed;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1a2980 0%, #26d0ce 100%);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #1a2980 0%, #26d0ce 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(26, 41, 128, 0.3);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid #e0e6ed;
    }
    
    /* Tab styling */
    .stTabs [role="tablist"] {
        gap: 10px;
        padding: 5px;
        background: #f0f4f9;
        border-radius: 12px;
    }
    
    .stTabs [role="tab"] {
        border-radius: 10px !important;
        padding: 8px 20px;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1a2980 0%, #26d0ce 100%);
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ================ UI CONSTANTS ================
DISPLAY_COLS = ['Name', 'Address', 'Phone', 'Website', 'Category', 'Map Link']
ENRICHED_DISPLAY_COLS = DISPLAY_COLS + [
    'is_startup', 'founded_year', 'estimated_revenue', 
    'estimated_funding', 'employee_count', 'industry', 'business_model',
    'sentiment_score'
]
SCORED_DISPLAY_COLS = ENRICHED_DISPLAY_COLS + ['priority_level', 'lead_score']
REPORT_COLS = ['Name', 'priority_level', 'lead_score', 'business_model', 'estimated_revenue', 'employee_count']
CLUSTER_COLORS = {'High': '🟢', 'Medium': '🟡', 'Low': '🔴'}
PAGE_SIZE = 20
DEFAULT_LEAD_COUNT = 100
PALETTE = ['#1a2980', '#26d0ce', '#ff6b6b', '#4ecdc4', '#ffa62b']

# ================ HEADER ================
st.markdown('<div class="header"><h1>🚀 LeadScout Pro</h1><p>AI-Powered Lead Generation & Prioritization</p></div>', unsafe_allow_html=True)

# ================ MAIN APP ================
def main():
    init_session_state()
    render_sidebar()
    
    # Use tabs for workflow stages
    tab1, tab2, tab3 = st.tabs(["🔍 Lead Generation", "✨ Lead Enrichment", "🏆 Lead Scoring"])
    
    with tab1:
        handle_lead_generation()
        display_leads_table(st.session_state.scraped_df, "Raw Leads", "current_page", DISPLAY_COLS)
        
    with tab2:
        handle_lead_enrichment()
        if not st.session_state.enriched_df.empty:
            display_leads_table(st.session_state.enriched_df, "Enriched Leads", "current_enriched_page", ENRICHED_DISPLAY_COLS)
            display_enrichment_insights()
        
    with tab3:
        handle_lead_scoring()
        if not st.session_state.scored_df.empty:
            display_scored_leads(st.session_state.scored_df)
            display_score_distribution()

# ================ SESSION STATE ================
def init_session_state():
    if "domain" not in st.session_state:
        st.session_state.domain = "Software"
    if "location" not in st.session_state:
        st.session_state.location = "Singapore"
    if "scraped_df" not in st.session_state:
        st.session_state.scraped_df = pd.DataFrame(columns=DISPLAY_COLS)
    if "enriched_df" not in st.session_state:
        st.session_state.enriched_df = pd.DataFrame(columns=ENRICHED_DISPLAY_COLS)
    if "scored_df" not in st.session_state:
        st.session_state.scored_df = pd.DataFrame(columns=SCORED_DISPLAY_COLS)
    if "run_search" not in st.session_state:
        st.session_state.run_search = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = 0
    if "current_enriched_page" not in st.session_state:
        st.session_state.current_enriched_page = 0
    if "current_scored_page" not in st.session_state:
        st.session_state.current_scored_page = 0
    if "total_leads" not in st.session_state:
        st.session_state.total_leads = DEFAULT_LEAD_COUNT
    if "generated_pages" not in st.session_state:
        st.session_state.generated_pages = set()

# ================ SIDEBAR ================
def render_sidebar():
    with st.sidebar:
                
        # Input fields with icons
        st.session_state.domain = st.text_input(
            "🏢 Industry/Service", 
            st.session_state.domain,
            key="domain_input"
        )
        
        st.session_state.location = st.text_input(
            "📍 Location", 
            st.session_state.location,
            key="location_input"
        )
        
        st.session_state.total_leads = st.number_input(
            "📊 Total Leads to Generate", 
            min_value=20, 
            max_value=500, 
            value=st.session_state.total_leads,
            step=20
        )
        
        # Scrape button
        if st.button("🔍 Scrape Leads", key="scrape_button", use_container_width=True):
            st.session_state.run_search = True
            st.session_state.current_page = 0
            st.session_state.generated_pages = set()
            st.session_state.enriched_df = pd.DataFrame()  
            st.session_state.scored_df = pd.DataFrame()
            
        st.markdown("---")

        if not st.session_state.scored_df.empty:
            st.markdown("### CRM Export")
            if st.button("📤 Export to Salesforce", key="salesforce_export"):
                crm_df = prepare_salesforce_export(st.session_state.scored_df)
                csv = crm_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="salesforce_leads.csv">📥 Download CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        st.markdown("---")
        # App info
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: #f0f8ff; border-radius: 10px;">
            <h4>About LeadScout</h4>
            <p>AI-powered lead generation and scoring platform</p>
            <p style="font-size: 0.8rem; margin-top: 20px;">v2.1 | © 2023</p>
        </div>
        """, unsafe_allow_html=True)

# ================ SALESFORCE CRM REPORTING =======
def prepare_salesforce_export(df):
    """Convert to Salesforce import format"""
    crm_df = pd.DataFrame()
    crm_df['FirstName'] = ''  # Placeholder
    crm_df['LastName'] = df['Name'].str.split().str[-1]
    crm_df['Company'] = df['Name']
    crm_df['Street'] = df['Address']
    crm_df['Phone'] = df['Phone']
    crm_df['Website'] = df['Website']
    crm_df['LeadSource'] = 'LeadScout Pro'
    crm_df['Description'] = (
        f"Industry: " + df['industry'].fillna('') + " | " +
        f"Score: " + df['lead_score'].astype(str) + " | " +
        f"Employees: " + df['employee_count'].astype(str)
    )
    return crm_df
# ================ LEAD GENERATION ================
def handle_lead_generation():
    if st.session_state.run_search:
        if st.session_state.scraped_df.empty:  # Only run if we haven't scraped yet
            with st.spinner(f"🔍 Generating {st.session_state.total_leads} leads..."):
                try:
                    # Get all leads at once with parallel processing
                    scraped_df = search_google_maps(
                        st.session_state.domain,
                        st.session_state.location,
                        total_leads=st.session_state.total_leads,
                        page_size=PAGE_SIZE
                    )
                    
                    if scraped_df.empty:
                        st.warning("⚠️ No results found. Try different search terms.")
                    else:
                        # Trim to requested number of leads
                        st.session_state.scraped_df = scraped_df.head(st.session_state.total_leads)
                        
                        # Remove duplicates
                        n_before = len(st.session_state.scraped_df)
                        st.session_state.scraped_df = st.session_state.scraped_df.drop_duplicates(
                            subset=['Name', 'Address']
                        )
                        n_after = len(st.session_state.scraped_df)
                        duplicates_removed = n_before - n_after
                        if duplicates_removed > 0:
                            st.info(f"🧹 Removed {duplicates_removed} duplicate leads.")
                            
                except Exception as e:
                    st.error(f"❌ Search failed: {e}")
        
        st.session_state.run_search = False
# ============= REPORT GENERATION ==================#
def generate_report(df):
    """Generate PDF report with key insights"""
    if df.empty:
        return None
        
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"LeadScout Pro Report - {datetime.now().strftime('%Y-%m-%d')}", ln=1)
    pdf.set_font("Arial", size=12)
    
    # Summary stats
    pdf.cell(0, 10, f"Total Leads: {len(df)}", ln=1)
    pdf.cell(0, 10, f"Average Score: {df['lead_score'].mean():.1f}", ln=1)
    pdf.cell(0, 10, f"High Priority: {len(df[df['priority_level'] == 'High'])}", ln=1)
    
    # Top leads table
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Top 10 Leads:", ln=1)
    pdf.set_font("Arial", size=10)
    
    # Create table header
    col_widths = [60, 20, 20, 20, 30, 20]
    headers = ['Name', 'Priority', 'Score', 'Model', 'Revenue', 'Employees']
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1)
    pdf.ln()
    
    # Add table rows
    top_leads = df.sort_values('lead_score', ascending=False).head(10)
    for _, row in top_leads.iterrows():
        for i, col in enumerate(REPORT_COLS):
            value = str(row[col])[:30]  # Truncate long values
            pdf.cell(col_widths[i], 10, value, border=1)
        pdf.ln()
    
    return pdf.output(dest='S').encode('latin1')

# ================ LEAD ENRICHMENT ================
def handle_lead_enrichment():
    if not st.session_state.scraped_df.empty and st.session_state.enriched_df.empty:
        with st.container():
            st.subheader("✨ AI Enrichment")
            st.caption("Enhance leads with AI-powered business intelligence")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("✨ Enrich Leads", key="enrich_button", use_container_width=True):
                    with st.spinner("🔮 Enriching leads with AI..."):
                        enricher = LeadEnricher()
                        n_before = len(st.session_state.scraped_df)
                        unique_leads = st.session_state.scraped_df.drop_duplicates(
                            subset=['Name', 'Address']
                        )
                        n_after = len(unique_leads)
                        duplicates_removed = n_before - n_after
                        if duplicates_removed > 0:
                            st.info(f"🧹 Removed {duplicates_removed} duplicates before enrichment.")
                        
                        enriched = enricher.enrich_leads(unique_leads)
                        st.session_state.enriched_df = enriched
                        st.session_state.current_enriched_page = 0
                        st.rerun()

# ================ LEAD SCORING ================
def handle_lead_scoring():
    if not st.session_state.enriched_df.empty and st.session_state.scored_df.empty:
        with st.container():
            st.subheader("🏆 Lead Scoring")
            st.caption("Prioritize leads based on growth potential and fit")
            
            with st.expander("📊 Scoring Methodology"):
                st.markdown("""
                **Our AI-driven prioritization focuses on:**
                - 💰 **Revenue Potential** (30% base weight)
                - 💸 **Funding Status** (25% base weight)
                - 👥 **Team Size** (15% base weight)
                - 🎂 **Company Maturity** (15% base weight)
                - 🤝 **Business Model** (15% base weight)
                - 📊 **Dynamic Adjustment**: Features with more variation get higher weights
                """)
            
            if st.button("📈 Score Leads", key="score_button", use_container_width=True):
                with st.spinner("🧠 Calculating lead scores..."):
                    scorer = LeadScorer()
                    scored = scorer.calculate_scores(st.session_state.enriched_df)
                    st.session_state.scored_df = scored
                    st.session_state.current_scored_page = 0
                    st.rerun()

# ================ DATA DISPLAY ================
def display_leads_table(df, title, page_key, display_cols):
    if df.empty:
        return
    
    st.markdown(f"### {title}")
    
    total_pages = (len(df) + PAGE_SIZE - 1) // PAGE_SIZE
    start_idx = st.session_state[page_key] * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, len(df))
    page_df = df.iloc[start_idx:end_idx].copy()
    
    if 'priority_level' in page_df.columns:
        page_df['Priority'] = page_df['priority_level'].map(CLUSTER_COLORS)
        display_cols = ['Priority'] + [c for c in display_cols if c != 'Priority']
    
    if 'is_startup' in page_df.columns:
        page_df['is_startup'] = page_df['is_startup'].astype('boolean')
    
    display_df = page_df[display_cols]
    display_df.index = range(start_idx + 1, end_idx + 1)
    
    column_config = {
        "Map Link": st.column_config.LinkColumn("📍 View Map", display_text="Open Map", help="View on Google Maps"),
        "is_startup": st.column_config.CheckboxColumn("🚀 Startup?"),
        "founded_year": st.column_config.NumberColumn("📅 Founded"),
        "estimated_revenue": st.column_config.NumberColumn(
            "💰 Revenue (USD)", 
            format="$%d",
            help="Estimated annual revenue"
        ),
        "estimated_funding": st.column_config.NumberColumn(
            "💸 Funding (USD)", 
            format="$%d",
            help="Total estimated funding"
        ),
        "employee_count": st.column_config.NumberColumn("👥 Employees"),
        "business_model": st.column_config.SelectboxColumn(
            "🤝 Business Model",
            options=["B2B", "B2C", "B2B2C"]
        ),
        "review_count": st.column_config.NumberColumn("⭐ Reviews"),
        "sentiment_score": st.column_config.ProgressColumn(
            "😊 Sentiment",
            help="Customer sentiment score (0-1)",
            format="%.2f",
            min_value=0,
            max_value=1
        )
    }
    
    if 'lead_score' in df.columns:
        column_config["lead_score"] = st.column_config.ProgressColumn(
            "📈 Lead Score",
            help="Lead quality score (0-100)",
            format="%d",
            min_value=0,
            max_value=100
        )
    
    # Display in card
    with st.container():
        st.dataframe(
            display_df,
            column_config=column_config,
            use_container_width=True,
            height=600
        )
        
        # Pagination
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if st.button("⏪ Previous", 
                            disabled=(st.session_state[page_key] == 0),
                            key=f"{page_key}_prev"):
                    st.session_state[page_key] = max(0, st.session_state[page_key] - 1)
                    st.rerun()
            with col2:
                st.markdown(f"<div style='text-align: center;'>📄 Page {st.session_state[page_key] + 1} of {total_pages}</div>", unsafe_allow_html=True)
            with col3:
                if st.button("Next ⏩", 
                            disabled=(st.session_state[page_key] == total_pages - 1),
                            key=f"{page_key}_next"):
                    st.session_state[page_key] = min(total_pages - 1, st.session_state[page_key] + 1)
                    st.rerun()

# ================ VISUALIZATIONS ================
def display_enrichment_insights():
    if st.session_state.enriched_df.empty:
        return
        
    with st.container():
        st.subheader("📊 Enrichment Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'founded_year' in st.session_state.enriched_df:
                fig = px.histogram(
                    st.session_state.enriched_df, 
                    x='founded_year', 
                    title='Company Founding Years',
                    color_discrete_sequence=[PALETTE[0]]
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'business_model' in st.session_state.enriched_df:
                model_counts = st.session_state.enriched_df['business_model'].value_counts().reset_index()
                model_counts.columns = ['Model', 'Count']
                fig = px.pie(
                    model_counts, 
                    names='Model', 
                    values='Count', 
                    title='Business Models',
                    color_discrete_sequence=PALETTE
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if 'employee_count' in st.session_state.enriched_df:
                df = st.session_state.enriched_df.copy()
                df['Size'] = pd.cut(
                    df['employee_count'],
                    bins=[0, 10, 50, 200, 500, float('inf')],
                    labels=['1-10', '11-50', '51-200', '201-500', '500+']
                )
                size_counts = df['Size'].value_counts().reset_index()
                size_counts.columns = ['Size', 'Count']
                fig = px.bar(
                    size_counts, 
                    x='Size', 
                    y='Count', 
                    title='Company Sizes',
                    color='Size',
                    color_discrete_sequence=PALETTE
                )
                st.plotly_chart(fig, use_container_width=True)

    with st.container():
        st.subheader("😊 Customer Sentiment")
        if 'sentiment_score' in st.session_state.enriched_df:
            fig = px.histogram(
                st.session_state.enriched_df, 
                x='sentiment_score', 
                nbins=10,
                title='Sentiment Distribution',
                color_discrete_sequence=[PALETTE[3]]
            )
            fig.update_layout(xaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)

def display_score_distribution():
    if st.session_state.scored_df.empty:
        return
        
    with st.container():
        st.subheader("📈 Score Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.histogram(
                st.session_state.scored_df, 
                x='lead_score', 
                nbins=20,
                title='Lead Score Distribution',
                color_discrete_sequence=[PALETTE[1]]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            priority_counts = st.session_state.scored_df['priority_level'].value_counts().reset_index()
            priority_counts.columns = ['Priority', 'Count']
            fig = px.pie(
                priority_counts, 
                names='Priority', 
                values='Count', 
                title='Priority Levels',
                color='Priority',
                color_discrete_map={'High': PALETTE[4], 'Medium': PALETTE[3], 'Low': PALETTE[2]}
            )
            st.plotly_chart(fig, use_container_width=True)

def display_scored_leads(df):
    st.markdown("### 🏆 Priority Leads")
    
    # Calculate metrics
    try:
        top_score = f"{df['lead_score'].max():.1f}" if not df.empty else "N/A"
        avg_score = f"{df['lead_score'].mean():.1f}" if not df.empty else "N/A"
        high_priority = len(df[df['priority_level'] == 'High']) if not df.empty else "N/A"
        b2b_leads = len(df[df['business_model'] == 'B2B']) if not df.empty else "N/A"
    except Exception:
        top_score = avg_score = high_priority = b2b_leads = "N/A"
    
    # Metrics in cards
    cols = st.columns(4)
    with cols[0]:
        st.metric("🚀 Top Score", top_score)
    with cols[1]:
        st.metric("📊 Average Score", avg_score)
    with cols[2]:
        st.metric("🔝 High Priority", high_priority)
    with cols[3]:
        st.metric("🤝 B2B Leads", b2b_leads)
    
    if st.button("📊 Generate Report", key="report_button"):
        with st.spinner("Generating PDF report..."):
            report_bytes = generate_report(df)
            b64 = base64.b64encode(report_bytes).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="lead_report.pdf">📥 Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Show scored leads with pagination
    display_leads_table(df, "Scored Leads", "current_scored_page", SCORED_DISPLAY_COLS)

# ================ RUN APP ================
if __name__ == "__main__":
    main()
