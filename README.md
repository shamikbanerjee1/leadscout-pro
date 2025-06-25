# 🚀 LeadScout Pro

**AI-Powered Lead Generation, Enrichment, and Prioritization**

LeadScout Pro is an intelligent B2B lead generation tool built with Streamlit, OpenAI, and scikit-learn. It enables users to find, enrich, and prioritize company leads using AI-powered insights, realistic review sentiment analysis, and dynamic lead scoring algorithms.

---

## 📦 Features

### 🔍 1. Lead Generation
- Uses **GPT-4 Turbo** to simulate realistic lead scraping from Google Maps
- Retrieves essential fields like:
  - Company Name, Address, Phone, Website, Category
  - Google Maps search link

### ✨ 2. Lead Enrichment
- Uses GPT to estimate:
  - 💰 Revenue & Funding
  - 👥 Employee Count
  - 📅 Founded Year
  - 🔍 Industry, 🤝 Business Model
  - 🚀 Startup Status
- Adds **realistic review texts**
- Performs **sentiment analysis** using `TextBlob`

### 🧠 3. Lead Scoring
- Weighted scoring based on:
  - Estimated Revenue (25%)
  - Funding (20%)
  - Sentiment Score (25%)
  - Employee Count, Founded Year, Business Model
- Adjusts weights **dynamically** using feature variability (CV)

### 📊 4. Interactive UI & Insights
- Built in **Streamlit** with custom CSS
- Dashboards for:
  - Business models
  - Company sizes
  - Review sentiment distribution
  - Lead score and priority levels

### 📤 5. Export & Reports
- Export prioritized leads to **Salesforce CSV**
- Generate PDF summary reports for stakeholders

---

## ⚙️ Installation & Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/your-username/leadscout-pro.git
   cd leadscout-pro

2. **Install dependencies**
    ```bash
   pip install -r requirements.txt
   
3. Set your OpenAI API key

4. **Run the app**
    ```bash
   streamlit run app.py




📜 License
MIT License. © 2025 Shamik Banerjee
