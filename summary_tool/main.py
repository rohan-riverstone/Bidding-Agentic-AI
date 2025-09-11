
import os
import sys
import json
import warnings
import logging
from typing import Optional, Dict, Any

# Silence noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

from mcp.server.fastmcp import FastMCP
from datetime import datetime
from rapidfuzz import fuzz, process

mcp = FastMCP("summarize the pdf")

# add root path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from systems.llm_config import chunking
from logs.data_logging import data_logger

log = data_logger()

def normalize_date(date_str: str) -> str:
    try:
        # ✅ Case 1: Already in "April 12, 2024" format → return as-is
        datetime.strptime(date_str, "%B %d, %Y")
        return date_str
    except ValueError:
        pass  # Not in the correct format, try next
    
    try:
        # ✅ Case 2: Convert from "12 April 2024" to "April 12, 2024"
        date_obj = datetime.strptime(date_str, "%d %B %Y")
        return date_obj.strftime("%B %d, %Y")
    except ValueError:
        return "Invalid date format"

CACHE_FILE = "company_names.json"

def _cache_path():
    # store cache next to this file
    return os.path.join(os.path.dirname(__file__), CACHE_FILE)

def load_cache():
    path = _cache_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            raw = json.load(f)
            return raw
    except Exception:
        return []

def save_cache(cache_list):
    path = _cache_path()
    with open(path, "w") as f:
        json.dump({"names": cache_list}, f, indent=2)

def normalize_org_name(input_name: str) -> str:
    # Simulating your cache
    cache = load_cache()
    cache_names = cache

    # Work with the actual list of names
    existing_names = cache_names.get("names", [])

    # Use fuzzy matching against cache
    match, score, idx = process.extractOne(
        input_name,
        existing_names,
        scorer=fuzz.token_sort_ratio
    ) if existing_names else (None, 0, None)

    if match and score > 70:  # ✅ threshold for similarity
        return match  # return normalized cached name

    # ❌ If not found in cache, add it
    cache_names["names"].append(input_name)
    save_cache(cache_names["names"])
    # save_cache(cache_names)
    return input_name
    
@mcp.tool(description="""
Extracts all necessary information from a furniture RFP PDF for preparing a vendor bid.

Covers: general info, dates, scope, furniture specifications, eligibility, 
evaluation criteria, financial terms, submission instructions, contacts, 
special conditions, annexes, and any other relevant details.

Dates must be normalized to "Month DD, YYYY".  
Runs automatically on PDF upload, even without user input.
""")

def summarize_pdf_content(
    content: str,
    document_name: str,
    rfp_number: Optional[str] = "",
    issue_date: Optional[str] = "",
    client_name: Optional[str] = ""
) -> Dict[str, Any]:
    """
    Summarize an RFP PDF into structured sections.
    """
    if not content or not content.strip():
        return {"error": "❌ No PDF content provided."}

    try:
        # Ensure storage JSON is ready
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(path, "html_content.json")

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                json_data = json.load(f)
        else:
            json_data = {}

        # Initialize placeholders for later stages
        json_data.setdefault("quotation", {})
        json_data.setdefault("cutsheet", {})

        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)

        # Prompt for summarization
        summarization_prompt = """
Analyze the furniture RFP and return ONLY valid JSON:

{
  "executive_summary": "5–7 sentences covering project background, scope, objectives, key dates, requirements, and evaluation approach.",
  "important_dates": [],
  "evaluation_criteria": [],
  "financial_terms": {},
  "contact_info": [],
  "furniture_requirements": [],
  "other_requirements": {}
}

Rules:
- Only include fields present in the RFP; else leave empty.
- Executive summary must be a full paragraph, not less than 5 sentences.
- Dates → "Month DD, YYYY".
- Be concise and factual, no extra text outside JSON.
"""

        # Run through LLM with chunking
        t = chunking(content)
        result = t.invoke({"query": summarization_prompt})
        summary = result.get("result", "")
        

        # Log extracted summary with raw text
        rfp_id = log.log_rfp(
            document_name=document_name,
            rfp_number=rfp_number or "",
            issue_date=normalize_date(issue_date) or "",
            client_name=normalize_org_name(client_name) or "",
            extracted_data={
                "summary": summary,
                "raw_text": content
            }
        )

        return {
            "rfp_id": rfp_id,
            "summary": summary
        }

    except Exception as e:
        return {"error": f"❌ Error during summarization: {str(e)}"}

    
# ===== START SERVER =====
if __name__ == "__main__":
    try:
        print("✅ Starting MCP Server...")
        mcp.run()
#         print(summarize_pdf_content(
#             content = """This is a Request for Proposal (RFP) from the International Organization for Migration (IOM) Washington DC office for new office furniture, electrical and networking services. 

# **Key Details:**
# - RFP Reference: IOM-WAS-RFP/2024-0001
# - Issue Date: April 12, 2024
# - Country: United States
# - Client: IOM Washington DC Office

# **Project Background:**
# IOM's Washington DC office is moving to a new location on October 1, 2024, to reduce office footprint and needs furniture for the new space. They have worked with an architect and design firm to develop the space and identify furniture requirements.

# **Scope of Services:**
# - Supply of office furniture as specified in detailed requirements
# - Electrical and networking cable management pass-through requirements
# - Planters based on design test fits
# - Moving services from current office to new location
# - Delivery and installation services

# **Key Dates:**
# - Proposal confirmation deadline: April 23, 2024, 5:00 PM EST
# - Pre-proposal conference: April 23, 2024, 10:30 AM EST (not mandatory)
# - Questions deadline: April 23, 2024, 5:00 PM EST  
# - Proposal submission deadline: May 7, 2024, 12:00 PM EST
# - Expected contract start: June 3, 2024
# - Move-in deadline: October 1, 2024

# **Furniture Requirements:**
# The RFP includes detailed specifications for various furniture items including:
# - Conference tables (various sizes)
# - Nesting chairs (50 black)
# - Lateral files (5-drawer, 36\" and 42\")
# - Mobile pedestals with black cushions (118 units)
# - Power workstations (30x72, 30x84, 30x60)
# - Screen dividers for workstations
# - Task chairs, pantry furniture, reception furniture
# - Counter stools and various office furniture items

# **Reused Items:**
# IOM will reuse some existing furniture including:
# - 27 conference chairs
# - 72 desk and workbench chairs  
# - 6 small circular tables
# - 14 club chairs
# - 46 task chairs
# - 30 desks for private offices
# - 14 nesting chairs
# - 43 mobile pedestals with cushions

# **Financial Terms:**
# - Currency: USD
# - Payment: Net 30 days after delivery and approval
# - Prices exclusive of VAT and other indirect taxes
# - Proposal validity: 120 days
# - Liquidated damages: 0.07% per week of delay up to 10% maximum

# **Evaluation Criteria:**
# - Combined scoring method: 60% technical, 40% financial
# - Minimum technical threshold: 55% of maximum points
# - Technical evaluation covers: proposer qualifications (30 points), methodology (55 points), management structure (15 points)

# **Eligibility:**
# - Open to local and international bidders with valid US business registration
# - Must comply with UN Supplier Code of Conduct
# - Various exclusion criteria apply (sanctions lists, conflicts of interest, etc.)

# **Submission Requirements:**
# - Separate emails for technical and financial proposals
# - PDF format, maximum 10MB per file
# - Technical proposal must not contain pricing information
# - Multiple forms required including proposer information, technical approach, financial breakdown

# **Contact Information:**
# - Focal Person: Thomas Truong
# - Email: ttruong@iom.int (for clarifications only)
# - Proposal submission: wasprocurement@iom.int
# - Address: 1625 Massachusetts Avenue NW, Suite 500, Washington, DC 20036""",
#   document_name = 'IOM Washington DC Furniture RFP 2024-0001',
#   rfp_number = 'IOM-WAS-RFP/2024-0001',
#   client_name = 'International Organization for Migration (IOM) Washington DC',
#   issue_date = 'April 12, 2024'
#             ))
    except Exception as ex:
        print(f"❌ MCP Server failed: {str(ex)}")