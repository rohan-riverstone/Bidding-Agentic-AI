from mcp.server.fastmcp import FastMCP
import json
import sys
import os
import warnings
import logging
import tempfile
import webbrowser
from datetime import date
from bs4 import BeautifulSoup, Tag, NavigableString
import re
from rapidfuzz import fuzz

# Silence noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logs.data_logging import data_logger
from views import template
from systems.pdf_tools import html_to_pdf
from systems.llm_config import proposal_change


log=data_logger()

mcp = FastMCP("Create Proposal")

def parse_html(html_content):
    return BeautifulSoup(html_content, "html.parser")

def extract_candidate_blocks(soup, tags=("tr","div","section","footer","header","p","span","td")):
    candidates = []
    for tag in soup.find_all(tags):
        text = tag.get_text(" ", strip=True)
        html = str(tag)
        if text:
            candidates.append({"text": text, "html": html, "tag": tag})
    return candidates

def find_target_block(prompt, candidates, threshold=50):
    best_score = 0
    best_block = None
    for c in candidates:
        score = fuzz.partial_ratio(prompt.lower(), c["text"].lower())
        if score > best_score:
            best_score = score
            best_block = c

    if best_block and best_score >= threshold:
        tag = best_block["tag"]
        if tag.name == "td" and tag.parent and tag.parent.name == "tr":
            return {"text": tag.parent.get_text(" ", strip=True), "html": str(tag.parent), "tag": tag.parent}
        return best_block
    return None

def locate_target_block(soup, prompt):
    candidates = extract_candidate_blocks(soup)
    return find_target_block(prompt, candidates)


def save_updated_html(rfp_id, html_content):
    proposal_temp_path = os.path.join(tempfile.gettempdir(), "proposal.html")
    with open(proposal_temp_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return proposal_temp_path

@mcp.tool(description="""
Display the proposal when user asks to display or prepare a proposal.
""")
async def display_proposal(rfp_id: str) -> str:
    """Render and display a saved proposal (from logs) as HTML and PDF."""

    try:
        # === Load JSON from logs ===
        proposal_data = log._load_logs()[rfp_id]["tools"]["quotation"]["result"]["updated_result_json"]

        # === Render HTML ===
        proposal_html = template.render_proposal(
            proposal_data,
            today=date.today().strftime("%m/%d/%Y"),
            basic=proposal_data[list(proposal_data.keys())[0]]
        )
        await html_to_pdf(proposal_html['template'], rfp_id, "proposal.pdf")

    except Exception as e:
        return f"❌ Proposal rendering failed: {e}"

    # === Save to shared html_content.json ===
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(path, "html_content.json")

    content = {"rfp_id": rfp_id, "proposal_html": proposal_html['template'], "proposal_json": proposal_data}
    with open(file_path, "w") as f:
        json.dump(content, f, indent=4)

    # === Preview in browser ===
    try:
        proposal_temp_path = os.path.join(tempfile.gettempdir(), f"proposal.html")
        with open(proposal_temp_path, "w", encoding="utf-8") as f:
            f.write(proposal_html['template'])

        # Open only once per rfp_id
        if not getattr(display_proposal, "opened", {}):
            display_proposal.opened = {}
        if rfp_id not in display_proposal.opened:
            webbrowser.open(f"file://{proposal_temp_path}")
            display_proposal.opened[rfp_id] = True
        log.log_proposal(rfp_id, {
            "proposal_html": proposal_html['template'],
            "json_data": proposal_html['data'],
            "updated_proposal_html": proposal_html['template']
        })

    except Exception as e:
        print(f"⚠️ Preview failed: {e}")

    return f"✅ Proposal displayed successfully for RFP {rfp_id}."
def detect_action(prompt: str) -> str:
    pl = (prompt or "").lower()
    if pl.startswith("add") or " add " in pl or "append" in pl:
        return "add"
    if any(x in pl for x in ["remove", "delete"]):
        return "remove"
    # update patterns like "change X to Y" / "update X to Y"
    if re.search(r'\b(change|update|replace)\b', pl):
        # if it contains "to" or "as" it's likely an update with new value
        if re.search(r'\b(to|as|=)\b', pl):
            return "update"
        return "update"
    return "update"

def extract_add_text(prompt: str) -> str:
    # prefer quoted text
    q = re.search(r'["\'](.+?)["\']', prompt)
    if q:
        return q.group(1).strip()
    # common add patterns: "add X", "append X"
    m = re.search(r'(?:add|append)\s+(?:an?|another|the)?\s*(.+)$', prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: last 6 words
    parts = prompt.split()
    return " ".join(parts[-6:]).strip()

def extract_remove_text(prompt: str) -> str:
    q = re.search(r'["\'](.+?)["\']', prompt)
    if q:
        return q.group(1).strip()
    m = re.search(r'(?:remove|delete)\s+(?:the\s+)?(.+)$', prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    parts = prompt.split()
    return " ".join(parts[-6:]).strip()

def extract_update_texts(prompt: str):
    # try "change X to Y" / "update X to Y" / "replace X with Y"
    m = re.search(r'(?:change|update|replace)\s+(.+?)\s+(?:to|as|with|=)\s+[\'"]?(.+?)[\'"]?$', prompt, re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # fallback: if quoted two groups: "old" -> "new"
    qs = re.findall(r'["\'](.+?)["\']', prompt)
    if len(qs) >= 2:
        return qs[0].strip(), qs[1].strip()
    return None, None

def find_best_li(list_tag: Tag, target_text: str, threshold: int = 55):
    """
    Returns (li_tag, score) for best matching <li> inside list_tag.
    """
    best_score = 0
    best_li = None
    for li in list_tag.find_all('li', recursive=False):
        txt = li.get_text(" ", strip=True)
        score = fuzz.partial_ratio(txt.lower(), target_text.lower())
        if score > best_score:
            best_score = score
            best_li = li
    if best_score >= threshold:
        return best_li, best_score
    return None, best_score
def get_edit_target(tag: Tag) -> Tag:
    """
    Expand the matched element into the right editable block.

    - If matched tag is <td> -> return <tr>
    - If matched tag is <li> -> return that <li> (caller can decide)
    - If matched tag is a title DIV/H2/H3/H4 and next sibling is UL/OL -> return the UL/OL
    - If matched tag itself is UL/OL -> return it
    - Otherwise return the tag itself
    """
    if not isinstance(tag, Tag):
        return tag

    # If inside a table cell -> bubble to the row
    if tag.name == "td" and tag.parent and getattr(tag.parent, "name", "") == "tr":
        return tag.parent

    # If tag is span inside a td -> try to bubble up
    if tag.name == "span" and tag.parent and getattr(tag.parent, "name", "") == "td":
        tr = tag.parent.parent
        if tr and getattr(tr, "name", "") == "tr":
            return tr
        return tag.parent

    # If tag is li -> we operate on the list (return parent) OR allow li-specific ops
    if tag.name == "li":
        return tag  # caller can choose to operate on parent list or this li

    # If tag already a list -> return it directly
    if tag.name in ("ul", "ol"):
        return tag

    # If it's a section heading / title, check immediate next siblings for a list
    if tag.name in ("div", "h2", "h3", "h4", "h1"):
        # skip if tag is itself a list-like header but not a textual header? we treat generically
        # look at immediate next sibling(s), skipping whitespace/text nodes
        nxt = tag.find_next_sibling()
        while nxt and (isinstance(nxt, NavigableString) or (getattr(nxt, "name", None) is None)):
            nxt = nxt.find_next_sibling()
        if nxt and getattr(nxt, "name", "") in ("ul", "ol"):
            return nxt
        # sometimes list is wrapped in a div; check within next few siblings for nearest ul/ol before next header/div.section
        cursor = tag.find_next_sibling()
        steps = 0
        while cursor and steps < 6:
            if isinstance(cursor, Tag) and cursor.name in ("ul", "ol"):
                return cursor
            # stop if we hit another major section header
            if isinstance(cursor, Tag) and cursor.name in ("div", "section", "header", "footer", "h2", "h3", "h4"):
                break
            cursor = cursor.find_next_sibling()
            steps += 1

    # default: return tag itself
    return tag

@mcp.tool(description="""after creating proposal when user ask to make chaneges in the proposal call this tool with rfp_id and user query
        user_queries is a list of strings containing the changes to be made in the proposal""")
def make_changes_in_proposal(rfp_id: str, user_queries: list) -> str:
    """
    For each user_query:
      - locate target block (existing function find_target_block)
      - expand using get_edit_target(...)
      - if edit_target is a list (ul/ol) -> perform local action (add/remove/update)
      - else -> fallback to proposal_change(...) (LLM) like before
    """
    result = log._load_logs()[rfp_id]["tools"]["proposal"]["result"]
    html_content = result["updated_proposal_html"]

    for user_query in user_queries:
        soup = parse_html(html_content)
        candidates = extract_candidate_blocks(soup)
        target = find_target_block(user_query, candidates)

        if not target:
            print("⚠️ No matching block found for:", user_query)
            continue

        # target is a dict {"text":..., "html":..., "tag": Tag}
        base_tag = target["tag"]
        edit_target = get_edit_target(base_tag)  # this returns a Tag

        action = detect_action(user_query)

        print(f"\n🔎 Matched Block Before Edit (tag: <{getattr(edit_target,'name',type(edit_target))}>):")
        # print a concise preview
        preview = edit_target.get_text(" ", strip=True) if isinstance(edit_target, Tag) else str(edit_target)
        print("Preview text:", preview[:250])

        # If edit_target is a list, do local list operations
        if isinstance(edit_target, Tag) and edit_target.name in ("ul", "ol"):
            ul = edit_target

            if action == "add":
                new_item_text = extract_add_text(user_query)
                if not new_item_text:
                    print("⚠️ Could not parse new list item from query:", user_query)
                else:
                    new_li = soup.new_tag("li")
                    new_li.string = new_item_text
                    ul.append(new_li)
                    print(f"➕ Added list item: '{new_item_text}'")

            elif action == "remove":
                target_text = extract_remove_text(user_query)
                # try to find best li match
                li_tag, score = find_best_li(ul, target_text)
                if li_tag:
                    removed_text = li_tag.get_text(" ", strip=True)
                    li_tag.decompose()
                    print(f"➖ Removed list item (score={score}): '{removed_text}'")
                else:
                    print("⚠️ No matching list item found to remove for:", target_text)

            elif action == "update":
                old_text, new_text = extract_update_texts(user_query)
                if old_text and new_text:
                    # find old li
                    li_tag, score = find_best_li(ul, old_text)
                    if li_tag:
                        li_tag.string.replace_with(new_text)
                        print(f"✏️ Updated list item (score={score}): '{old_text}' -> '{new_text}'")
                    else:
                        # fallback: try to match old_text with fuzzy against li's or, if old_text not present,
                        # maybe user intended to append new item instead of update
                        print("⚠️ Could not find existing list item to update for:", old_text)
                else:
                    # no explicit "old -> new" found: try to match any li in prompt and replace with remainder
                    # e.g., "update BIFMA Certified to BIFMA Gold" already handled above; otherwise skip.
                    print("⚠️ Could not parse update pair from query:", user_query)
            else:
                print("⚠️ Unknown action for list:", action)

            # After local modification, update html_content and continue to next query
            html_content = str(soup)

        else:
            # Non-list case: fallback to proposal_change (LLM) if available
            try:
                updated_html = proposal_change(user_query, str(edit_target), action)
                new_block = BeautifulSoup(updated_html, "html.parser")
                # replace edit_target in the original soup
                edit_target.replace_with(new_block)
                html_content = str(soup)
                print("🔄 Performed LLM-based update for non-list element.")
            except Exception as e:
                print("❌ proposal_change failed; error:", e)
                continue

    # Save back to logs after all edits
    result["updated_proposal_html"] = html_content
    log.log_proposal(rfp_id=rfp_id, result=result)

    # Save updated HTML to disk (display file)
    save_updated_html(rfp_id, html_content)

    return "Changes made successfully."
    
# ===== START SERVER =====
if __name__ == "__main__":
    try:
        import asyncio
        print("✅ Starting MCP Server...", file=sys.stderr)
        mcp.run()
        # print(make_changes_in_proposal(rfp_id='474c5d7aafd4aa6da6ad0a948a98c615c8f20581593c64ff607aa000f4d02735', user_query = 'change the authorization level of Manufacturer 1 to Basic'))
        # result = asyncio.run(display_proposal(rfp_id='474c5d7aafd4aa6da6ad0a948a98c615c8f20581593c64ff607aa000f4d02735'))
        # print(result)
#         print(make_changes_in_proposal(rfp_id= '474c5d7aafd4aa6da6ad0a948a98c615c8f20581593c64ff607aa000f4d02735',user_queries= [
#     'change the Shipment Charges to $ 1500.00',
#     "add one more Additional Services Provided by Dealer: 'On-site training for staff'"
#   ]))
    except Exception as ex:
        print(f"❌ MCP Server failed: {str(ex)}", file=sys.stderr)