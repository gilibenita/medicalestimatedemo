from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import os
import json
import re
from openai import OpenAI


def _plain_language_cpt_help(cpt: str, description: str) -> str:
    cpt_str = (str(cpt) if cpt is not None else '').strip()
    desc = (str(description) if description is not None else '').strip()
    d = desc.lower()

    # Common office visit codes (Evaluation & Management)
    if re.match(r"^9920[2-5]$", cpt_str) or re.match(r"^9921[2-5]$", cpt_str) or ('office o/p' in d and 'visit' in d):
        return "Office visit with a clinician (evaluation & management)."

    # Acute URI testing (keep practical, patient-friendly)
    if ('strep' in d) and any(x in d for x in [' ag', 'antigen', 'eia']):
        return "Rapid strep throat test (antigen)."

    has_covid = any(x in d for x in ['covid', 'sarscov', 'sars-cov', 'coronavirus', 'sars'])
    has_flu = 'influenza' in d or re.search(r"\bflu\b", d) is not None
    has_antigen = any(x in d for x in [' ag', 'antigen', 'rapid', ' ia', 'if'])
    has_pcr = any(x in d for x in ['pcr', 'amplified', 'probe'])

    if has_covid and has_flu and has_antigen:
        return "Rapid antigen test for COVID-19 and influenza A/B."
    if has_covid and has_antigen:
        return "Rapid antigen test for COVID-19."
    if has_covid and has_pcr:
        return "Lab test for COVID-19 (PCR / molecular)."
    if has_flu and has_antigen:
        return "Rapid influenza test (antigen)."

    # Imaging (generic plain English)
    if 'xray' in d or 'x-ray' in d or 'radiograph' in d:
        return "X-ray imaging."
    if 'ct' in d or 'computed tomography' in d:
        return "CT scan (a type of detailed X-ray imaging)."
    if 'mri' in d or 'magnetic resonance' in d:
        return "MRI scan (detailed imaging)."
    if 'ultrasound' in d or 'sonogram' in d:
        return "Ultrasound imaging."

    # Fallback: expand a few common abbreviations without over-claiming.
    expansions = []
    if ' ag' in d or 'ag,' in d:
        expansions.append('AG = antigen')
    if ' ia' in d:
        expansions.append('IA = immunoassay')
    if 'eia' in d:
        expansions.append('EIA = enzyme immunoassay')
    if 'pcr' in d:
        expansions.append('PCR = molecular test')
    if expansions:
        return "Plain-English note: " + "; ".join(expansions) + "."
    return "Plain-English summary not available for this procedure."


def _render_cpt_info(label: str, cpt: str, description: str, key: str) -> None:
    cpt_str = (str(cpt) if cpt is not None else '').strip()
    help_text = _plain_language_cpt_help(cpt, description)

    col_a, col_b = st.columns([20, 1], gap="small")
    state_key = f"cpt_info_open::{key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    with col_a:
        st.caption(f"{label}: {cpt_str}")
        if st.session_state[state_key]:
            st.caption(help_text)
    with col_b:
        # Use a keyed toggle button (Streamlit popover doesn't support keys in 1.52.2,
        # which can break pages with repeated ⓘ controls).
        if st.button("ⓘ", key=f"cpt_info_btn::{key}", help="Plain-English explanation"):
            st.session_state[state_key] = not st.session_state[state_key]


def search_procedures_ai(query: str, df: pd.DataFrame, price_col: str, max_results: int = 5):
    """AI-assisted semantic-ish search over procedure descriptions.

    Returns (results, debug_info) where results is a list of dicts with keys: cpt, description, price.
    """
    debug_info = {'query': query}
    q = (query or '').strip()
    if not q:
        return [], debug_info

    api_key = _get_openai_api_key()
    if not api_key:
        debug_info['error'] = 'missing_api_key'
        return [], debug_info

    # Narrow candidates locally first (cheap token overlap) to keep the prompt small.
    q_tokens = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) >= 3]
    stop = {
        'with', 'without', 'and', 'the', 'for', 'from', 'this', 'that', 'test', 'scan', 'mri', 'ct', 'xray', 'x-ray',
        'w', 'wo', 'w/o', 'wo/', 'contrast', 'noncontrast', 'non', 'iv', 'or'
    }
    # Keep modality tokens, but de-emphasize them for filtering
    filter_tokens = [t for t in q_tokens if t not in stop]

    desc = df['description'].astype(str)
    candidates = df
    if filter_tokens:
        pattern = '|'.join(re.escape(t) for t in filter_tokens[:8])
        mask = desc.str.contains(pattern, case=False, na=False, regex=True)
        subset = df[mask]
        if len(subset) > 0:
            candidates = subset

    # Rank candidates by token overlap and a few modality hints
    def _local_score(text: str) -> int:
        t = (text or '').lower()
        score = 0
        for tok in filter_tokens:
            if tok in t:
                score += 10
        if 'mri' in q.lower() and 'mri' in t:
            score += 15
        if 'ct' in q.lower() and re.search(r"\bct\b|computed tomography", t):
            score += 15
        if any(x in q.lower() for x in ['xray', 'x-ray']) and ('xray' in t or 'x-ray' in t or 'radiograph' in t):
            score += 15
        if 'spine' in q.lower() and 'spine' in t:
            score += 10
        return score

    sample = candidates[['cpt code number', 'description', price_col]].copy()
    sample['__score'] = sample['description'].astype(str).map(_local_score)
    sample = sample.sort_values(['__score'], ascending=False)
    sample = sample.head(250)
    debug_info['candidate_pool'] = int(len(sample))

    # Build indexed options for the model
    rows = []
    records = sample.to_dict('records')
    for i, r in enumerate(records):
        price_val = r.get(price_col)
        price_txt = f", ${float(price_val):,.2f}" if pd.notna(price_val) else ""
        rows.append(f"{i}. {r.get('description')} (CPT: {r.get('cpt code number')}{price_txt})")
    options_text = "\n".join(rows)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": """You map user search phrases to the closest matching CPT procedure descriptions.

Return ONLY a JSON array of 1-5 integers (indices). Prefer the most exact anatomical match and modality.
If multiple variants exist (e.g., with vs without contrast), return a few best options.""",
            },
            {
                "role": "user",
                "content": f"User search: {q}\n\nAvailable procedures (indexed):\n{options_text}\n\nReturn ONLY a JSON array of indices.",
            },
        ],
        temperature=0.2,
        max_tokens=120,
    )
    content = (response.choices[0].message.content or '').strip()
    debug_info['model_response'] = content[:300]

    try:
        idxs = json.loads(content)
        if not isinstance(idxs, list):
            raise ValueError('not_list')
        picked = []
        for idx in idxs:
            if isinstance(idx, bool):
                continue
            if not isinstance(idx, int):
                continue
            if 0 <= idx < len(records):
                picked.append(records[idx])
        picked = picked[:max_results]
    except Exception:
        debug_info['error'] = 'parse_failed'
        return [], debug_info

    results = []
    for r in picked:
        price_val = r.get(price_col)
        if pd.isna(price_val):
            price_val = None
        results.append({
            'cpt': r.get('cpt code number'),
            'description': r.get('description'),
            'price': price_val,
        })

    return results, debug_info

# Page config
st.set_page_config(
    page_title="NYU Langone Cost Estimator",
    page_icon="🏥",
    layout="wide"
)

# Load rules from optional rules.json with sensible defaults
@st.cache_resource
def load_rules():
    default = {
        "categories": {
            "uri_ent": {
                "match_any": [
                    "throat", "sore throat", "pharyngitis", "tonsillitis", "cough", "fever", "chills",
                    "congestion", "runny nose", "headache", "sinus", "flu", "influenza", "covid"
                ],
                "prefer": [
                    "office", "visit", "evaluation", "e/m", "strep", "rapid strep", "throat culture",
                    "flu", "influenza", "covid", "antigen", "pcr", "respiratory", "panel"
                ],
                "avoid_without_red_flags": ["ct", "mri"]
            },
            "musculoskeletal": {
                "match_any": [
                    "ankle", "knee", "hip", "shoulder", "elbow", "wrist", "hand", "foot",
                    "sprain", "fracture", "injury", "swelling"
                ],
                "prefer": ["xray", "x-ray", "radiograph", "xr", "ankle", "knee", "hip", "shoulder", "elbow", "wrist", "hand", "foot"]
            },
            "chest": {
                "match_any": ["chest pain", "shortness of breath", "sob"],
                "prefer": ["xray", "x-ray", "chest", "ecg", "ekg", "troponin"]
            },
            "abdominal": {
                "match_any": ["abdominal", "stomach pain", "belly pain", "nausea", "vomit"],
                "prefer": ["ultrasound", "ct", "abdomen", "cbc", "cmp", "lipase", "urinalysis"]
            },
            "general": {"match_any": [], "prefer": ["visit", "evaluation", "exam"]}
        },
        "red_flags": [
            "worst headache", "thunderclap", "focal weakness", "slurred speech", "confusion",
            "loss of consciousness", "seizure", "stiff neck", "meningitis", "neuro deficit",
            "severe head injury", "high fever", "immunocompromised", "cancer", "anticoagulant"
        ]
    }
    try:
        if os.path.exists('rules.json'):
            with open('rules.json', 'r') as f:
                data = json.load(f)
                # Merge categories and red_flags if present
                if isinstance(data, dict):
                    if 'categories' in data and isinstance(data['categories'], dict):
                        default['categories'].update(data['categories'])
                    if 'red_flags' in data and isinstance(data['red_flags'], list):
                        default['red_flags'] = data['red_flags']
    except Exception:
        # Ignore issues and return defaults
        pass
    return default

# Helper: determine if a key value is real vs placeholder
def _is_valid_key(key: str) -> bool:
    try:
        k = str(key).strip()
    except Exception:
        return False
    if not k:
        return False
    # Ignore known placeholders
    up = k.upper()
    if 'REPLACE' in up or 'YOUR_REAL_KEY' in up:
        return False
    # Common prefixes for OpenAI keys
    if k.startswith('sk-') or k.startswith('sk-proj-'):
        return True
    # Allow other formats silently
    return True

# Helper: retrieve OpenAI API key from Streamlit secrets or environment
def _get_openai_api_key():
    candidate_names = [
        'OPENAI_API_KEY',
        'OPENAI_API_KEY_12_18_2025',  # supports your current secret name
    ]

    # Prefer Streamlit secrets for local/dev setups
    try:
        if hasattr(st, 'secrets'):
            for name in candidate_names:
                key = st.secrets.get(name)
                if key and _is_valid_key(key):
                    return key
    except Exception:
        pass

    # Fallback to environment variables
    for name in candidate_names:
        key = os.getenv(name)
        if key and _is_valid_key(key):
            return key

    return None

# Load the pricing data automatically
@st.cache_data
def load_pricing_data():
    """Load the NYU pricing CSV that's stored in the project"""
    try:
        # Detect header row by scanning the first lines for expected column names
        header_row = None
        with open('nyu_prices.csv', 'r', encoding='utf-8', errors='ignore') as f:
            for i in range(50):
                line = f.readline()
                if not line:
                    break
                l = line.lower()
                # look for a line that includes both description and a price-like header
                if 'description' in l and ('discount' in l or 'charge' in l or 'charg' in l):
                    header_row = i
                    break

        if header_row is not None:
            df = pd.read_csv('nyu_prices.csv', header=header_row)
        else:
            df = pd.read_csv('nyu_prices.csv')

        # Clean up the data
        df.columns = df.columns.str.strip().str.lower()
        # Find the price column robustly (handle minor typos like 'Standart')
        candidates = [c for c in df.columns if 'discount' in c and 'charg' in c]
        if candidates:
            price_col = candidates[0]
        else:
            st.error("Price column not found in CSV (expected something like 'Discounted Standard Charges').")
            return None

        # Ensure price column is numeric
        df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')

        # Return both dataframe and detected price column name
        return df, price_col
    except FileNotFoundError:
        st.error("⚠️ CSV file not found! Make sure 'nyu_prices.csv' is uploaded to your project.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Helper: extract medical keywords from symptom text
def _extract_medical_keywords(symptom_text):
    """Extract relevant medical keywords from symptom description."""
    symptom_lower = symptom_text.lower()
    
    # Common symptom/body part and modality keywords
    keywords = [
        # MSK
        'ankle', 'knee', 'hip', 'shoulder', 'elbow', 'wrist', 'hand', 'foot',
        'back', 'spine', 'neck', 'head', 'chest', 'abdomen',
        'fracture', 'sprain', 'strain', 'tear', 'pain', 'injury', 'swelling',
        # Respiratory/ENT
        'throat', 'sore throat', 'pharyngitis', 'tonsillitis', 'cough', 'fever', 'chills', 'congestion', 'runny nose',
        'headache', 'migraine', 'sinus', 'ear', 'otitis', 'flu', 'influenza', 'covid', 'pcr', 'strep', 'rapid', 'antigen', 'culture',
        # Diagnostics
        'xray', 'x-ray', 'mri', 'ct', 'ultrasound', 'imaging', 'test', 'exam', 'laboratory', 'cbc', 'panel',
        # Procedures/treatments
        'procedure', 'treatment', 'injection', 'drainage', 'repair', 'surgery', 'visit', 'evaluation'
    ]
    
    found_keywords = [kw for kw in keywords if kw in symptom_lower]
    return found_keywords if found_keywords else ['procedure', 'exam']

# Helper: detect red-flag symptoms where advanced imaging might be warranted
def _has_red_flags(symptom_text: str) -> bool:
    s = symptom_text.lower()
    rules = load_rules()
    red_flag_terms = rules.get('red_flags', [])
    return any(term in s for term in red_flag_terms)

# Helper: categorize the case for rule-based curation
def _categorize_case(symptom_text: str) -> str:
    s = symptom_text.lower()
    rules = load_rules()
    cats = rules.get('categories', {})
    for name, cfg in cats.items():
        match_any = cfg.get('match_any', [])
        if any(x in s for x in match_any):
            return name
    return 'general'

# Helper: curated candidate selection based on category/keywords
def _curate_candidates(df: pd.DataFrame, price_col: str, category: str, red_flags: bool) -> pd.DataFrame:
    desc = df['description'].astype(str)
    rules = load_rules()
    cfg = rules.get('categories', {}).get(category, {})
    prefer = cfg.get('prefer', [])
    avoid = [] if red_flags else cfg.get('avoid_without_red_flags', [])

    mask_prefer = pd.Series(False, index=df.index)
    for kw in prefer:
        mask_prefer |= desc.str.contains(kw, case=False, na=False)

    if avoid:
        for kw in avoid:
            mask_prefer &= ~desc.str.contains(kw, case=False, na=False)

    candidates = df[mask_prefer]
    if len(candidates) == 0:
        return df.head(200)
    return candidates.head(200)

# Helper function for symptom-based suggestions (define before use)
def get_procedure_suggestions(symptom_text, df, price_col):
    """
    AI-powered symptom-to-procedure matching using OpenAI API.
    Two-stage approach:
    1. Pre-filter procedures based on symptom keywords
    2. Use AI to intelligently rank relevant procedures
    """
    suggestions = []
    debug_info = {}

    def _score_for_acute_triage(desc: str) -> int:
        d = (desc or "").lower()
        score = 0
        # Prefer point-of-care / acute diagnostics (antigen/rapid > PCR)
        if any(x in d for x in ["rapid", "antigen", " ag", "ag,"]):
            score += 50
        elif any(x in d for x in ["pcr", "amplified", "probe"]):
            score += 25
        if any(x in d for x in ["ia", "if"]):
            score += 10
        if any(x in d for x in ["covid", "sars-cov", "sarscov", "coronavirus", "influenza", "flu", "strep", "throat"]):
            score += 10
        # De-prioritize serology/antibody and uncommon immune tests for typical acute URI
        if any(x in d for x in ["antibody", "igg", "igm", "titer", "complement", "antistreptolysin", "aso", "neutraliz", "neutrlzg"]):
            score -= 50
        return score

    def _pick_best_row(candidates: pd.DataFrame) -> pd.Series | None:
        if candidates is None or len(candidates) == 0:
            return None
        usable = candidates[pd.notna(candidates[price_col])]
        if len(usable) == 0:
            return None
        scored = usable.copy()
        scored["__score"] = scored["description"].astype(str).map(_score_for_acute_triage)
        scored = scored.sort_values(["__score"], ascending=False)
        return scored.iloc[0]
    
    try:
        # Initialize OpenAI client with layered key lookup
        api_key = _get_openai_api_key()
        if not api_key:
            st.error("⚠️ OpenAI API key not configured. Expected one of: OPENAI_API_KEY, OPENAI_API_KEY_12_18_2025 (env or st.secrets).")
            return suggestions

        client = OpenAI(api_key=api_key)
        
        # Stage 1: categorize + curated candidates, then optional keyword filter
        category = _categorize_case(symptom_text)
        red_flags = _has_red_flags(symptom_text)
        filtered_df = _curate_candidates(df, price_col, category, red_flags)
        keywords = _extract_medical_keywords(symptom_text)
        debug_info.update({
            'category': category,
            'case_category': category,
            'red_flags': red_flags,
            'candidate_count': int(len(filtered_df)),
            'keywords': keywords,
        })
        # Preserve original curated pool for post-processing (COVID, E/M)
        curated_pool_df = filtered_df.copy()
        if keywords:
            keyword_pattern = '|'.join(keywords)
            kmask = filtered_df['description'].astype(str).str.contains(keyword_pattern, case=False, na=False)
            subset = filtered_df[kmask]
            if len(subset) > 0:
                filtered_df = subset
        
        # Stage 2: Get unique procedures from filtered dataframe
        procedures_df = filtered_df[['cpt code number', 'description', price_col]].drop_duplicates(subset=['description']).reset_index(drop=True)
        procedures_list = procedures_df.to_dict('records')
        
        # Format procedures for the prompt as indexed items
        display_rows = []
        for i, p in enumerate(procedures_list):
            price_txt = f", Price: ${p[price_col]:,.2f}" if pd.notna(p[price_col]) else ""
            display_rows.append(f"{i}. {p['description']} (CPT: {p['cpt code number']}{price_txt})")
        procedures_text = "\n".join(display_rows[:500])
        
        # Call OpenAI API with the user's symptom description
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": """You are a conservative medical triage assistant. Suggest ONLY what a clinic doctor would actually do.

STRICT RULES:
1. Select up to 7 items, but ONLY if truly needed. Typical cases should be 2-4 items (office visit + 1-3 basic tests).
2. Prioritize office/E/M visit and simple point-of-care tests (strep, flu, COVID antigen) and basic labs.
3. Avoid broad/rare send-out or esoteric immune tests unless clearly indicated by symptoms.
4. NO advanced imaging (CT/MRI) unless red flags.
5. Return ONLY a JSON array of up to 7 integers (the selected item indices), no other text.
6. Example response: [0,1,3]
"""
                },
                {
                    "role": "user",
                    "content": f"Red flags present: {red_flags}\n\nPatient symptoms: {symptom_text}\n\nAvailable procedures (indexed):\n{procedures_text}\n\nSelect only what is needed for a typical clinic visit. Suggest 2-4 items for most cases; up to 7 only if justified. Return ONLY a JSON array."
                }
            ],
            temperature=0.4,
            max_tokens=300
        )
        
        # Parse the response to get procedure descriptions
        response_text = response.choices[0].message.content.strip()
        debug_info['model_response'] = response_text[:600]
        # Try structured JSON parsing (indices)
        idx_list = []
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                idx_list = [int(x) for x in parsed if isinstance(x, int) or (isinstance(x, str) and x.isdigit())][:7]
        except Exception:
            idx_list = []
        
        used = set()
        for idx in idx_list:
            if 0 <= idx < len(procedures_list):
                row = procedures_df.iloc[idx]
                cpt = row['cpt code number']
                desc = row['description']
                price = row[price_col]
                key = (str(cpt), str(desc).lower())
                if key in used or pd.isna(price):
                    continue
                used.add(key)
                suggestions.append({'cpt': cpt, 'description': desc, 'price': price})
        
        # Fallback to text parsing if needed
        if not suggestions:
            matched_descriptions = response_text.split('\n')
            matched_descriptions = [
                desc.strip().lstrip('-').lstrip('0123456789. ').strip()
                for desc in matched_descriptions if desc.strip()
            ]
            # De-duplicate while preserving order
            seen_desc = set()
            unique_descriptions = []
            for d in matched_descriptions:
                k = d.lower()
                if k and k not in seen_desc:
                    seen_desc.add(k)
                    unique_descriptions.append(d)
        
        # Look up matched procedures in the dataframe
        for matched_desc in unique_descriptions if 'unique_descriptions' in locals() else []:
            if not matched_desc or len(matched_desc) < 3:
                continue
            
            # Try exact phrase match first
            mask = filtered_df['description'].astype(str).str.contains(matched_desc, case=False, na=False, regex=False)
            results = filtered_df[mask]
            
            # If no exact match, try partial word match (first few words)
            if len(results) == 0 and len(matched_desc.split()) > 1:
                first_words = ' '.join(matched_desc.split()[:3])
                mask = filtered_df['description'].astype(str).str.contains(first_words, case=False, na=False, regex=False)
                results = filtered_df[mask]
            
            # If still no match, try matching against all words
            if len(results) == 0:
                words = matched_desc.split()
                for word in words:
                    if len(word) > 4:  # Only match meaningful words
                        mask = filtered_df['description'].astype(str).str.contains(word, case=False, na=False, regex=False)
                        results = filtered_df[mask]
                        if len(results) > 0:
                            break
            
            if len(results) > 0:
                # Take the first non-duplicate by CPT or description
                row = results.iloc[0]
                price = row[price_col]
                if pd.notna(price):
                    if not any(
                        s.get('cpt') == row['cpt code number'] or s.get('description', '').lower() == str(row['description']).lower()
                        for s in suggestions
                    ):
                        suggestions.append({
                            'cpt': row['cpt code number'],
                            'description': row['description'],
                            'price': price
                        })
        
        # Post-processing: ensure COVID test presence for URI ENT without red flags
        if category == 'uri_ent' and not red_flags:
            has_covid = any(any(x in str(s['description']).lower() for x in ['covid', 'sarscov', 'sars-cov', 'coronavirus']) for s in suggestions)
            if not has_covid:
                # Use curated pool if it contains COVID tests; otherwise fall back to the full dataset
                base_df = curated_pool_df
                base_desc = base_df['description'].astype(str)
                covid_df = base_df[
                    base_desc.str.contains('covid|sarscov|sars-cov|coronavirus', case=False, na=False, regex=True)
                ]
                if len(covid_df) == 0:
                    base_df = df
                    base_desc = base_df['description'].astype(str)
                    covid_df = base_df[
                        base_desc.str.contains('covid|sarscov|sars-cov|coronavirus', case=False, na=False, regex=True)
                    ]
                # Strongly prefer antigen-style COVID tests for acute symptoms
                covid_acute_df = covid_df[
                    covid_df['description'].astype(str).str.contains('ag|antigen|ia|pcr|amplified', case=False, na=False, regex=True)
                ]
                pick = _pick_best_row(covid_acute_df)
                if pick is None:
                    pick = _pick_best_row(covid_df)
                if pick is not None:
                    covid_item = {
                        'cpt': pick['cpt code number'],
                        'description': pick['description'],
                        'price': pick[price_col]
                    }
                    suggestions.append(covid_item)
                    debug_info['covid_adjustment'] = f"appended_{pick['cpt code number']}"

        # Post-processing: prefer acute flu antigen tests over flu antibody tests (typical clinic visit)
        if category == 'uri_ent' and not red_flags:
            has_flu_antibody = any('influenza virus antibody' in str(s['description']).lower() or 'flu antibody' in str(s['description']).lower() for s in suggestions)
            # Avoid false positives from unrelated strings like "IADNA" in COVID PCR tests.
            has_flu_acute = any(
                (('influenza' in str(s['description']).lower()) or ('flu' in str(s['description']).lower()))
                and any(x in str(s['description']).lower() for x in [' ag', 'ag,', 'flu ag', 'antigen', 'rapid'])
                for s in suggestions
            )
            if has_flu_antibody and not has_flu_acute:
                # Use curated pool if it contains acute flu tests; otherwise fall back to the full dataset
                base_df = curated_pool_df
                base_desc = base_df['description'].astype(str)
                flu_acute_df = base_df[
                    base_desc.str.contains(r'\binfluenza\b|\bflu\b', case=False, na=False, regex=True)
                    & base_desc.str.contains('ag|antigen|rapid|ia|if', case=False, na=False, regex=True)
                ]
                if len(flu_acute_df) == 0:
                    base_df = df
                    base_desc = base_df['description'].astype(str)
                    flu_acute_df = base_df[
                        base_desc.str.contains(r'\binfluenza\b|\bflu\b', case=False, na=False, regex=True)
                        & base_desc.str.contains('ag|antigen|rapid|ia|if', case=False, na=False, regex=True)
                    ]
                pick = _pick_best_row(flu_acute_df)
                if pick is not None:
                    # Replace the first flu antibody item
                    for i, s in enumerate(list(suggestions)):
                        d = str(s['description']).lower()
                        if 'influenza' in d and 'antibody' in d:
                            suggestions[i] = {
                                'cpt': pick['cpt code number'],
                                'description': pick['description'],
                                'price': pick[price_col]
                            }
                            debug_info['flu_adjustment'] = f"replaced_antibody_with_{pick['cpt code number']}"
                            break

        # If a combined COVID+flu antigen test is present (e.g., 87428), drop redundant standalone COVID-only or flu-only tests.
        # This keeps URI suggestions practical (visit + strep + combo test), and avoids double-counting influenza.
        if category == 'uri_ent' and not red_flags and len(suggestions) > 1:
            def _is_covid_flu_combo(item: dict) -> bool:
                desc = str(item.get('description', '')).lower()
                cpt = str(item.get('cpt', '')).strip()
                if cpt == '87428':
                    return True
                has_covid = any(x in desc for x in ['covid', 'sarscov', 'sars-cov', 'coronavirus', 'sars'])
                has_flu = 'influenza' in desc
                has_acute = any(x in desc for x in [' ag', 'ag,', 'antigen', 'rapid', ' ia', 'ia '])
                return has_covid and has_flu and has_acute

            has_combo = any(_is_covid_flu_combo(s) for s in suggestions)
            if has_combo:
                kept = []
                removed = 0
                for s in suggestions:
                    if _is_covid_flu_combo(s):
                        kept.append(s)
                        continue

                    desc = str(s.get('description', '')).lower()
                    is_flu_only = ('influenza' in desc) and not any(x in desc for x in ['covid', 'sarscov', 'sars-cov', 'coronavirus', 'sars'])
                    is_covid_only = any(x in desc for x in ['covid', 'sarscov', 'sars-cov', 'coronavirus', 'sars']) and ('influenza' not in desc)

                    if is_flu_only or is_covid_only:
                        removed += 1
                        continue
                    kept.append(s)
                if removed:
                    suggestions = kept
                    debug_info['combo_cleanup'] = f"removed_{removed}_redundant_tests"

        # Ensure an Evaluation & Management (office visit) is included
        has_em = any(
            ('e/m' in str(s['description']).lower()) or ('visit' in str(s['description']).lower()) or ('evaluation' in str(s['description']).lower()) or
            (str(s['cpt']).startswith('9920') or str(s['cpt']).startswith('9921'))
            for s in suggestions
        )
        if not has_em:
            # Search for typical office visit E/M codes or descriptions in curated pool
            df_desc = curated_pool_df['description'].astype(str)
            df_cpt = curated_pool_df['cpt code number'].astype(str)
            em_mask = (
                df_desc.str.contains('e/m|evaluation|visit|office', case=False, na=False, regex=True) |
                df_cpt.str.contains(r"\b9920[2-5]\b|\b9921[2-5]\b", case=False, na=False, regex=True)
            )
            em_df = curated_pool_df[em_mask]
            pick = None
            if len(em_df) > 0:
                # Prefer 99213 (established mid-level), then 99203 (new mid-level), else first
                em_df = em_df.copy()
                em_cpt = em_df['cpt code number'].astype(str)
                # Try exact CPT preferences
                pref_99213 = em_df[em_cpt.str.contains(r"\b99213\b", case=False, na=False, regex=True)]
                pref_99203 = em_df[em_cpt.str.contains(r"\b99203\b", case=False, na=False, regex=True)]
                if len(pref_99213) > 0:
                    pick = pref_99213.iloc[0]
                elif len(pref_99203) > 0:
                    pick = pref_99203.iloc[0]
                else:
                    pick = em_df.iloc[0]
            if pick is not None and pd.notna(pick[price_col]):
                em_item = {
                    'cpt': pick['cpt code number'],
                    'description': pick['description'],
                    'price': pick[price_col]
                }
                suggestions.append(em_item)
                debug_info['em_adjustment'] = 'appended'

        # For typical URI/ENT cases without red flags, keep results to a practical clinic-style shortlist.
        # (Office visit + 1-3 basic acute tests). This also naturally drops esoteric immune labs.
        if category == 'uri_ent' and not red_flags and len(suggestions) > 4:
            em_items = [
                s for s in suggestions
                if (str(s.get('cpt', '')).startswith('9920') or str(s.get('cpt', '')).startswith('9921'))
                or any(x in str(s.get('description', '')).lower() for x in ['office', 'visit', 'evaluation', 'e/m'])
            ]
            kept = []
            if em_items:
                def _em_rank(item: dict) -> int:
                    cpt = str(item.get('cpt', '')).strip()
                    if cpt == '99213':
                        return 0
                    if cpt == '99203':
                        return 1
                    if cpt.startswith('9921'):
                        return 2
                    if cpt.startswith('9920'):
                        return 3
                    return 4

                em_pick = sorted(em_items, key=_em_rank)[0]
                kept.append(em_pick)
                debug_info['em_shortlist_pick'] = str(em_pick.get('cpt', ''))

            remaining = [s for s in suggestions if s not in kept]
            # Avoid keeping multiple E/M codes in the final shortlist.
            remaining = [
                s for s in remaining
                if not (
                    str(s.get('cpt', '')).startswith('9920')
                    or str(s.get('cpt', '')).startswith('9921')
                    or any(x in str(s.get('description', '')).lower() for x in ['office', 'visit', 'evaluation', 'e/m'])
                )
            ]
            remaining_sorted = sorted(
                remaining,
                key=lambda s: _score_for_acute_triage(str(s.get('description'))),
                reverse=True,
            )
            for s in remaining_sorted:
                if len(kept) >= 4:
                    break
                kept.append(s)
            suggestions = kept
            debug_info['uri_shortlist_applied'] = True

        # Global hard cap to 7 suggestions total (keep the most acute/relevant)
        if len(suggestions) > 7:
            suggestions = sorted(
                suggestions,
                key=lambda s: _score_for_acute_triage(str(s.get('description'))),
                reverse=True,
            )[:7]
            debug_info['cap_applied'] = True

        debug_info['selected_count'] = len(suggestions)
        return suggestions, debug_info
    
    except Exception as e:
        st.error(f"⚠️ AI matching error: {str(e)}")
        return suggestions, debug_info

# Initialize the data
_loaded = load_pricing_data()
if _loaded is None:
    pricing_data = None
    price_col = None
else:
    pricing_data, price_col = _loaded

# Title
st.title("🏥 NYU Langone Cost Estimator")
st.markdown("Search for medical procedures and get instant price estimates")

if pricing_data is not None:
    # Display data info in sidebar
    with st.sidebar:
        st.header("📊 Database Info")
        st.metric("Total Procedures", f"{len(pricing_data):,}")
        st.markdown("---")
        st.markdown("### About")
        st.info("Search by procedure name or CPT code to find costs at NYU Langone Hospital.")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏥 Symptom Estimator", "🔍 Search Procedures", "💊 CPT Code Lookup", "📊 Build Estimate"])

    # TAB 1: Symptom Estimator (AI-ready)
    with tab1:
        st.header("Describe Your Symptoms or Injury")
        st.markdown("*Tell us what happened, and we'll suggest relevant procedures*")
        
        symptom_input = st.text_area(
            "Describe your situation:",
            placeholder="Example: I had a bike accident and my ankle is swollen and painful. I can't walk on it.",
            height=120,
            key="symptom_input"
        )
        
        # Run AI-powered matching with curated candidate list
        if st.button("💡 Get Treatment Estimate", type="primary"):
            if symptom_input:
                result = get_procedure_suggestions(symptom_input.lower(), pricing_data, price_col)
                if isinstance(result, tuple) and len(result) == 2:
                    suggestions, debug = result
                else:
                    suggestions = result
                    debug = None
                
                if suggestions:
                    st.success(f"Based on your description, you might need:")
                    
                    total_cost = 0
                    for proc in suggestions:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{proc['description']}**")
                            _render_cpt_info(
                                "CPT",
                                proc['cpt'],
                                proc['description'],
                                key=f"symptom::{proc.get('cpt')}",
                            )
                        with col2:
                            st.metric("Cost", f"${proc['price']:,.2f}")
                        total_cost += proc['price']
                    
                    st.markdown("---")
                    st.metric("💰 ESTIMATED TOTAL COST", f"${total_cost:,.2f}")
                    
                    st.info("💡 **AI-Powered Matching:** This estimate is based on AI analysis of your symptoms matched against available procedures.")
                    
                    if debug:
                        with st.expander("Why these suggestions?"):
                            st.markdown(f"- Category: **{debug.get('category')}**")
                            st.markdown(f"- Red flags detected: **{debug.get('red_flags')}**")
                            st.markdown(f"- Candidate pool size: **{debug.get('candidate_count')}**")
                            if debug.get('keywords'):
                                st.markdown(f"- Top keywords: {', '.join(debug.get('keywords')[:8])}")
                            if debug.get('model_response'):
                                st.markdown("- Model rationale (truncated):")
                                st.code(debug.get('model_response'))
                else:
                    st.warning("Couldn't find matching procedures. Try the procedure search tab!")
            else:
                st.warning("Please describe your symptoms or injury")

    # TAB 2: Search by Procedure Name
    with tab2:
        st.header("Search by Procedure Name")
        st.markdown("*Example: chest xray, blood test, MRI, etc.*")


        search_term = st.text_input(
            "What procedure are you looking for?",
            placeholder="e.g., thoracic spine MRI",
            key="procedure_search",
        )

        q = (search_term or '').strip()
        if len(q) >= 2:
            # First attempt: literal substring match
            mask = pricing_data['description'].astype(str).str.contains(q, case=False, na=False)
            results_df = pricing_data[mask]

            if len(results_df) > 0:
                st.success(f"✅ Found {int(len(results_df))} matching procedure(s)")
                results = [
                    {
                        'cpt': row['cpt code number'],
                        'description': row['description'],
                        'price': row[price_col] if pd.notna(row[price_col]) else None,
                    }
                    for _, row in results_df.head(10).iterrows()
                ]
            else:
                with st.spinner("Searching…"):
                    results, enhanced_debug = search_procedures_ai(q, pricing_data, price_col, max_results=5)
                if results:
                    st.success("Closest matches:")
                else:
                    err = (enhanced_debug or {}).get('error') if isinstance(enhanced_debug, dict) else None
                    if err == 'missing_api_key':
                        st.error("Enhanced search requires an API key (OPENAI_API_KEY) to be configured.")
                    else:
                        st.warning(f"❌ No procedures found matching '{q}'. Try a shorter query like 'thoracic spine mri'.")
                    results = []

            for i, proc in enumerate(results):
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{proc['description']}**")
                        _render_cpt_info("CPT Code", proc['cpt'], proc['description'], key=f"proc_search::{q}::{i}")
                    with col2:
                        if proc.get('price') is not None:
                            st.metric("Price", f"${float(proc['price']):,.2f}")
                        else:
                            st.metric("Price", "N/A")
                    st.markdown("---")
            if len(results_df) > 10:
                st.info(f"💡 Showing first 10 of {int(len(results_df))} results. Try a more specific search term to narrow down.")
    
    # TAB 3: Direct CPT Lookup
    with tab3:
        st.header("Look Up Specific CPT Code")
        
        cpt_input = st.text_input(
            "Enter CPT Code",
            placeholder="e.g., 73600, 99213",
            key="cpt_lookup"
        )
        
        if cpt_input:
            result = pricing_data[pricing_data['cpt code number'].astype(str) == cpt_input.strip()]
            
            if len(result) > 0:
                st.success("✅ Code Found!")
                row = result.iloc[0]
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"### {row['description']}")
                    _render_cpt_info(
                        "CPT Code",
                        row['cpt code number'],
                        row['description'],
                        key=f"lookup::{row['cpt code number']}",
                    )
                with col2:
                    price = row[price_col]
                    if pd.notna(price):
                        st.metric("Price", f"${price:,.2f}")
                    else:
                        st.metric("Price", "N/A")
            else:
                st.error(f"❌ CPT code '{cpt_input}' not found")
    
    # TAB 4: Manual Estimate Builder
    with tab4:
        st.header("Build Your Own Estimate")
        st.markdown("Search and add multiple procedures to calculate total cost")
        
        # Initialize session state
        if 'cart' not in st.session_state:
            st.session_state.cart = []
        
        # Search to add items
        col1, col2 = st.columns([3, 1])
        with col1:
            add_search = st.text_input(
                "Search for procedure to add",
                placeholder="e.g., xray, blood test",
                key="cart_search"
            )
        
        if add_search:
            mask = pricing_data['description'].astype(str).str.contains(add_search, case=False, na=False)
            search_results = pricing_data[mask].head(5)
            
            if len(search_results) > 0:
                st.markdown("**Click to add:**")
                for idx, row in search_results.iterrows():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"{row['description']}")
                    with col2:
                        price = row[price_col]
                        if pd.notna(price):
                            st.write(f"${price:,.2f}")
                        else:
                            st.write("N/A")
                    with col3:
                        if st.button("➕ Add", key=f"add_{idx}"):
                            st.session_state.cart.append({
                                'cpt': row['cpt code number'],
                                'description': row['description'],
                                'price': row[price_col] if pd.notna(row[price_col]) else 0
                            })
                            st.rerun()
        
        # Display cart
        if st.session_state.cart:
            st.markdown("---")
            st.subheader("Your Estimate:")
            
            total = 0
            for i, item in enumerate(st.session_state.cart):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{item['description']}**")
                    _render_cpt_info(
                        "CPT",
                        item['cpt'],
                        item['description'],
                        key=f"cart::{i}",
                    )
                with col2:
                    st.write(f"${item['price']:,.2f}")
                with col3:
                    if st.button("🗑️", key=f"remove_{i}"):
                        st.session_state.cart.pop(i)
                        st.rerun()
                
                total += item['price']
            
            st.markdown("---")
            col1, col2 = st.columns([2, 1])
            with col2:
                st.metric("💰 TOTAL", f"${total:,.2f}")
            
            if st.button("🔄 Clear All"):
                st.session_state.cart = []
                st.rerun()
        else:
            st.info("👆 Search for procedures above and click 'Add' to build your estimate")

else:
    st.error("Could not load pricing data. Make sure the CSV file is in your project directory.")

# Footer
st.markdown("---")
st.markdown("*💡 Prices shown are discounted cash rates from NYU Langone Hospital. Always verify with the hospital directly.*")
