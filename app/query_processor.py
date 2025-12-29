"""
Query Processor for Invest UP Chatbot
- Query expansion with domain synonyms
- Hindi-English bilingual handling
- Spelling correction for common terms
"""
import re
from typing import List, Tuple
from difflib import SequenceMatcher

# Domain-specific synonym mappings for query expansion
SYNONYMS = {
    # Solar/Energy
    "solar": ["photovoltaic", "pv", "solar energy", "solar power", "saur urja"],
    "solar panel": ["pv panel", "solar module", "photovoltaic panel"],
    "renewable": ["green energy", "clean energy", "sustainable"],

    # Investment/Business
    "incentive": ["benefit", "subsidy", "concession", "rebate", "exemption", "discount"],
    "investment": ["nivesh", "capital", "funding"],
    "business": ["enterprise", "company", "firm", "udyog", "vyapar"],
    "industry": ["manufacturing", "udyog", "factory", "plant"],
    "startup": ["start-up", "new venture", "entrepreneur"],

    # Land/Property
    "land": ["plot", "site", "bhumi", "jamin"],
    "property": ["real estate", "sampatti"],

    # Government/Policy
    "policy": ["niti", "scheme", "yojana", "guideline"],
    "government order": ["shasanadesh", "go", "government notification", "sarkar adesh"],
    "shasanadesh": ["government order", "go", "notification"],
    "subsidy": ["anudan", "grant", "financial assistance", "sahayata"],

    # Procedures
    "registration": ["panjikaran", "enrollment", "apply"],
    "license": ["licence", "permit", "approval", "anumati"],
    "clearance": ["approval", "noc", "permission", "manzuri"],
    "application": ["aavedan", "form", "request"],

    # Sectors
    "textile": ["kapda", "garment", "fabric", "vastra"],
    "food processing": ["khadya prasanskaran", "agro processing"],
    "electronics": ["it", "hardware", "electronic"],
    "pharma": ["pharmaceutical", "medicine", "dawa", "aushadhi"],
    "automobile": ["auto", "vehicle", "vahan"],

    # Tax
    "tax": ["kar", "duty", "levy"],
    "gst": ["goods and services tax", "vat"],
    "stamp duty": ["stamp shulk", "registration fee"],

    # Infrastructure
    "infrastructure": ["infra", "facility", "suvidha"],
    "electricity": ["power", "bijli", "urja"],
    "water": ["jal", "pani"],

    # Portal names
    "nivesh mitra": ["invest mitra", "niveshmitra", "single window"],
}

# Hindi to English mappings for common terms
HINDI_TO_ENGLISH = {
    # Solar/Energy
    "सौर": "solar",
    "सौर ऊर्जा": "solar energy",
    "बिजली": "electricity",
    "ऊर्जा": "energy",

    # Investment/Business
    "निवेश": "investment",
    "प्रोत्साहन": "incentive",
    "लाभ": "benefit",
    "छूट": "exemption",
    "अनुदान": "subsidy",
    "व्यापार": "business",
    "उद्योग": "industry",
    "कारखाना": "factory",

    # Land
    "भूमि": "land",
    "जमीन": "land",
    "प्लॉट": "plot",

    # Government
    "सरकार": "government",
    "नीति": "policy",
    "योजना": "scheme",
    "शासनादेश": "government order",
    "आदेश": "order",

    # Procedures
    "पंजीकरण": "registration",
    "आवेदन": "application",
    "अनुमति": "permission",
    "लाइसेंस": "license",

    # Tax
    "कर": "tax",
    "शुल्क": "duty",

    # Questions
    "क्या": "what",
    "कैसे": "how",
    "कहाँ": "where",
    "कब": "when",
    "कौन": "who",
    "क्यों": "why",
    "कितना": "how much",
    "कितने": "how many",
}

# Common spelling mistakes and corrections
SPELLING_CORRECTIONS = {
    # Nivesh Mitra variations
    "nivesh mtra": "nivesh mitra",
    "niveshmitra": "nivesh mitra",
    "nives mitra": "nivesh mitra",
    "nivesh miter": "nivesh mitra",
    "nivesh mitr": "nivesh mitra",

    # Common typos
    "incentiv": "incentive",
    "incetive": "incentive",
    "incentives": "incentive",
    "subsidey": "subsidy",
    "subsidy": "subsidy",
    "goverment": "government",
    "governement": "government",
    "govenment": "government",
    "registeration": "registration",
    "registation": "registration",
    "licnese": "license",
    "licence": "license",
    "licens": "license",
    "aplicaiton": "application",
    "aplication": "application",
    "bussiness": "business",
    "busines": "business",
    "industy": "industry",
    "indusrty": "industry",
    "investmnt": "investment",
    "invetsment": "investment",
    "proceedure": "procedure",
    "procedur": "procedure",
    "elctricity": "electricity",
    "electricty": "electricity",
    "manufacuring": "manufacturing",
    "manufactring": "manufacturing",
    "infrastucture": "infrastructure",
    "infrastrucure": "infrastructure",
    "shasandesh": "shasanadesh",
    "sasanadesh": "shasanadesh",
    "shaasanadesh": "shasanadesh",

    # Sector typos
    "textil": "textile",
    "texile": "textile",
    "pharamceutical": "pharmaceutical",
    "pharmceutical": "pharmaceutical",
    "electronis": "electronics",
    "elecrtonics": "electronics",

    # UP specific
    "uttar pradesh": "uttar pradesh",
    "utar pradesh": "uttar pradesh",
    "utter pradesh": "uttar pradesh",
}

# Known correct terms for fuzzy matching
KNOWN_TERMS = list(SPELLING_CORRECTIONS.values()) + [
    "solar", "incentive", "subsidy", "investment", "business", "industry",
    "land", "policy", "government", "registration", "license", "application",
    "textile", "pharma", "electronics", "automobile", "food processing",
    "nivesh mitra", "shasanadesh", "uttar pradesh", "electricity", "infrastructure"
]


class QueryProcessor:
    """Process and enhance user queries for better retrieval"""

    def __init__(self):
        self.synonyms = SYNONYMS
        self.hindi_to_english = HINDI_TO_ENGLISH
        self.spelling_corrections = SPELLING_CORRECTIONS
        self.known_terms = list(set(KNOWN_TERMS))

    def process(self, query: str) -> dict:
        """
        Process query and return enhanced versions
        Returns dict with:
        - original: original query
        - corrected: spelling-corrected query
        - expanded: query with synonyms added
        - translated: Hindi terms translated to English
        - search_query: final query for embedding/search
        """
        original = query.strip()

        # Step 1: Correct spelling
        corrected = self.correct_spelling(original)

        # Step 2: Handle Hindi-English
        translated = self.translate_hindi(corrected)

        # Step 3: Expand with synonyms (for BM25, not embedding)
        expanded = self.expand_query(translated)

        # Final search query combines corrections and translations
        search_query = translated

        return {
            "original": original,
            "corrected": corrected,
            "translated": translated,
            "expanded": expanded,
            "search_query": search_query,
            "modifications": self._get_modifications(original, corrected, translated)
        }

    def correct_spelling(self, query: str) -> str:
        """Correct common spelling mistakes"""
        query_lower = query.lower()
        corrected = query

        # Direct replacements
        for wrong, right in self.spelling_corrections.items():
            if wrong in query_lower:
                # Preserve case of first letter
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                corrected = pattern.sub(right, corrected)
                query_lower = corrected.lower()

        # Fuzzy match for unknown words
        words = corrected.split()
        corrected_words = []
        for word in words:
            word_lower = word.lower()
            # Skip short words and known correct words
            if len(word_lower) < 4 or word_lower in self.known_terms:
                corrected_words.append(word)
                continue

            # Check fuzzy match
            best_match = self._fuzzy_match(word_lower)
            if best_match and best_match != word_lower:
                corrected_words.append(best_match)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def _fuzzy_match(self, word: str, threshold: float = 0.8) -> str:
        """Find fuzzy match for a word"""
        best_match = None
        best_score = 0

        for term in self.known_terms:
            # Only compare with similar length terms
            if abs(len(term) - len(word)) > 3:
                continue

            score = SequenceMatcher(None, word, term).ratio()
            if score > threshold and score > best_score:
                best_score = score
                best_match = term

        return best_match

    def translate_hindi(self, query: str) -> str:
        """Translate Hindi terms to English equivalents"""
        translated = query

        # Sort by length (longer phrases first) to avoid partial replacements
        sorted_hindi = sorted(self.hindi_to_english.keys(), key=len, reverse=True)

        for hindi, english in [(h, self.hindi_to_english[h]) for h in sorted_hindi]:
            if hindi in translated:
                # Add English alongside Hindi for bilingual search
                translated = translated.replace(hindi, f"{hindi} ({english})")

        return translated

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms for better BM25 matching"""
        query_lower = query.lower()
        expansions = []

        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                # Add up to 2 synonyms
                for syn in synonyms[:2]:
                    if syn.lower() not in query_lower:
                        expansions.append(syn)

        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query

    def _get_modifications(self, original: str, corrected: str, translated: str) -> List[str]:
        """Track what modifications were made"""
        mods = []
        if original.lower() != corrected.lower():
            mods.append("spelling_corrected")
        if corrected != translated:
            mods.append("hindi_translated")
        return mods

    def get_search_variants(self, query: str) -> List[str]:
        """Get multiple query variants for multi-query retrieval"""
        result = self.process(query)
        variants = [result["search_query"]]

        # Add expanded version if different
        if result["expanded"] != result["search_query"]:
            variants.append(result["expanded"])

        # Add original if it was modified
        if result["original"].lower() != result["search_query"].lower():
            variants.append(result["original"])

        return list(set(variants))[:3]  # Max 3 variants


# Singleton instance
_processor = None


def get_query_processor() -> QueryProcessor:
    """Get or create query processor singleton"""
    global _processor
    if _processor is None:
        _processor = QueryProcessor()
    return _processor


def process_query(query: str) -> dict:
    """Convenience function to process a query"""
    return get_query_processor().process(query)
