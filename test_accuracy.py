#!/usr/bin/env python3
"""Comprehensive accuracy test for Invest UP RAG chatbot"""
import requests
import json
import time

BASE_URL = "http://localhost:3006/api/search"

# Test cases with expected keywords/concepts that should appear in answers
TEST_CASES = [
    # Test 1: Basic solar incentives query
    {
        "query": "What are the incentives for solar panel manufacturing in UP?",
        "category": "Solar/Renewable",
        "expected_concepts": ["subsidy", "incentive", "solar", "capital", "interest"],
        "description": "Basic solar incentives query"
    },

    # Test 2: Spelling error tolerance
    {
        "query": "what is nivesh mtra portal and how to registar",
        "category": "Spelling Correction",
        "expected_concepts": ["nivesh mitra", "single window", "registration", "portal", "online"],
        "description": "Query with spelling errors (mtra, registar)"
    },

    # Test 3: Hindi query
    {
        "query": "सोलर पैनल के लिए क्या प्रोत्साहन हैं UP में",
        "category": "Hindi Support",
        "expected_concepts": ["solar", "incentive", "subsidy", "UP"],
        "description": "Hindi query about solar incentives"
    },

    # Test 4: Complex multi-part query
    {
        "query": "I want to set up a textile factory in UP. What land subsidies are available and what is the registration process on nivesh mitra?",
        "category": "Complex Multi-part",
        "expected_concepts": ["textile", "land", "subsidy", "nivesh mitra", "registration"],
        "description": "Multi-part query (should trigger query rewriting)"
    },

    # Test 5: Specific policy query
    {
        "query": "What is the UP Industrial Investment and Employment Promotion Policy 2022?",
        "category": "Policy Specific",
        "expected_concepts": ["policy", "2022", "industrial", "investment", "incentive"],
        "description": "Specific policy details query"
    },

    # Test 6: Stamp duty exemption
    {
        "query": "stamp duty exemption for new industries in UP",
        "category": "Tax/Duty",
        "expected_concepts": ["stamp duty", "exemption", "100%", "reimbursement"],
        "description": "Stamp duty exemption query"
    },

    # Test 7: Food processing sector
    {
        "query": "incentives for food processing industry UP",
        "category": "Sector-specific",
        "expected_concepts": ["food processing", "subsidy", "capital", "interest"],
        "description": "Food processing sector query"
    },

    # Test 8: MSME specific
    {
        "query": "What benefits are available for MSMEs in Uttar Pradesh?",
        "category": "MSME",
        "expected_concepts": ["MSME", "micro", "small", "subsidy", "benefit"],
        "description": "MSME benefits query"
    },

    # Test 9: Land allocation
    {
        "query": "How to get land in industrial area of UP for factory?",
        "category": "Land/Infrastructure",
        "expected_concepts": ["land", "industrial", "UPSIDC", "allotment", "plot"],
        "description": "Land allocation query"
    },

    # Test 10: Employment incentives
    {
        "query": "What is the employment generation subsidy in UP?",
        "category": "Employment",
        "expected_concepts": ["employment", "subsidy", "job", "worker", "EPF"],
        "description": "Employment incentives query"
    },
]

def test_query(test_case, test_num):
    """Run a single test case and evaluate accuracy"""
    print(f"\n{'='*70}")
    print(f"TEST {test_num}: {test_case['description']}")
    print(f"Category: {test_case['category']}")
    print(f"Query: {test_case['query'][:70]}...")
    print('='*70)

    try:
        start_time = time.time()
        resp = requests.post(BASE_URL, json={"query": test_case["query"]}, timeout=120)
        latency = time.time() - start_time

        data = resp.json()

        if 'error' in data:
            print(f"ERROR: {data['error']}")
            return {"passed": False, "error": data['error'], "latency": latency}

        answer = data.get('answer', '').lower()
        num_sources = data.get('num_sources', 0)
        grounding_score = data.get('grounding_score', 'N/A')
        warning = data.get('warning', None)

        # Check expected concepts
        found_concepts = []
        missing_concepts = []
        for concept in test_case['expected_concepts']:
            if concept.lower() in answer:
                found_concepts.append(concept)
            else:
                missing_concepts.append(concept)

        accuracy = len(found_concepts) / len(test_case['expected_concepts']) * 100

        # Print results
        print(f"\nResults:")
        print(f"  Latency: {latency:.2f}s")
        print(f"  Sources: {num_sources} documents")
        print(f"  Grounding Score: {grounding_score}")
        if warning:
            print(f"  Warning: {warning}")

        print(f"\nConcept Coverage: {accuracy:.0f}%")
        print(f"  Found: {', '.join(found_concepts) if found_concepts else 'None'}")
        print(f"  Missing: {', '.join(missing_concepts) if missing_concepts else 'None'}")

        print(f"\nAnswer Preview:")
        print(f"  {data.get('answer', '')[:400]}...")

        passed = accuracy >= 60 and num_sources >= 3
        print(f"\nSTATUS: {'PASS' if passed else 'FAIL'}")

        return {
            "passed": passed,
            "accuracy": accuracy,
            "grounding_score": grounding_score,
            "num_sources": num_sources,
            "latency": latency,
            "found_concepts": found_concepts,
            "missing_concepts": missing_concepts
        }

    except Exception as e:
        print(f"REQUEST FAILED: {e}")
        return {"passed": False, "error": str(e), "latency": 0}

def main():
    print("\n" + "="*70)
    print("  INVEST UP CHATBOT - COMPREHENSIVE ACCURACY TEST")
    print("="*70)

    results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        result = test_query(test_case, i)
        result['category'] = test_case['category']
        results.append(result)
        time.sleep(1)  # Small delay between tests

    # Summary
    print("\n\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r.get('passed', False))
    total = len(results)

    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.0f}%)")

    avg_accuracy = sum(r.get('accuracy', 0) for r in results if 'accuracy' in r) / len([r for r in results if 'accuracy' in r])
    avg_latency = sum(r.get('latency', 0) for r in results) / len(results)
    avg_sources = sum(r.get('num_sources', 0) for r in results if 'num_sources' in r) / len([r for r in results if 'num_sources' in r])

    print(f"Average Concept Coverage: {avg_accuracy:.1f}%")
    print(f"Average Latency: {avg_latency:.2f}s")
    print(f"Average Sources Retrieved: {avg_sources:.1f}")

    # Grounding scores
    grounding_scores = [r.get('grounding_score') for r in results if r.get('grounding_score') and r.get('grounding_score') != 'N/A']
    if grounding_scores:
        avg_grounding = sum(grounding_scores) / len(grounding_scores)
        print(f"Average Grounding Score: {avg_grounding:.2f}")

    print("\nPer-Category Results:")
    categories = set(r['category'] for r in results)
    for cat in categories:
        cat_results = [r for r in results if r['category'] == cat]
        cat_passed = sum(1 for r in cat_results if r.get('passed', False))
        print(f"  {cat}: {cat_passed}/{len(cat_results)} passed")

    print("\n" + "="*70)
    print("  TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
