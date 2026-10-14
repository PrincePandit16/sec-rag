import asyncio

from src.search import verify_citations

async def run_test(name, answer, results):
    print(f"\n--- Test: {name} ---")
    print(f"Answer: {answer}")
    is_verified, report = await verify_citations(None, answer, results)
    print(f"Verified: {is_verified}")
    print(f"Report: {report}")
    return is_verified, report

async def main():
    # Example 1: Perfect Match (Different Format)
    # Source has $10M, Answer has $10 million.
    res1 = [{"document": "The company reported revenue of $10M for the quarter.", "metadata": {}}]
    ans1 = "The company reported revenue of $10 million [1]."

    # Example 2: Citation Drift (The core problem)
    # Value is in Chunk 2, but cited as [1].
    res2 = [
        {"document": "Apple revenue was $100B.", "metadata": {}},
        {"document": "Microsoft revenue was $123.456B.", "metadata": {}},
    ]
    ans2 = "Microsoft revenue was $123.456B [1]."

    # Example 3: Out of Bounds
    res3 = [{"document": "Simple text", "metadata": {}}]
    ans3 = "Value is $50B [5]."

    # Example 4: Complete Hallucination
    res4 = [{"document": "Revenue was $10B.", "metadata": {}}]
    ans4 = "Revenue was $99B [1]."

    await run_test("Format Agnostic", ans1, res1)
    await run_test("Citation Drift", ans2, res2)
    await run_test("Out of Bounds", ans3, res3)
    await run_test("Hallucination", ans4, res4)

if __name__ == "__main__":
    asyncio.run(main())
