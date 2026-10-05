# Improvement 2: Financial-Specific Data Engineering

## Overview
SEC filings are structure-heavy. Standard text chunking destroys the meaning of financial tables. We will move to a "Structure-Aware" ingestion pipeline.

## Implementation Strategy
1. **Advanced Table Parsing**:
   - Use **Unstructured.io** or **Marker** to convert SEC HTML tables into **Markdown** format.
   - Store the Markdown version in the vector DB, as LLMs can reason over `| Column |` structures much better than flat text.
2. **Recursive Section Splitting**:
   - Implement a `RecursiveCharacterTextSplitter` that prioritizes splits at `\n\n` (paragraphs) and then `\n` (lines), ensuring financial statements aren't sliced mid-row.
3. **Metadata Breadcrumbs**:
   - Append the hierarchy to every chunk: `[Company] -> [Year] -> [Form] -> [Section]`.
   - Example: `Apple Inc. | 2023 | 10-K | Consolidated Balance Sheets` prepended to the text.

## Why This Standouts
Most RAG systems treat PDFs/HTML as a "string of text." Treating them as "structured data" is the difference between a toy and a tool.

## Action Items
- [ ] Replace basic chunking in `src/chunk_documents.py` with a structure-aware splitter.
- [ ] Implement HTML-to-Markdown conversion for tables.
- [ ] Update `src/ingest.py` to include hierarchical metadata.
