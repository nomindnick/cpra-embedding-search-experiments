# CPRA Embedding Search Experiments - LinkedIn/Blog Post Draft

California Public Records Act (CPRA) requests are a burden for public agencies in large part because they have to sort through which documents are responsive from those that are not. I started experimenting to see if there is a way to make this easier.

My plan was to test a series of techniques - hybrid search, reranking, maybe LLM-based classification - but first I needed a baseline. How well do embeddings perform on their own, with just simple cosine similarity? The answer surprised me: even before any advanced techniques, a small open-source embedding model cut the number of non-responsive documents by half while still catching 95% of responsive documents.

There are two metrics that matter here. Recall measures what percentage of all responsive documents your search actually returns. This is critical in the CPRA context, as missing responsive documents could subject an agency to a lawsuit. Precision measures what percentage of returned documents are actually responsive. The lower the precision, the more extraneous documents staff have to weed out.

For testing I created a set of five different CPRA requests and 2,500 synthetic emails. Test emails were designed to challenge different scenarios, including near misses, ambiguous terms, indirect references, partial matches, and temporal mismatches.

A comprehensive keyword search achieved 94.1% recall but only 56.7% precision. That means keyword search found most responsive documents, but nearly half of everything it returned was non-responsive - a lot of unnecessary review work.

My goal is to find a solution that optimizes both recall and precision while running entirely on consumer-grade hardware - a CPU-only machine with 16GB of RAM, no GPU required. For this baseline test, I evaluated several embedding models with cosine similarity, tuning thresholds per model. The results:

- **embeddinggemma**: 95.5% recall, 63.7% precision
- **Jina Embeddings v3**: 95.2% recall, 68.9% precision
- **Snowflake Arctic Embed L v2.0**: 95.2% recall, 86.0% precision

That last result is the standout. Compared to keyword search, Snowflake Arctic maintains the same legal-compliance-level recall while improving precision from 57% to 86%. In practical terms, staff would review roughly half as many irrelevant documents.

This is a synthetic benchmark, so real-world performance will vary. But it suggests you don't need LLMs for everything - embeddings alone can do a significant amount of the heavy lifting. Keyword search isn't great, and there are probably additional gains to be made with reranking, hybrid approaches, and other techniques. But I was surprised how far embeddings get you before any of that, and thought it was worth sharing.
