## Beyond the RAG Chatbot: Building an AI System That Refuses to Guess

**Why I built a financial-analysis pipeline that blocks its own hallucinations instead of just labeling them — and what a real audit log taught me about the difference.**

---

Every GenAI portfolio in 2026 has the same project in it: upload a PDF, ask it questions, watch an LLM answer with a citation stapled on at the end for good measure. It looks convincing. It demos well. And it would get you fired if you shipped it into a compliance workflow.

I wanted to build something different: a system for analyzing corporate financial filings where the standard I was designing against wasn't "does this sound right" but "would an auditor accept this." That distinction sounds subtle. In practice, it rewrites almost every architectural decision.

The result is **FinSight** — a multi-agent retrieval system that reads annual reports from Infosys, TCS, and Wipro and answers analyst-style questions ("compare operating margins FY2022–2024 and flag anomalies," "what drove Wipro's revenue in FY2024") with a hard guarantee: no claim reaches the user unless it can be traced to a verbatim sentence in a specific document, on a specific page. Claims that can't clear that bar aren't flagged with a caveat — they're deleted from the output before it's ever generated.

### The problem with "just add citations"

Most RAG systems treat citations as decoration. The LLM writes an answer, and — often in the same generation pass — appends a source reference. But nothing stops the model from citing a real, retrieved passage that doesn't actually say what the claim asserts. The citation is *present*; it just isn't *load-bearing*. You get the appearance of traceability without the substance.

FinSight treats verification as a separate, adversarial step performed by a different agent, with a different system prompt, whose entire job is to try to break the claim it's given. If it can't confirm the claim is entailed by its cited snippet, the claim doesn't survive — full stop.

### Six agents, one accountable pipeline

I orchestrated the system with Microsoft's Semantic Kernel — not as a convenience wrapper, but as the actual backbone. A query moves through six stages:

1. **Planner** decomposes a natural-language query into 2–6 concrete subtasks. "Compare Infosys and TCS margins" becomes separate subtasks per company per fiscal year, because that's what's actually retrievable.
2. **Retriever** runs a hybrid search per subtask — BM25 for exact figures like `20.7%`, dense embeddings for semantic recall — fused with Reciprocal Rank Fusion and reranked with a cross-encoder.
3. **Analyst** extracts candidate claims from the retrieved chunks, under a system prompt that explicitly forbids reporting anything the LLM had to *compute* — no ratios, no unit conversions, no cross-referencing two numbers into a third. If it's not printed verbatim in the filing, it's not a claim yet.
4. **Auditor** takes every claim from every subtask and, in one batched LLM call, checks whether the cited snippet actually entails it. Verified, uncertain, or blocked.
5. **Comparator** runs concurrently with the Auditor, placing pre-normalized figures side by side across documents and flagging deviations over 15% — again, doing zero arithmetic itself.
6. **Synthesizer** assembles everything that survived into a structured Markdown report.

Every run also writes a JSON audit log: the plan, every retrieval, every claim and its verdict, every agent invoked, total latency. Nothing about the run is a black box after the fact.

### The design choice that mattered most: no LLM arithmetic, anywhere

Early on I noticed something predictable but underappreciated: LLMs are bad at trustworthy arithmetic, especially unit conversion and multi-hop derivation, and they're bad at it in ways that *look* fine. Ask for a USD-to-INR-crore conversion buried in a sentence and you'll get a plausible number that's wrong by a factor you won't notice unless you already know the answer.

So I pulled every bit of arithmetic out of the LLM's hands. `core/unit_normalizer.py` is a deterministic, regex-and-`Decimal`-based module that converts every currency and scale expression — `$2.3 billion`, `₹18,500 crore`, `Rs. 4.2 lakh` — into one canonical form *before* the Comparator or Synthesizer ever sees it. The LLM's job downstream is reduced to comparing numbers that are already correct, not computing new ones. It's a small module, and it's probably the single highest-leverage piece of the whole system: it eliminates an entire category of hallucination by construction, rather than trying to catch it after the fact.

### A real catch, from a real audit log

The best evidence that a design works isn't the happy path — it's what happens when the system is pushed slightly off it. Here's a query I ran during evaluation: *"revenue per employee of all the companies."*

Revenue-per-employee isn't a number that appears in any filing. It's a ratio you'd have to compute — which is exactly the category of claim the Analyst agent is instructed to refuse to fabricate. In this run, the Auditor caught a claim along the lines of "revenue per employee for Infosys ≈ ₹453 lakh" and classified it **uncertain**, not verified — because the cited snippet confirmed the revenue figure but said nothing about headcount, so the ratio wasn't something the retrieved evidence actually supported.

Separately, the Planner had over-expanded the query to include companies not in the corpus at all (HCL, Cognizant). Those subtasks retrieved nothing, and — because there was no snippet to entail against — produced no claims rather than confidently hallucinated ones.

Neither of those behaviors was hand-coded for this specific query. They fell out of the architecture: claims without evidence don't get invented, and derived figures don't get silently verified just because they sound plausible. That's the payoff of building the constraint into the pipeline rather than the prompt.

### Confidence as a spectrum, not a coin flip

Binary answer/no-answer systems throw away useful signal. FinSight scores every claim on a five-signal composite — retrieval relevance, document-section type (audited financials outrank a shareholder letter), recency, cross-filing corroboration, and retrieval consistency — and buckets the result into three tiers: **verified**, **uncertain** (shown, but flagged), and **unverifiable** (blocked outright). An MD&A-sourced claim about *why* margins moved is treated with real skepticism next to a number lifted straight from an audited balance sheet, because they don't deserve the same trust.

### What this cost

None of this is free. The batched entailment check, the concurrent Auditor/Comparator stages, request hedging across a primary and reserve LLM endpoint — these exist because a compliance-grade pipeline has a much lower tolerance for tail latency and silent failure than a chatbot does. A query that touches five companies and asks for cross-document analysis can take 30–50 seconds end-to-end. That's the trade: slower and stricter, in exchange for output you could actually hand to someone who'd get in trouble for being wrong.

### Where this goes

The evaluation harness currently scores 4/4 on happy-path queries (verifiable facts, correctly surfaced) and 3/4 on adversarial ones (fabricated events, future-dated queries, cross-domain traps) — the fourth intentionally times out by design rather than guessing on a five-company query that shouldn't be answerable cheaply. That gap between "correct" and "confidently wrong" is the entire point of the project.

The full technical documentation — architecture diagrams, every module's responsibilities, the complete API and configuration reference — is in the repository. The code, the audit log samples, and the evaluation queries are all public:

**GitHub:** [github.com/debarshi29/FinSight](https://github.com/debarshi29/FinSight)

If you're building anything where an AI system's output has to survive contact with a human who can say "prove it" — this is one way to build it so that it can.
