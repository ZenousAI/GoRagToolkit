# GoRagToolkit

A Go toolkit for building RAG (Retrieval-Augmented Generation) applications. Provides model metadata, multi-provider tokenization, token-aware context assembly, and query shape inference — filling critical gaps in the Go AI/ML ecosystem where only Python alternatives exist.

## Requirements

- Go 1.26 or later

## Installation

Install the full toolkit:

```bash
go get github.com/zenousai/goragtoolkit@latest
```

Or install individual packages:

```bash
go get github.com/zenousai/goragtoolkit/catalog
go get github.com/zenousai/goragtoolkit/tokenizer
go get github.com/zenousai/goragtoolkit/ctxbudget
go get github.com/zenousai/goragtoolkit/answershape
```

## Packages

### `catalog` — AI Model Catalog

Centralized registry of AI model metadata across 10 providers. Provider detection from model name, context window sizes, max output tokens, reasoning model flags, embedding dimensions.

**Supported providers:** OpenAI, Anthropic, Cohere, Groq, AWS Bedrock, HuggingFace, LM Studio, Ollama, Jina, Mixedbread.

```go
package main

import (
    "fmt"
    "github.com/zenousai/goragtoolkit/catalog"
)

func main() {
    // Detect provider from model name
    provider := catalog.DetectProvider("gpt-4o")
    fmt.Println(provider) // "openai"

    provider = catalog.DetectProvider("claude-sonnet-4-5")
    fmt.Println(provider) // "anthropic"

    // Get context window size
    maxTokens := catalog.GetDefaultMaxTokens("gpt-4o")
    fmt.Println(maxTokens) // 128000

    maxTokens = catalog.GetDefaultMaxTokens("claude-sonnet-4-5")
    fmt.Println(maxTokens) // 200000

    // Get max output tokens
    maxOutput := catalog.GetDefaultMaxOutputTokens("gpt-4o")
    fmt.Println(maxOutput) // 16384

    // Check if a model uses reasoning APIs
    fmt.Println(catalog.IsReasoningModelByName("o3"))   // true
    fmt.Println(catalog.IsReasoningModelByName("gpt-4o")) // false

    // Look up a specific model
    model := catalog.GetModel(catalog.ProviderOpenAI, "text-embedding-3-small")
    if model != nil {
        fmt.Println(model.Name, model.Type, *model.Dimension) // text-embedding-3-small embedding 1536
    }

    // List all embedding models across all providers
    embeddings := catalog.GetModelsByType(catalog.ModelTypeEmbedding)
    fmt.Printf("Found %d embedding models\n", len(embeddings))

    // Browse a provider's full catalog
    cohere := catalog.GetProviderCatalog(catalog.ProviderCohere)
    for _, m := range cohere.Models {
        fmt.Printf("  %s (%s)\n", m.Name, m.Type)
    }
}
```

**Zero external dependencies.** No Go equivalent exists — the closest is LiteLLM's model list (Python-only).

---

### `tokenizer` — Multi-Provider Token Counting

Accurate tokenization for OpenAI (via tiktoken), Anthropic Claude (BPE), and Cohere (BPE) models. Falls back to estimation (~4 chars/token) for unknown models. Tokenizers are cached for reuse.

```go
package main

import (
    "fmt"
    "github.com/zenousai/goragtoolkit/tokenizer"
)

func main() {
    // Get a tokenizer for a specific model (cached after first call)
    tok := tokenizer.ForModel("gpt-4o")
    fmt.Println(tok.Provider())     // "openai"
    fmt.Println(tok.IsAccurate())   // true
    fmt.Println(tok.EncodingName()) // "o200k_base"

    // Count tokens
    count := tok.Count("Hello, world!")
    fmt.Println(count) // 4

    // Count tokens in chat messages (includes per-message overhead)
    messages := []tokenizer.Message{
        {Role: "system", Content: "You are a helpful assistant."},
        {Role: "user", Content: "Hello!"},
    }
    total := tok.CountMessages(messages)
    fmt.Println(total) // ~18

    // Truncate text to fit within a token limit
    longText := "This is a very long document that needs to be truncated..."
    truncated := tok.Truncate(longText, 5)
    fmt.Println(truncated)

    // Encode/decode for advanced use cases
    tokens := tok.Encode("Hello, world!")
    text := tok.Decode(tokens)
    fmt.Println(text) // "Hello, world!"

    // Works with any provider
    claude := tokenizer.ForModel("claude-sonnet-4-5") // Anthropic BPE tokenizer
    cohere := tokenizer.ForModel("command-a-03-2025") // Cohere BPE tokenizer
    unknown := tokenizer.ForModel("my-custom-model")  // Falls back to estimation
    fmt.Println(claude.Provider(), cohere.Provider(), unknown.Provider())

    // Get a tokenizer by provider (uses default model for that provider)
    tok = tokenizer.ForProvider(tokenizer.ProviderAnthropic)

    // Convenience functions (no need to create a tokenizer)
    count = tokenizer.CountTokens("gpt-4o", "Hello!")
    msgCount := tokenizer.CountMessageTokens("gpt-4o", messages)
    estimate := tokenizer.EstimateTokens("Quick rough count") // ~4 chars/token
    fmt.Println(count, msgCount, estimate)
}
```

Existing Go tokenizers (tiktoken-go, etc.) only support OpenAI. This is the only Go library covering OpenAI + Anthropic + Cohere with a unified interface.

---

### `ctxbudget` — RAG Context Budget Manager

Token-aware context assembly for LLM prompts. Allocates a model's context window across system prompt, retrieved sources, conversation history, and user query — with configurable strategies and graceful truncation.

```go
package main

import (
    "fmt"
    "log"
    "github.com/zenousai/goragtoolkit/ctxbudget"
)

func main() {
    // Quick start with sensible defaults
    mgr := ctxbudget.NewManagerWithDefaults("gpt-4o")

    result, err := mgr.Assemble(ctxbudget.AssembleInput{
        SystemPrompt: "You are a helpful assistant. Answer based on the provided sources.",
        Sources: []ctxbudget.Source{
            {ID: "doc1", Content: "Go is a statically typed language...", Title: "Go Overview", Score: 0.95},
            {ID: "doc2", Content: "Python is dynamically typed...", Title: "Python Overview", Score: 0.82},
            {ID: "doc3", Content: "Rust focuses on memory safety...", Title: "Rust Overview", Score: 0.71},
        },
        History: []ctxbudget.Message{
            {Role: "user", Content: "Tell me about programming languages"},
            {Role: "assistant", Content: "There are many programming languages..."},
        },
        UserQuery: "How does Go compare to Python?",
    })
    if err != nil {
        log.Fatal(err)
    }

    // result.Messages is ready to send to your LLM API
    for _, msg := range result.Messages {
        fmt.Printf("[%s] %s...\n", msg.Role, msg.Content[:min(80, len(msg.Content))])
    }

    // Inspect what happened
    fmt.Printf("Total tokens: %d\n", result.TotalTokens)
    fmt.Printf("Budget used: %.1f%%\n", result.BudgetUsed*100)
    fmt.Printf("Sources included: %d/%d\n", result.OutputSourceCount, result.InputSourceCount)
    fmt.Printf("History included: %d/%d\n", result.OutputHistoryCount, result.InputHistoryCount)
    for _, w := range result.Warnings {
        fmt.Println("Warning:", w)
    }

    // Custom configuration
    mgr = ctxbudget.NewManager("claude-sonnet-4-5", ctxbudget.NewTokenCounter("claude-sonnet-4-5"), ctxbudget.Config{
        MaxContextTokens:    100000,                              // Override model default
        ResponseReserve:     8000,                                // Reserve tokens for response
        Strategy:            ctxbudget.StrategyHistoryFirst,      // Prioritize conversation context
        SourceTruncation:    ctxbudget.SourceTruncateDiversity,   // Keep sources from different documents
        MinHistoryMessages:  4,                                   // Keep at least 4 recent messages
        SafetyMargin:        200,                                 // Buffer for tokenizer variance
    })

    // Shape-aware response reservation
    mgr.SetResponseReserveFromShape(ctxbudget.ShapeEnumerative, 10) // Reserve tokens for a 10-item list
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

**Allocation strategies:**

| Strategy | Constant | Use case |
|----------|----------|----------|
| Source-first | `StrategySourceFirst` | RAG — sources ARE the answer |
| History-first | `StrategyHistoryFirst` | Chat — conversation continuity matters |
| Balanced | `StrategyBalanced` | Agents — both sources and history matter equally |

**Source truncation modes:**

| Mode | Constant | Behavior |
|------|----------|----------|
| Drop lowest | `SourceTruncateDropLowest` | Remove entire lowest-scoring sources first |
| Proportional | `SourceTruncateProportional` | Shorten all sources proportionally by score |
| Diversity | `SourceTruncateDiversity` | Round-robin across documents to maximize coverage |

No equivalent exists in any language as a standalone library.

---

### `answershape` — Query Shape Inference

Classifies user queries into answer shapes using pattern matching, then generates machine-enforceable response contracts for LLM prompting. Optimizes both retrieval strategy and response structure.

```go
package main

import (
    "context"
    "fmt"
    "github.com/zenousai/goragtoolkit/answershape"
)

func main() {
    ctx := context.Background()
    inferrer := answershape.NewPatternInferrer(nil) // nil = default config

    // Infer the answer shape from a query
    result, _ := inferrer.Infer(ctx, "What are the main risk factors?")
    fmt.Println("Shape:", result.Shape)                           // enumerative
    fmt.Println("Coverage:", result.CoverageExpectation)          // high
    fmt.Println("Depth:", result.Depth)                           // medium
    fmt.Println("Retrieval:", result.SuggestedRetrievalStrategy)  // coverage_aware
    fmt.Println("Confidence:", result.Confidence)                 // 0.95
    fmt.Println("Min items:", result.ExpectedMinItems)            // 3

    // Different queries produce different shapes
    shapes := map[string]string{
        "How to deploy the app?":             "procedural",
        "Compare REST vs GraphQL":            "comparative",
        "What is a microservice?":            "factual",
        "Explain the system architecture":    "hierarchical",
        "What are all the requirements?":     "exhaustive",
        "Tell me about the product":          "exploratory",
    }
    for query, expected := range shapes {
        r, _ := inferrer.Infer(ctx, query)
        fmt.Printf("  %q → %s (expected %s)\n", query, r.Shape, expected)
    }

    // Generate a response contract from the shape
    contract := answershape.ContractFromShape(result)
    fmt.Println("Numbered list:", contract.Structure.RequireNumberedList)  // true
    fmt.Println("No merging:", contract.Coverage.NoMerging)               // true
    fmt.Println("Citations:", contract.Citation.CitationRequired)         // true
    fmt.Println("Token budget:", contract.Format.MaxTokens)               // 300

    // Convert the contract to LLM prompt instructions
    instructions := contract.ToPromptInstructions()
    fmt.Println(instructions)
    // Output:
    //   You are answering an ENUMERATIVE question.
    //   List each item from the reference units.
    //   ...
    //   Format your response as a numbered list.
    //   Cite your sources for each claim or item.

    // Build a custom contract with the fluent builder
    custom := answershape.NewContractBuilder().
        WithShape(answershape.ShapeComparative).
        RequireHeadings().
        RequireCitations().
        WithCitationFormat(answershape.CitationFormatBracket).
        WithInlineCitations().
        WithMaxTokens(500).
        WithLanguage("English").
        WithTone("formal").
        WithPresentationDepth(answershape.PresentationDepthDetailed).
        Build()

    if err := custom.Validate(); err != nil {
        fmt.Println("Invalid contract:", err)
    }

    // Serialize/deserialize contracts
    data, _ := custom.ToJSON()
    restored, _ := answershape.ContractFromJSON(data)
    fmt.Println(restored.Shape) // comparative

    // Get token budget for a shape + depth combination
    budget := answershape.TokenBudgetForShape(answershape.ShapeExhaustive, answershape.DepthDeep)
    fmt.Println("Budget:", budget) // 900

    // Infer with conversation context
    history := []answershape.Message{
        {Role: "user", Content: "Tell me about the security features"},
        {Role: "assistant", Content: "The system has several security features..."},
    }
    result, _ = inferrer.InferWithContext(ctx, "What are the specific types?", history)
    fmt.Println(result.Shape) // enumerative

    // Custom config
    cfg := answershape.DefaultConfig()
    cfg.EnableCaching = false
    cfg.DefaultShape = answershape.ShapeFactual
    customInferrer := answershape.NewPatternInferrer(cfg)
    _ = customInferrer
}
```

**Answer shapes:**

| Shape | Query examples | Retrieval strategy |
|-------|----------------|-------------------|
| `Factual` | "What is X?", "When was Y?" | Top-K |
| `Enumerative` | "What are the types of X?" | Coverage-aware |
| `Exhaustive` | "List all requirements" | Exhaustive |
| `Procedural` | "How to configure X?" | Top-K |
| `Comparative` | "Compare A vs B" | Coverage-aware |
| `Hierarchical` | "Explain the architecture" | Hierarchical |
| `Exploratory` | "Tell me about X" | Top-K |

No equivalent library exists — general intent classifiers solve a different problem (routing to skills), not shaping answer structure.

## End-to-End Example

Using all four packages together in a RAG pipeline:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/zenousai/goragtoolkit/answershape"
    "github.com/zenousai/goragtoolkit/catalog"
    "github.com/zenousai/goragtoolkit/ctxbudget"
    "github.com/zenousai/goragtoolkit/tokenizer"
)

func main() {
    ctx := context.Background()
    modelID := "gpt-4o"

    // 1. Verify model metadata
    provider := catalog.DetectProvider(modelID)
    contextWindow := catalog.GetDefaultMaxTokens(modelID)
    fmt.Printf("Model: %s (provider: %s, context: %d tokens)\n", modelID, provider, contextWindow)

    // 2. Infer the answer shape from the user's query
    query := "What are the main benefits of microservices?"
    inferrer := answershape.NewPatternInferrer(nil)
    shape, _ := inferrer.Infer(ctx, query)
    fmt.Printf("Query shape: %s (confidence: %.2f)\n", shape.Shape, shape.Confidence)

    // 3. Assemble context within token budget
    mgr := ctxbudget.NewManager(modelID, ctxbudget.NewTokenCounter(modelID), ctxbudget.Config{
        Strategy:         ctxbudget.StrategySourceFirst,
        SourceTruncation: ctxbudget.SourceTruncateProportional,
        SafetyMargin:     100,
    })
    mgr.SetResponseReserveFromShape(ctxbudget.ShapeEnumerative, shape.ExpectedMinItems)

    result, err := mgr.Assemble(ctxbudget.AssembleInput{
        SystemPrompt: "Answer based on the provided sources.\n" +
            answershape.ContractFromShape(shape).ToPromptInstructions(),
        Sources: []ctxbudget.Source{
            {ID: "s1", Content: "Microservices enable independent deployment...", Score: 0.9},
            {ID: "s2", Content: "Each service can be scaled independently...", Score: 0.85},
            {ID: "s3", Content: "Teams can work on services autonomously...", Score: 0.78},
        },
        UserQuery: query,
    })
    if err != nil {
        log.Fatal(err)
    }

    // 4. Count tokens before sending to LLM
    tok := tokenizer.ForModel(modelID)
    for _, msg := range result.Messages {
        tokens := tok.Count(msg.Content)
        fmt.Printf("[%s] %d tokens\n", msg.Role, tokens)
    }
    fmt.Printf("Total: %d tokens, %.1f%% of budget\n", result.TotalTokens, result.BudgetUsed*100)

    // result.Messages is now ready to send to your LLM API
}
```

## Dependencies

| Dependency | Used by | Purpose |
|-----------|---------|---------|
| `github.com/pkoukk/tiktoken-go` | `tokenizer` | OpenAI token encoding |
| `github.com/google/uuid` | `answershape` | Response contract IDs |
| `github.com/prometheus/client_golang` | `ctxbudget` | Observability metrics |

`catalog` and `tokenizer` have **zero** external dependencies beyond the above (and `catalog` has none at all).

## Contributing

Contributions are welcome. Please open an issue to discuss your idea before submitting a pull request.

```bash
# Clone and test
git clone https://github.com/zenousai/goragtoolkit.git
cd goragtoolkit
go test ./...
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
