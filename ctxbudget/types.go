// Package ctxbudget provides token-aware context budget management for LLM interactions.
//
// The package implements intelligent allocation of token budgets across different
// components of an LLM context (system prompt, sources, conversation history, user query)
// with configurable strategies optimized for different use cases like RAG and agents.
//
// Key features:
//   - Source-first allocation for RAG (sources ARE the answer)
//   - Shape-aware response reservation based on answer type
//   - Graceful truncation with recency preservation
//   - Observability via Prometheus metrics
//
// Usage:
//
//	mgr := ctxbudget.NewManager("gpt-4o", ctxbudget.DefaultConfig())
//	result, err := mgr.Assemble(ctxbudget.AssembleInput{
//	    SystemPrompt: prompt,
//	    Sources:      sources,
//	    History:      history,
//	    UserQuery:    query,
//	})
package ctxbudget

// AllocationStrategy controls how budget is distributed among components
type AllocationStrategy string

const (
	// StrategySourceFirst prioritizes sources over history (RAG default).
	// Order: SystemPrompt -> UserQuery -> Sources -> History
	// This ensures retrieved content gets priority since sources ARE the answer in RAG.
	StrategySourceFirst AllocationStrategy = "source_first"

	// StrategyHistoryFirst prioritizes history over sources (continuity chat).
	// Order: SystemPrompt -> UserQuery -> History -> Sources
	// Use for conversation-heavy flows where context continuity matters more.
	StrategyHistoryFirst AllocationStrategy = "history_first"

	// StrategyBalanced splits remaining budget evenly (agent mode).
	// Order: SystemPrompt -> UserQuery -> 50% Sources / 50% History
	// Useful when both history and sources are equally important.
	StrategyBalanced AllocationStrategy = "balanced"
)

// SourceTruncationMode controls how sources are reduced when over budget
type SourceTruncationMode string

const (
	// SourceTruncateDropLowest drops entire lowest-scoring sources first.
	// Simple and predictable - sources are either fully included or excluded.
	SourceTruncateDropLowest SourceTruncationMode = "drop_lowest"

	// SourceTruncateProportional truncates all sources proportionally by score.
	// Higher-scored sources keep more content. Preserves partial context from more sources.
	SourceTruncateProportional SourceTruncationMode = "proportional"

	// SourceTruncateDiversity keeps sources from different documents even if lower scoring.
	// Maintains answer diversity at the cost of potentially dropping high-scoring duplicates.
	SourceTruncateDiversity SourceTruncationMode = "diversity"
)

// Message represents a chat message for context management
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Source represents a retrieved source/SU for context
type Source struct {
	// ID is the unique identifier for this source (e.g., SU ID)
	ID string `json:"id"`

	// Content is the source text content
	Content string `json:"content"`

	// Title is the source title (optional)
	Title string `json:"title,omitempty"`

	// Score is the relevance score for prioritization (higher = more relevant)
	Score float64 `json:"score"`

	// DocumentID groups sources from the same document (for diversity truncation)
	DocumentID string `json:"document_id,omitempty"`

	// Weight is an engagement-based multiplier for prioritization (default 1.0).
	// Sources the user has engaged with (cited in conversation history) receive
	// higher weights so they are preserved during context budget truncation.
	Weight float64 `json:"weight,omitempty"`
}

// EffectiveScore returns Score * Weight for prioritization.
// If Weight is zero or negative, it defaults to 1.0.
func (s Source) EffectiveScore() float64 {
	w := s.Weight
	if w <= 0 {
		w = 1.0
	}
	return s.Score * w
}

// Config controls context assembly behavior
type Config struct {
	// MaxContextTokens is the total token budget for the context.
	// 0 = use model default from catalog (recommended).
	MaxContextTokens int

	// ResponseReserve is tokens reserved for model response.
	// 0 = use shape-aware default based on answer type.
	ResponseReserve int

	// MaxTokensPerMessage limits individual message size.
	// 0 = unlimited (not recommended for production).
	MaxTokensPerMessage int

	// Strategy controls allocation priority among components.
	// Default: StrategySourceFirst for RAG use cases.
	Strategy AllocationStrategy

	// SourceTruncation controls how sources are reduced when over budget.
	// Default: SourceTruncateProportional.
	SourceTruncation SourceTruncationMode

	// MinHistoryMessages is the minimum history messages to preserve.
	// Even in source_first mode, this many recent messages are kept.
	// Default: 2 (keeps at least the last exchange for continuity).
	MinHistoryMessages int

	// SafetyMargin is extra tokens to reserve as buffer.
	// Accounts for tokenizer variance and message overhead.
	// Default: 100.
	SafetyMargin int
}

// DefaultConfig returns sensible defaults for RAG use cases
func DefaultConfig() Config {
	return Config{
		MaxContextTokens:    0, // Use model default
		ResponseReserve:     0, // Shape-aware
		MaxTokensPerMessage: 4000,
		Strategy:            StrategySourceFirst,
		SourceTruncation:    SourceTruncateProportional,
		MinHistoryMessages:  2,
		SafetyMargin:        100,
	}
}

// AssembleInput contains all inputs for context assembly
type AssembleInput struct {
	// SystemPrompt is the system/instruction prompt (never truncated)
	SystemPrompt string

	// Sources are the retrieved sources/SUs to include
	Sources []Source

	// History is the conversation history (may be truncated)
	History []Message

	// UserQuery is the current user query (never truncated)
	UserQuery string
}

// AssembleResult is the output of context assembly
type AssembleResult struct {
	// Messages is the final message list ready for LLM
	Messages []Message

	// TotalTokens is the token count of assembled context
	TotalTokens int

	// BudgetUsed is the utilization ratio (0.0-1.0)
	BudgetUsed float64

	// ResponseBudget is tokens remaining for model response
	ResponseBudget int

	// Truncations records what was truncated
	Truncations []TruncationEvent

	// Warnings are human-readable messages about context assembly
	Warnings []string

	// === Input/Output Tracking for Metrics ===

	// InputSourceCount is the number of sources provided as input
	InputSourceCount int

	// InputSourceTokens is the total tokens in input sources
	InputSourceTokens int

	// OutputSourceCount is the number of sources included in output
	OutputSourceCount int

	// OutputSourceTokens is the tokens used by sources in output
	OutputSourceTokens int

	// InputHistoryCount is the number of history messages provided
	InputHistoryCount int

	// InputHistoryTokens is the total tokens in input history
	InputHistoryTokens int

	// OutputHistoryCount is the number of history messages included
	OutputHistoryCount int

	// OutputHistoryTokens is the tokens used by history in output
	OutputHistoryTokens int
}

// TruncationEvent records a truncation action
type TruncationEvent struct {
	// Component identifies what was truncated: "source", "history", "message"
	Component string `json:"component"`

	// ItemID is the identifier of the truncated item (if applicable)
	ItemID string `json:"item_id,omitempty"`

	// TokensBefore is the token count before truncation
	TokensBefore int `json:"tokens_before"`

	// TokensAfter is the token count after truncation
	TokensAfter int `json:"tokens_after"`

	// Reason explains why truncation occurred
	Reason string `json:"reason"`
}

// TokenCounter abstracts token counting for testability
type TokenCounter interface {
	// Count returns the token count for a text string
	Count(text string) int

	// CountMessages returns total tokens for a list of messages
	CountMessages(msgs []Message) int
}

// ShapeType identifies the answer shape for response reservation
type ShapeType string

const (
	ShapeFactual      ShapeType = "factual"
	ShapeEnumerative  ShapeType = "enumerative"
	ShapeExhaustive   ShapeType = "exhaustive"
	ShapeExploratory  ShapeType = "exploratory"
	ShapeProcedural   ShapeType = "procedural"
	ShapeComparative  ShapeType = "comparative"
	ShapeHierarchical ShapeType = "hierarchical"
)

// ResponseReservation holds shape-aware response budget information
type ResponseReservation struct {
	// Shape is the inferred answer shape
	Shape ShapeType

	// ItemCount is the expected number of items (for list-type shapes)
	ItemCount int

	// ReservedTokens is the calculated response reservation
	ReservedTokens int
}

// DefaultContextWindow is used when model is not found in any catalog
const DefaultContextWindow = 32000

// DefaultResponseReserve is used when shape is unknown
const DefaultResponseReserve = 4096
