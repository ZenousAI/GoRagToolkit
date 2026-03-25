// Package answershape provides answer shape inference for RAG retrieval optimization.
// It determines the expected structure of answers to guide coverage-aware retrieval.
package answershape

import (
	"context"

	"github.com/zenousai/goragtoolkit/message"
)

// Shape represents the expected structure of an answer
type Shape string

const (
	// ShapeEnumerative - Lists, risks, components, features (expects multiple distinct items)
	ShapeEnumerative Shape = "enumerative"
	// ShapeExhaustive - All rules, all requirements (expects complete coverage)
	ShapeExhaustive Shape = "exhaustive"
	// ShapeHierarchical - Architecture, system explanation (expects tree structure)
	ShapeHierarchical Shape = "hierarchical"
	// ShapeComparative - A vs B comparisons (expects balanced presentation)
	ShapeComparative Shape = "comparative"
	// ShapeProcedural - Step-by-step instructions (expects ordered sequence)
	ShapeProcedural Shape = "procedural"
	// ShapeExploratory - Broad explanation (expects flexible coverage)
	ShapeExploratory Shape = "exploratory"
	// ShapeFactual - Single fact or definition (expects focused answer)
	ShapeFactual Shape = "factual"
)

// CoverageExpectation indicates how complete the answer should be
type CoverageExpectation string

const (
	// CoverageLow - Answer can be partial/representative
	CoverageLow CoverageExpectation = "low"
	// CoverageMedium - Answer should cover main points
	CoverageMedium CoverageExpectation = "medium"
	// CoverageHigh - Answer must be comprehensive
	CoverageHigh CoverageExpectation = "high"
	// CoverageComplete - Answer must include all relevant items
	CoverageComplete CoverageExpectation = "complete"
)

// Depth indicates how detailed the answer should be
type Depth string

const (
	// DepthShallow - Brief, high-level answer
	DepthShallow Depth = "shallow"
	// DepthMedium - Standard detail level
	DepthMedium Depth = "medium"
	// DepthDeep - Comprehensive with full details
	DepthDeep Depth = "deep"
)

// InferenceResult contains the inferred answer shape and related metadata
type InferenceResult struct {
	// Shape is the primary expected answer structure
	Shape Shape `json:"shape"`

	// CoverageExpectation indicates how complete the answer should be
	CoverageExpectation CoverageExpectation `json:"coverage_expectation"`

	// Depth indicates how detailed the answer should be
	Depth Depth `json:"depth"`

	// MergeAllowed indicates if similar items can be combined
	MergeAllowed bool `json:"merge_allowed"`

	// ExpectedMinItems is the minimum number of distinct items expected (for enumerative/exhaustive)
	ExpectedMinItems int `json:"expected_min_items,omitempty"`

	// Confidence is the confidence score of this inference (0.0-1.0)
	Confidence float32 `json:"confidence"`

	// Signals contains the indicators that led to this inference
	Signals []InferenceSignal `json:"signals,omitempty"`

	// SuggestedRetrievalStrategy hints at the best retrieval approach
	SuggestedRetrievalStrategy RetrievalStrategy `json:"suggested_retrieval_strategy"`
}

// InferenceSignal represents a signal that contributed to shape inference
type InferenceSignal struct {
	// Type identifies the signal type
	Type SignalType `json:"type"`
	// Pattern is the matched pattern or indicator
	Pattern string `json:"pattern"`
	// Contribution is how much this signal influenced the result (0.0-1.0)
	Contribution float32 `json:"contribution"`
}

// SignalType identifies the type of inference signal
type SignalType string

const (
	// SignalKeywordMatch - Matched a keyword pattern
	SignalKeywordMatch SignalType = "keyword_match"
	// SignalQuestionWord - Matched a question word pattern
	SignalQuestionWord SignalType = "question_word"
	// SignalQuantifier - Found a quantifier (all, every, each, etc.)
	SignalQuantifier SignalType = "quantifier"
	// SignalListIndicator - Found list-indicating language
	SignalListIndicator SignalType = "list_indicator"
	// SignalComparison - Found comparison language
	SignalComparison SignalType = "comparison"
	// SignalSequence - Found sequence/step language
	SignalSequence SignalType = "sequence"
	// SignalStructure - Found structural reference (hierarchy, tree, etc.)
	SignalStructure SignalType = "structure"
	// SignalMLClassifier - Result from ML-based classification
	SignalMLClassifier SignalType = "ml_classifier"
)

// RetrievalStrategy suggests how to retrieve content for this shape
type RetrievalStrategy string

const (
	// RetrievalTopK - Standard top-K similarity search
	RetrievalTopK RetrievalStrategy = "topk"
	// RetrievalCoverageAware - Select to ensure coverage across groups
	RetrievalCoverageAware RetrievalStrategy = "coverage_aware"
	// RetrievalHierarchical - Select respecting parent-child relationships
	RetrievalHierarchical RetrievalStrategy = "hierarchical"
	// RetrievalExhaustive - Select all items within scope
	RetrievalExhaustive RetrievalStrategy = "exhaustive"
)

// Inferrer is the interface for answer shape inference
type Inferrer interface {
	// Infer analyzes a query and returns the expected answer shape
	Infer(ctx context.Context, query string) (*InferenceResult, error)

	// InferWithContext analyzes a query with conversation context
	InferWithContext(ctx context.Context, query string, history []Message) (*InferenceResult, error)
}

// Message is an alias for the shared message type.
type Message = message.Message

// Config contains configuration for the answer shape inferrer
type Config struct {
	// UseLLM enables LLM-based inference for ambiguous cases
	UseLLM bool `json:"use_llm"`

	// LLMConfidenceThreshold is the minimum confidence before falling back to LLM
	LLMConfidenceThreshold float32 `json:"llm_confidence_threshold"`

	// DefaultShape is the shape to use when confidence is too low
	DefaultShape Shape `json:"default_shape"`

	// EnableCaching enables caching of inference results
	EnableCaching bool `json:"enable_caching"`

	// CacheTTLSeconds is the TTL for cached results
	CacheTTLSeconds int `json:"cache_ttl_seconds"`
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	return &Config{
		UseLLM:                 false,
		LLMConfidenceThreshold: 0.6,
		DefaultShape:           ShapeExploratory,
		EnableCaching:          true,
		CacheTTLSeconds:        300,
	}
}

// ShapeToRetrievalStrategy maps shapes to their recommended retrieval strategy
var ShapeToRetrievalStrategy = map[Shape]RetrievalStrategy{
	ShapeEnumerative:  RetrievalCoverageAware,
	ShapeExhaustive:   RetrievalExhaustive,
	ShapeHierarchical: RetrievalHierarchical,
	ShapeComparative:  RetrievalCoverageAware,
	ShapeProcedural:   RetrievalTopK,
	ShapeExploratory:  RetrievalTopK,
	ShapeFactual:      RetrievalTopK,
}

// ShapeToCoverageExpectation maps shapes to their typical coverage expectation
var ShapeToCoverageExpectation = map[Shape]CoverageExpectation{
	ShapeEnumerative:  CoverageHigh,
	ShapeExhaustive:   CoverageComplete,
	ShapeHierarchical: CoverageMedium,
	ShapeComparative:  CoverageHigh,
	ShapeProcedural:   CoverageHigh,
	ShapeExploratory:  CoverageLow,
	ShapeFactual:      CoverageLow,
}
