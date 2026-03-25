// Package tokenizer provides token counting functionality for various LLM providers.
// It supports accurate tokenization for OpenAI (via tiktoken), Anthropic (Claude),
// and Cohere models, with fallback estimation for unknown models.
//
// Usage:
//
//	// Get a tokenizer for a specific model
//	tok := tokenizer.ForModel("gpt-4o")
//	count := tok.Count("Hello, world!")
//
//	// Count tokens in chat messages
//	messages := []tokenizer.Message{
//	    {Role: "system", Content: "You are a helpful assistant."},
//	    {Role: "user", Content: "Hello!"},
//	}
//	total := tok.CountMessages(messages)
//
//	// Truncate text to fit within token limit
//	truncated := tok.Truncate("very long text...", 100)
package tokenizer

import (
	"log"
	"sync"

	"github.com/zenousai/goragtoolkit/message"
)

// Provider identifies the model provider
type Provider string

const (
	ProviderOpenAI    Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
	ProviderCohere    Provider = "cohere"
	ProviderUnknown   Provider = "unknown"
)

// Message is an alias for the shared message type.
type Message = message.Message

// Tokenizer provides token counting functionality for a specific model/encoding
type Tokenizer interface {
	// Count returns the token count for a single text string
	Count(text string) int

	// CountMessages returns total tokens for a list of chat messages.
	// This accounts for message overhead (role tokens, separators, etc.)
	// which varies by provider.
	CountMessages(messages []Message) int

	// Truncate returns text truncated to fit within maxTokens.
	// If the text already fits, it's returned unchanged.
	Truncate(text string, maxTokens int) string

	// TruncateMessages truncates message content to fit within a total token budget.
	// It preserves the first message (usually system prompt) and truncates from
	// the middle, keeping the most recent messages intact.
	TruncateMessages(messages []Message, maxTokens int) []Message

	// Encode returns the token IDs for a text string.
	// This is useful for advanced use cases like token manipulation.
	Encode(text string) []int

	// Decode converts token IDs back to text.
	Decode(tokens []int) string

	// ModelName returns the model name this tokenizer is configured for
	ModelName() string

	// Provider returns the provider this tokenizer is for
	Provider() Provider

	// EncodingName returns the encoding name (e.g., "cl100k_base", "o200k_base")
	EncodingName() string

	// IsAccurate returns true if this tokenizer provides accurate counts.
	// Returns false for estimation-based tokenizers.
	IsAccurate() bool
}

// maxCacheSize is the maximum number of tokenizers to cache.
// When exceeded, the cache is cleared to prevent unbounded memory growth.
// This is a simple strategy; accurate tokenizers (OpenAI, Claude, Cohere) are
// re-created cheaply since their data is embedded. Estimators are trivial.
const maxCacheSize = 256

// tokenizer cache to avoid recreating tokenizers
var (
	tokenizerCache = make(map[string]Tokenizer)
	cacheMu        sync.RWMutex
)

// ForModel returns a tokenizer for the given model name.
// It automatically detects the provider and selects the appropriate tokenizer.
// The tokenizer is cached for reuse (up to 256 entries; cache resets when full).
//
// Examples:
//   - "gpt-4o", "gpt-5", "o3" -> OpenAI tiktoken tokenizer
//   - "claude-sonnet-4-5" -> Anthropic Claude tokenizer
//   - "command-r-plus" -> Cohere tokenizer
//   - Unknown models -> Estimation-based tokenizer
func ForModel(modelName string) Tokenizer {
	cacheMu.RLock()
	if tok, ok := tokenizerCache[modelName]; ok {
		cacheMu.RUnlock()
		return tok
	}
	cacheMu.RUnlock()

	// Create new tokenizer
	tok := createTokenizer(modelName)

	// Cache it, evicting all entries if cache is full
	cacheMu.Lock()
	if len(tokenizerCache) >= maxCacheSize {
		tokenizerCache = make(map[string]Tokenizer)
	}
	tokenizerCache[modelName] = tok
	cacheMu.Unlock()

	return tok
}

// ForProvider returns a tokenizer for the given provider using the default encoding.
// This is useful when you know the provider but not the specific model.
func ForProvider(provider Provider) Tokenizer {
	switch provider {
	case ProviderOpenAI:
		return ForModel("gpt-4o") // Uses cl100k_base or o200k_base
	case ProviderAnthropic:
		return ForModel("claude-sonnet-4-5")
	case ProviderCohere:
		return ForModel("command-a-03-2025")
	default:
		return newEstimator("unknown", ProviderUnknown)
	}
}

// MustForModel returns a tokenizer for the given model, panicking on error.
// This is useful for initialization where errors should be fatal.
func MustForModel(modelName string) Tokenizer {
	tok := ForModel(modelName)
	if tok == nil {
		panic("failed to create tokenizer for model: " + modelName)
	}
	return tok
}

// ClearCache clears the tokenizer cache.
// This is mainly useful for testing.
func ClearCache() {
	cacheMu.Lock()
	tokenizerCache = make(map[string]Tokenizer)
	cacheMu.Unlock()
}

// createTokenizer creates the appropriate tokenizer for a model
func createTokenizer(modelName string) Tokenizer {
	provider := DetectProvider(modelName)

	switch provider {
	case ProviderOpenAI:
		tok, err := newTiktokenTokenizer(modelName)
		if err != nil {
			log.Printf("goragtoolkit/tokenizer: accurate tokenizer failed for %q (provider=%s), falling back to estimation: %v", modelName, provider, err)
			return newEstimator(modelName, provider)
		}
		return tok

	case ProviderAnthropic:
		tok, err := newClaudeTokenizer(modelName)
		if err != nil {
			log.Printf("goragtoolkit/tokenizer: accurate tokenizer failed for %q (provider=%s), falling back to estimation: %v", modelName, provider, err)
			return newEstimator(modelName, provider)
		}
		return tok

	case ProviderCohere:
		tok, err := newCohereTokenizer(modelName)
		if err != nil {
			log.Printf("goragtoolkit/tokenizer: accurate tokenizer failed for %q (provider=%s), falling back to estimation: %v", modelName, provider, err)
			return newEstimator(modelName, provider)
		}
		return tok

	default:
		return newEstimator(modelName, provider)
	}
}

// CountTokens is a convenience function that counts tokens for text using the specified model.
// This is equivalent to ForModel(modelName).Count(text).
func CountTokens(modelName, text string) int {
	return ForModel(modelName).Count(text)
}

// CountMessageTokens is a convenience function that counts tokens for messages using the specified model.
// This is equivalent to ForModel(modelName).CountMessages(messages).
func CountMessageTokens(modelName string, messages []Message) int {
	return ForModel(modelName).CountMessages(messages)
}

// EstimateTokens provides a quick estimation without loading a full tokenizer.
// This uses the simple heuristic of ~4 characters per token.
// Use this when you need a rough estimate and don't want tokenizer initialization overhead.
func EstimateTokens(text string) int {
	return (len(text) + 3) / 4
}
