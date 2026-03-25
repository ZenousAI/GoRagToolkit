package ctxbudget

import (
	"github.com/zenousai/goragtoolkit/tokenizer"
)

// tokenizerCounter wraps the tokenizer package to implement TokenCounter
type tokenizerCounter struct {
	tok tokenizer.Tokenizer
}

// NewTokenCounter creates a TokenCounter for the given model.
// It uses the tokenizer package which supports OpenAI, Anthropic, and Cohere models.
func NewTokenCounter(modelID string) TokenCounter {
	return &tokenizerCounter{
		tok: tokenizer.ForModel(modelID),
	}
}

// Count returns the token count for a text string
func (tc *tokenizerCounter) Count(text string) int {
	return tc.tok.Count(text)
}

// CountMessages returns total tokens for a list of messages.
// This accounts for message overhead (role tokens, separators, etc.)
func (tc *tokenizerCounter) CountMessages(msgs []Message) int {
	// Convert to tokenizer.Message format
	tokMsgs := make([]tokenizer.Message, len(msgs))
	for i, m := range msgs {
		tokMsgs[i] = tokenizer.Message{
			Role:    m.Role,
			Content: m.Content,
		}
	}
	return tc.tok.CountMessages(tokMsgs)
}

// estimatorCounter provides a simple estimation-based counter for testing
type estimatorCounter struct {
	charsPerToken float64
}

// NewEstimatorCounter creates a TokenCounter that uses simple estimation.
// This is useful for testing or when you don't need accurate counts.
// The default estimate is ~4 characters per token.
func NewEstimatorCounter() TokenCounter {
	return &estimatorCounter{charsPerToken: 4.0}
}

// Count returns an estimated token count for text
func (ec *estimatorCounter) Count(text string) int {
	return int(float64(len(text))/ec.charsPerToken + 0.5)
}

// CountMessages returns estimated tokens for messages.
// Adds overhead for role tokens (~4 per message).
func (ec *estimatorCounter) CountMessages(msgs []Message) int {
	total := 0
	for _, m := range msgs {
		// Content tokens + overhead for role/separators
		total += ec.Count(m.Content) + 4
	}
	return total
}

// mockCounter is used for testing with predictable token counts
type mockCounter struct {
	textCounts    map[string]int
	messageCounts map[string]int
	defaultCount  int
}

// NewMockCounter creates a TokenCounter for testing with configurable counts.
// Pass textCounts to specify exact counts for specific strings.
// Any string not in the map returns defaultCount.
func NewMockCounter(textCounts map[string]int, defaultCount int) TokenCounter {
	if textCounts == nil {
		textCounts = make(map[string]int)
	}
	return &mockCounter{
		textCounts:   textCounts,
		defaultCount: defaultCount,
	}
}

// Count returns the configured count for text, or default
func (mc *mockCounter) Count(text string) int {
	if count, ok := mc.textCounts[text]; ok {
		return count
	}
	return mc.defaultCount
}

// CountMessages returns sum of individual message counts + overhead
func (mc *mockCounter) CountMessages(msgs []Message) int {
	total := 0
	for _, m := range msgs {
		total += mc.Count(m.Content) + 4 // 4 tokens overhead per message
	}
	return total
}
