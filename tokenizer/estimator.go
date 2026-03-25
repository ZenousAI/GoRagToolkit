package tokenizer

import (
	"strings"
	"unicode/utf8"
)

// estimatorTokenizer uses heuristics for token counting when no
// accurate tokenizer is available. This is a fallback for unknown
// models or when tokenizer initialization fails.
type estimatorTokenizer struct {
	model    string
	provider Provider
}

// newEstimator creates a new estimation-based tokenizer
func newEstimator(modelName string, provider Provider) *estimatorTokenizer {
	return &estimatorTokenizer{
		model:    modelName,
		provider: provider,
	}
}

// Count estimates token count using character and word-based heuristics
func (e *estimatorTokenizer) Count(text string) int {
	if text == "" {
		return 0
	}

	// Use a combination of character-based and word-based estimation
	// for better accuracy across different text types
	charCount := utf8.RuneCountInString(text)
	wordCount := len(strings.Fields(text))

	// Character-based estimate: ~4 chars per token for English
	charEstimate := (charCount + 3) / 4

	// Word-based estimate: ~1.3 tokens per word for English
	wordEstimate := int(float64(wordCount) * 1.3)

	// Use the larger estimate to be conservative (avoid underestimating)
	estimate := max(wordEstimate, charEstimate)

	// Adjust for different providers/languages
	switch e.provider {
	case ProviderAnthropic:
		// Claude tends to have slightly more tokens than GPT for the same text
		estimate = int(float64(estimate) * 1.05)
	case ProviderCohere:
		// Cohere's BPE tokenizer is similar to GPT
		// No adjustment needed
	}

	// Minimum 1 token for non-empty text
	if estimate < 1 {
		estimate = 1
	}

	return estimate
}

// CountMessages estimates tokens for a message list
func (e *estimatorTokenizer) CountMessages(messages []Message) int {
	info := GetModelInfo(e.model)
	total := 0

	for _, msg := range messages {
		// Message overhead
		total += info.TokensPerMsg
		// Role
		total += info.TokensPerRole
		// Content
		total += e.Count(msg.Content)
		// Optional name field
		if msg.Name != "" {
			total += e.Count(msg.Name)
		}
	}

	// Reply priming tokens
	total += info.TokensPerReply

	return total
}

// Truncate truncates text to fit within maxTokens
func (e *estimatorTokenizer) Truncate(text string, maxTokens int) string {
	if maxTokens <= 0 {
		return ""
	}

	currentTokens := e.Count(text)
	if currentTokens <= maxTokens {
		return text
	}

	// Estimate characters to keep
	// Since we estimate ~4 chars per token, multiply by 4
	// Use a slightly lower ratio to ensure we don't exceed
	targetChars := maxTokens * 3 // Conservative: 3 chars per token

	runes := []rune(text)
	if targetChars >= len(runes) {
		return text
	}

	return string(runes[:targetChars])
}

// TruncateMessages truncates messages to fit within maxTokens
func (e *estimatorTokenizer) TruncateMessages(messages []Message, maxTokens int) []Message {
	if len(messages) == 0 || maxTokens <= 0 {
		return messages
	}

	currentTokens := e.CountMessages(messages)
	if currentTokens <= maxTokens {
		return messages
	}

	// Strategy: Keep first message (system) and last N messages
	// Remove messages from the middle until we fit

	result := make([]Message, len(messages))
	copy(result, messages)

	// If only one message, truncate its content
	if len(result) == 1 {
		info := GetModelInfo(e.model)
		contentBudget := maxTokens - info.TokensPerMsg - info.TokensPerRole - info.TokensPerReply
		if contentBudget > 0 {
			result[0].Content = e.Truncate(result[0].Content, contentBudget)
		}
		return result
	}

	// Remove messages from position 1 (after system) until we fit
	for len(result) > 2 && e.CountMessages(result) > maxTokens {
		// Remove the second message (index 1)
		result = append(result[:1], result[2:]...)
	}

	// If still over budget and we have more than 1 message, truncate the last one
	if e.CountMessages(result) > maxTokens && len(result) > 1 {
		info := GetModelInfo(e.model)
		overhead := info.TokensPerMsg*len(result) + info.TokensPerRole*len(result) + info.TokensPerReply

		// Calculate content budget
		contentBudget := maxTokens - overhead
		if contentBudget > 0 {
			// Distribute budget: keep all of first message, truncate last
			firstTokens := e.Count(result[0].Content)
			lastBudget := contentBudget - firstTokens
			if lastBudget > 0 {
				result[len(result)-1].Content = e.Truncate(result[len(result)-1].Content, lastBudget)
			}
		}
	}

	return result
}

// Encode returns an approximation of token IDs (not accurate for estimation)
func (e *estimatorTokenizer) Encode(text string) []int {
	// For estimation, we can't provide actual token IDs
	// Return a slice of the estimated length with placeholder values
	count := e.Count(text)
	result := make([]int, count)
	for i := range result {
		result[i] = i // Placeholder values
	}
	return result
}

// Decode returns the original text (best effort for estimation)
func (e *estimatorTokenizer) Decode(tokens []int) string {
	// For estimation, we can't decode token IDs
	// This is a limitation of the estimation approach
	return ""
}

// ModelName returns the model name
func (e *estimatorTokenizer) ModelName() string {
	return e.model
}

// Provider returns the provider
func (e *estimatorTokenizer) Provider() Provider {
	return e.provider
}

// EncodingName returns "estimate" to indicate this is not accurate
func (e *estimatorTokenizer) EncodingName() string {
	return "estimate"
}

// IsAccurate returns false since this is estimation-based
func (e *estimatorTokenizer) IsAccurate() bool {
	return false
}
