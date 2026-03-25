package tokenizer

import (
	"github.com/pkoukk/tiktoken-go"
)

// tiktokenTokenizer uses the tiktoken library for accurate OpenAI token counting
type tiktokenTokenizer struct {
	encoding *tiktoken.Tiktoken
	model    string
	encName  string
}

// newTiktokenTokenizer creates a new tiktoken-based tokenizer for OpenAI models
func newTiktokenTokenizer(modelName string) (*tiktokenTokenizer, error) {
	encName := string(GetOpenAIEncoding(modelName))

	// Try to get encoding by name
	encoding, err := tiktoken.GetEncoding(encName)
	if err != nil {
		// Try model-based encoding as fallback
		encoding, err = tiktoken.EncodingForModel(modelName)
		if err != nil {
			return nil, err
		}
	}

	return &tiktokenTokenizer{
		encoding: encoding,
		model:    modelName,
		encName:  encName,
	}, nil
}

// Count returns the exact token count for text using tiktoken
func (t *tiktokenTokenizer) Count(text string) int {
	if text == "" {
		return 0
	}
	tokens := t.encoding.Encode(text, nil, nil)
	return len(tokens)
}

// CountMessages returns total tokens for a message list with OpenAI format overhead
func (t *tiktokenTokenizer) CountMessages(messages []Message) int {
	// OpenAI message format tokens:
	// Each message: <|im_start|>role\ncontent<|im_end|>\n
	// This adds ~4 tokens per message for the special tokens
	//
	// For tool/function messages, there may be additional tokens for the name field

	info := GetModelInfo(t.model)
	total := 0

	for _, msg := range messages {
		// Base message overhead
		total += info.TokensPerMsg

		// Role tokens
		total += t.Count(msg.Role)

		// Content tokens
		total += t.Count(msg.Content)

		// Name field (for function calls)
		if msg.Name != "" {
			total += t.Count(msg.Name)
			total += 1 // separator
		}

		// Tool call ID
		if msg.ToolCallID != "" {
			total += t.Count(msg.ToolCallID)
		}
	}

	// Reply priming: <|im_start|>assistant<|im_sep|>
	total += info.TokensPerReply

	return total
}

// Truncate returns text truncated to fit within maxTokens
func (t *tiktokenTokenizer) Truncate(text string, maxTokens int) string {
	if maxTokens <= 0 {
		return ""
	}

	tokens := t.encoding.Encode(text, nil, nil)
	if len(tokens) <= maxTokens {
		return text
	}

	// Truncate to maxTokens and decode back
	truncatedTokens := tokens[:maxTokens]
	return t.encoding.Decode(truncatedTokens)
}

// TruncateMessages truncates messages to fit within maxTokens
func (t *tiktokenTokenizer) TruncateMessages(messages []Message, maxTokens int) []Message {
	if len(messages) == 0 || maxTokens <= 0 {
		return messages
	}

	currentTokens := t.CountMessages(messages)
	if currentTokens <= maxTokens {
		return messages
	}

	// Strategy: Keep first message (system) and last N messages
	// Remove messages from the middle until we fit
	result := make([]Message, len(messages))
	copy(result, messages)

	// If only one message, truncate its content
	if len(result) == 1 {
		info := GetModelInfo(t.model)
		contentBudget := maxTokens - info.TokensPerMsg - t.Count(result[0].Role) - info.TokensPerReply
		if contentBudget > 0 {
			result[0].Content = t.Truncate(result[0].Content, contentBudget)
		}
		return result
	}

	// Remove messages from position 1 (after system) until we fit
	for len(result) > 2 && t.CountMessages(result) > maxTokens {
		// Remove the second message (index 1)
		result = append(result[:1], result[2:]...)
	}

	// If still over budget and we have more than 1 message, truncate the last one
	if t.CountMessages(result) > maxTokens && len(result) > 1 {
		// Calculate how much we need to trim from the last message
		currentTokens := t.CountMessages(result)
		excess := currentTokens - maxTokens

		lastMsg := &result[len(result)-1]
		lastContentTokens := t.Count(lastMsg.Content)
		newContentTokens := lastContentTokens - excess

		if newContentTokens > 0 {
			lastMsg.Content = t.Truncate(lastMsg.Content, newContentTokens)
		} else {
			// Content would be empty, remove this message too
			result = result[:len(result)-1]
		}
	}

	return result
}

// Encode returns the token IDs for text
func (t *tiktokenTokenizer) Encode(text string) []int {
	return t.encoding.Encode(text, nil, nil)
}

// Decode converts token IDs back to text
func (t *tiktokenTokenizer) Decode(tokens []int) string {
	return t.encoding.Decode(tokens)
}

// ModelName returns the model name
func (t *tiktokenTokenizer) ModelName() string {
	return t.model
}

// Provider returns OpenAI
func (t *tiktokenTokenizer) Provider() Provider {
	return ProviderOpenAI
}

// EncodingName returns the tiktoken encoding name
func (t *tiktokenTokenizer) EncodingName() string {
	return t.encName
}

// IsAccurate returns true since tiktoken is accurate
func (t *tiktokenTokenizer) IsAccurate() bool {
	return true
}
