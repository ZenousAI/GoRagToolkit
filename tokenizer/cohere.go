package tokenizer

import (
	"embed"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

//go:embed data/cohere_tokenizer.json
var cohereTokenizerData embed.FS

// cohereTokenizer implements tokenization for Cohere models
// using the BPE tokenizer vocabulary from HuggingFace
type cohereTokenizer struct {
	model        string
	vocab        map[string]int
	reverseVocab map[int]string
	merges       []bpeMerge
	initialized  bool
	initMu       sync.Mutex
}

// newCohereTokenizer creates a new Cohere tokenizer
func newCohereTokenizer(modelName string) (*cohereTokenizer, error) {
	t := &cohereTokenizer{
		model: modelName,
	}

	// Lazy initialization - tokenizer data will be loaded on first use
	return t, nil
}

// ensureInitialized loads the tokenizer data if not already loaded
func (c *cohereTokenizer) ensureInitialized() error {
	if c.initialized {
		return nil
	}

	c.initMu.Lock()
	defer c.initMu.Unlock()

	if c.initialized {
		return nil
	}

	// Try to load embedded tokenizer data
	data, err := cohereTokenizerData.ReadFile("data/cohere_tokenizer.json")
	if err != nil {
		// If embedded data not available, use estimation
		return fmt.Errorf("cohere tokenizer data not embedded: %w", err)
	}

	if err := c.loadTokenizerJSON(data); err != nil {
		return err
	}

	c.initialized = true
	return nil
}

// loadTokenizerJSON loads tokenizer configuration from HuggingFace JSON format
func (c *cohereTokenizer) loadTokenizerJSON(data []byte) error {
	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return fmt.Errorf("failed to parse tokenizer JSON: %w", err)
	}

	c.vocab = tj.Model.Vocab
	c.reverseVocab = make(map[int]string, len(c.vocab))
	for token, id := range c.vocab {
		c.reverseVocab[id] = token
	}

	// Parse merge rules
	c.merges = make([]bpeMerge, 0, len(tj.Model.Merges))
	for _, merge := range tj.Model.Merges {
		parts := strings.SplitN(merge, " ", 2)
		if len(parts) == 2 {
			c.merges = append(c.merges, bpeMerge{
				pair:   [2]string{parts[0], parts[1]},
				result: parts[0] + parts[1],
			})
		}
	}

	return nil
}

// Count returns the token count for text
func (c *cohereTokenizer) Count(text string) int {
	if text == "" {
		return 0
	}

	// Try to use BPE tokenization
	if err := c.ensureInitialized(); err != nil {
		// Fall back to estimation if tokenizer not available
		return estimateTokens(text)
	}

	tokens := c.tokenize(text)
	return len(tokens)
}

// tokenize performs BPE tokenization
func (c *cohereTokenizer) tokenize(text string) []string {
	if len(c.vocab) == 0 {
		// Fallback: split by characters
		return strings.Split(text, "")
	}

	// Pre-tokenize: split into words/chunks
	words := preTokenize(text)

	var result []string
	for _, word := range words {
		tokens := c.bpeEncode(word)
		result = append(result, tokens...)
	}

	return result
}

// bpeEncode encodes a single word using BPE
func (c *cohereTokenizer) bpeEncode(word string) []string {
	// Start with individual characters
	symbols := make([]string, 0, len(word))
	for _, r := range word {
		symbols = append(symbols, string(r))
	}

	if len(symbols) <= 1 {
		return symbols
	}

	// Build merge priority map
	mergePriority := make(map[string]int, len(c.merges))
	for i, merge := range c.merges {
		key := merge.pair[0] + "\x00" + merge.pair[1]
		mergePriority[key] = i
	}

	// Iteratively apply merges
	for {
		// Find the best merge
		bestIdx := -1
		bestPriority := len(c.merges)

		for i := 0; i < len(symbols)-1; i++ {
			key := symbols[i] + "\x00" + symbols[i+1]
			if priority, ok := mergePriority[key]; ok && priority < bestPriority {
				bestPriority = priority
				bestIdx = i
			}
		}

		if bestIdx < 0 {
			break
		}

		// Apply the merge
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}

	return symbols
}

// CountMessages returns total tokens for messages
func (c *cohereTokenizer) CountMessages(messages []Message) int {
	// Cohere message format overhead
	info := GetModelInfo(c.model)
	total := 0

	for _, msg := range messages {
		total += info.TokensPerMsg
		total += info.TokensPerRole
		total += c.Count(msg.Content)
		if msg.Name != "" {
			total += c.Count(msg.Name)
		}
	}

	total += info.TokensPerReply
	return total
}

// Truncate returns text truncated to maxTokens
func (c *cohereTokenizer) Truncate(text string, maxTokens int) string {
	if maxTokens <= 0 {
		return ""
	}

	if err := c.ensureInitialized(); err != nil {
		// Fallback: character-based truncation (use runes for proper UTF-8 handling)
		targetChars := maxTokens * 3
		runes := []rune(text)
		if targetChars >= len(runes) {
			return text
		}
		return string(runes[:targetChars])
	}

	tokens := c.tokenize(text)
	if len(tokens) <= maxTokens {
		return text
	}

	return strings.Join(tokens[:maxTokens], "")
}

// TruncateMessages truncates messages to fit within maxTokens
func (c *cohereTokenizer) TruncateMessages(messages []Message, maxTokens int) []Message {
	if len(messages) == 0 || maxTokens <= 0 {
		return messages
	}

	currentTokens := c.CountMessages(messages)
	if currentTokens <= maxTokens {
		return messages
	}

	result := make([]Message, len(messages))
	copy(result, messages)

	// Remove messages from middle until we fit
	for len(result) > 2 && c.CountMessages(result) > maxTokens {
		result = append(result[:1], result[2:]...)
	}

	return result
}

// Encode returns token IDs for text
func (c *cohereTokenizer) Encode(text string) []int {
	if err := c.ensureInitialized(); err != nil {
		// Return estimated count as placeholder IDs
		count := estimateTokens(text)
		result := make([]int, count)
		for i := range result {
			result[i] = i
		}
		return result
	}

	tokens := c.tokenize(text)
	result := make([]int, len(tokens))
	for i, token := range tokens {
		if id, ok := c.vocab[token]; ok {
			result[i] = id
		} else {
			result[i] = 0 // Unknown token
		}
	}
	return result
}

// Decode converts token IDs back to text
func (c *cohereTokenizer) Decode(tokens []int) string {
	if err := c.ensureInitialized(); err != nil {
		return ""
	}

	var result strings.Builder
	for _, id := range tokens {
		if token, ok := c.reverseVocab[id]; ok {
			result.WriteString(token)
		}
	}
	return result.String()
}

// ModelName returns the model name
func (c *cohereTokenizer) ModelName() string {
	return c.model
}

// Provider returns Cohere
func (c *cohereTokenizer) Provider() Provider {
	return ProviderCohere
}

// EncodingName returns "cohere"
func (c *cohereTokenizer) EncodingName() string {
	return "cohere"
}

// IsAccurate returns true if the tokenizer data is loaded
func (c *cohereTokenizer) IsAccurate() bool {
	return c.initialized
}

// Ensure cohereTokenizer implements Tokenizer
var _ Tokenizer = (*cohereTokenizer)(nil)
