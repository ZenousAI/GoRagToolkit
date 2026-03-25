package tokenizer

import (
	"embed"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
)

//go:embed data/claude_tokenizer.json
var claudeTokenizerData embed.FS

// claudeTokenizer implements tokenization for Anthropic Claude models
// using the BPE tokenizer vocabulary from HuggingFace
type claudeTokenizer struct {
	model         string
	vocab         map[string]int
	reverseVocab  map[int]string
	merges        []bpeMerge
	mergePriority map[string]int // cached merge priority map, built once at init
	initialized   bool
	initMu        sync.Mutex
}

// bpeMerge represents a BPE merge rule
type bpeMerge struct {
	pair   [2]string
	result string
}

// newClaudeTokenizer creates a new Claude tokenizer
func newClaudeTokenizer(modelName string) (*claudeTokenizer, error) {
	t := &claudeTokenizer{
		model: modelName,
	}

	// Lazy initialization - tokenizer data will be loaded on first use
	return t, nil
}

// ensureInitialized loads the tokenizer data if not already loaded
func (c *claudeTokenizer) ensureInitialized() error {
	if c.initialized {
		return nil
	}

	c.initMu.Lock()
	defer c.initMu.Unlock()

	if c.initialized {
		return nil
	}

	// Try to load embedded tokenizer data
	data, err := claudeTokenizerData.ReadFile("data/claude_tokenizer.json")
	if err != nil {
		// If embedded data not available, use estimation
		return fmt.Errorf("claude tokenizer data not embedded: %w", err)
	}

	if err := c.loadTokenizerJSON(data); err != nil {
		return err
	}

	c.initialized = true
	return nil
}

// tokenizerJSON represents the HuggingFace tokenizer.json format
type tokenizerJSON struct {
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges []string       `json:"merges"`
	} `json:"model"`
}

// loadTokenizerJSON loads tokenizer configuration from HuggingFace JSON format
func (c *claudeTokenizer) loadTokenizerJSON(data []byte) error {
	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return fmt.Errorf("failed to parse tokenizer JSON: %w", err)
	}

	c.vocab = tj.Model.Vocab
	c.reverseVocab = make(map[int]string, len(c.vocab))
	for token, id := range c.vocab {
		c.reverseVocab[id] = token
	}

	// Parse merge rules and build priority map once
	c.merges = make([]bpeMerge, 0, len(tj.Model.Merges))
	c.mergePriority = make(map[string]int, len(tj.Model.Merges))
	for _, merge := range tj.Model.Merges {
		parts := strings.SplitN(merge, " ", 2)
		if len(parts) == 2 {
			c.merges = append(c.merges, bpeMerge{
				pair:   [2]string{parts[0], parts[1]},
				result: parts[0] + parts[1],
			})
			key := parts[0] + "\x00" + parts[1]
			c.mergePriority[key] = len(c.merges) - 1
		}
	}

	return nil
}

// Count returns the token count for text
func (c *claudeTokenizer) Count(text string) int {
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
func (c *claudeTokenizer) tokenize(text string) []string {
	if len(c.vocab) == 0 {
		// Fallback: split by characters
		return strings.Split(text, "")
	}

	// Pre-tokenize: split into words/chunks
	// Claude uses a GPT-2 style pre-tokenizer
	words := preTokenize(text)

	var result []string
	for _, word := range words {
		tokens := c.bpeEncode(word)
		result = append(result, tokens...)
	}

	return result
}

// preTokenize splits text into pre-tokens (words/chunks)
// This follows GPT-2/Claude style: split on spaces but keep them attached
func preTokenize(text string) []string {
	var words []string
	var current strings.Builder

	for _, r := range text {
		if r == ' ' {
			if current.Len() > 0 {
				words = append(words, current.String())
				current.Reset()
			}
			current.WriteRune(r)
		} else {
			current.WriteRune(r)
		}
	}

	if current.Len() > 0 {
		words = append(words, current.String())
	}

	return words
}

// bpeEncode encodes a single word using BPE.
// Uses the pre-built mergePriority map for O(1) lookups per pair.
func (c *claudeTokenizer) bpeEncode(word string) []string {
	// Start with individual characters
	symbols := make([]string, 0, len(word))
	for _, r := range word {
		symbols = append(symbols, string(r))
	}

	if len(symbols) <= 1 {
		return symbols
	}

	// Iteratively apply merges using cached priority map
	for {
		// Find the best merge
		bestIdx := -1
		bestPriority := len(c.merges)

		for i := 0; i < len(symbols)-1; i++ {
			key := symbols[i] + "\x00" + symbols[i+1]
			if priority, ok := c.mergePriority[key]; ok && priority < bestPriority {
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

// estimateTokens provides a fallback estimation
func estimateTokens(text string) int {
	// Claude tokenization is roughly similar to GPT
	// Use ~4 characters per token as estimate
	return (len(text) + 3) / 4
}

// CountMessages returns total tokens for messages
func (c *claudeTokenizer) CountMessages(messages []Message) int {
	// Claude message format:
	// Human: <content>
	// Assistant: <content>
	// Overhead is roughly 3 tokens per message

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
func (c *claudeTokenizer) Truncate(text string, maxTokens int) string {
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

	// Join truncated tokens and ensure valid UTF-8 by re-encoding through runes
	joined := strings.Join(tokens[:maxTokens], "")
	// Validate UTF-8: converting to []rune and back drops invalid sequences
	return string([]rune(joined))
}

// TruncateMessages truncates messages to fit within maxTokens
func (c *claudeTokenizer) TruncateMessages(messages []Message, maxTokens int) []Message {
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

// Encode returns token strings (not IDs, since we use BPE strings)
func (c *claudeTokenizer) Encode(text string) []int {
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
func (c *claudeTokenizer) Decode(tokens []int) string {
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
func (c *claudeTokenizer) ModelName() string {
	return c.model
}

// Provider returns Anthropic
func (c *claudeTokenizer) Provider() Provider {
	return ProviderAnthropic
}

// EncodingName returns "claude"
func (c *claudeTokenizer) EncodingName() string {
	return "claude"
}

// IsAccurate returns true if the tokenizer data is loaded
func (c *claudeTokenizer) IsAccurate() bool {
	return c.initialized
}

// Ensure claudeTokenizer implements Tokenizer
var _ Tokenizer = (*claudeTokenizer)(nil)

// Helper to sort strings for deterministic output
func sortedKeys(m map[string]int) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
