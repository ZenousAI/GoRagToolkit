package tokenizer

import (
	"strings"
)

// DetectProvider identifies the provider from a model name for tokenization purposes.
//
// NOTE: This is intentionally a separate, simplified implementation from packages/catalog.
// While catalog is the "single source of truth" for general provider/model information,
// the tokenizer maintains its own minimal provider detection for these reasons:
//
//  1. Different scope: Tokenizer only needs 4 categories (OpenAI, Anthropic, Cohere, Unknown)
//     while catalog supports many more (Bedrock, HuggingFace, Jina, Mixedbread, LMStudio, etc.)
//     that all map to "Unknown" (estimation) for tokenization anyway.
//
//  2. Avoid circular dependencies: Keeping tokenizer standalone prevents import cycles
//     if packages using catalog also need tokenization.
//
//  3. Minimize dependencies: Tokenizer is designed to be lightweight and portable.
//     Adding catalog dependency would pull in unnecessary code.
//
//  4. Different return types: Tokenizer uses a simple string internally, while catalog
//     uses ProviderType with many more variants.
func DetectProvider(modelName string) Provider {
	model := strings.ToLower(modelName)

	// OpenAI models
	if isOpenAIModel(model) {
		return ProviderOpenAI
	}

	// Anthropic/Claude models
	if isAnthropicModel(model) {
		return ProviderAnthropic
	}

	// Cohere models
	if isCohereModel(model) {
		return ProviderCohere
	}

	return ProviderUnknown
}

// isOpenAIModel checks if a model name is from OpenAI
func isOpenAIModel(model string) bool {
	openAIPrefixes := []string{
		// GPT series
		"gpt-3", "gpt-4", "gpt-5",
		// Reasoning models
		"o1", "o3", "o4",
		// Embedding models
		"text-embedding",
		// Legacy models
		"davinci", "curie", "babbage", "ada",
		// Codex
		"code-", "codex",
		// Chat models
		"chatgpt",
		// TTS/Whisper (for completeness)
		"tts-", "whisper",
		// Open-weight models
		"gpt-oss",
	}

	for _, prefix := range openAIPrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}

	return false
}

// isAnthropicModel checks if a model name is from Anthropic
func isAnthropicModel(model string) bool {
	return strings.HasPrefix(model, "claude")
}

// isCohereModel checks if a model name is from Cohere
func isCohereModel(model string) bool {
	coherePrefixes := []string{
		"command",  // command-r, command-a, command-nightly
		"c4ai-aya", // Aya multilingual models
		"embed-",   // Cohere embedding models (embed-english-v3, etc.)
		"rerank",   // Cohere rerank models
	}

	for _, prefix := range coherePrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}

	return false
}

// OpenAIEncoding represents OpenAI tokenizer encodings
type OpenAIEncoding string

const (
	// EncodingO200kBase is used by GPT-4o, GPT-4.1, GPT-4.5, GPT-5, o1, o3, o4 models
	EncodingO200kBase OpenAIEncoding = "o200k_base"

	// EncodingCL100kBase is used by GPT-4, GPT-3.5-turbo, text-embedding-3-*
	EncodingCL100kBase OpenAIEncoding = "cl100k_base"

	// EncodingP50kBase is used by older Codex models
	EncodingP50kBase OpenAIEncoding = "p50k_base"

	// EncodingR50kBase is used by GPT-3 models (davinci, etc.)
	EncodingR50kBase OpenAIEncoding = "r50k_base"
)

// GetOpenAIEncoding returns the tiktoken encoding for an OpenAI model
func GetOpenAIEncoding(modelName string) OpenAIEncoding {
	model := strings.ToLower(modelName)

	// GPT-4o and newer models use o200k_base
	// This includes: gpt-4o, gpt-4.1, gpt-4.5, gpt-5, o1, o3, o4 series
	if strings.HasPrefix(model, "gpt-4o") ||
		strings.HasPrefix(model, "gpt-4.1") ||
		strings.HasPrefix(model, "gpt-4.5") ||
		strings.HasPrefix(model, "gpt-5") ||
		strings.HasPrefix(model, "gpt-oss") ||
		strings.HasPrefix(model, "o1") ||
		strings.HasPrefix(model, "o3") ||
		strings.HasPrefix(model, "o4") ||
		strings.HasPrefix(model, "chatgpt") {
		return EncodingO200kBase
	}

	// GPT-4 (non-o variants), GPT-3.5-turbo, and embedding models use cl100k_base
	if strings.HasPrefix(model, "gpt-4") ||
		strings.HasPrefix(model, "gpt-3.5") ||
		strings.HasPrefix(model, "text-embedding") {
		return EncodingCL100kBase
	}

	// Legacy Codex models use p50k_base
	if strings.HasPrefix(model, "code-") || strings.HasPrefix(model, "codex") {
		return EncodingP50kBase
	}

	// Very old GPT-3 models use r50k_base
	if strings.HasPrefix(model, "davinci") ||
		strings.HasPrefix(model, "curie") ||
		strings.HasPrefix(model, "babbage") ||
		strings.HasPrefix(model, "ada") {
		return EncodingR50kBase
	}

	// Default to cl100k_base for unknown OpenAI models
	return EncodingCL100kBase
}

// ModelInfo contains metadata about a model relevant for tokenization
type ModelInfo struct {
	Name           string
	Provider       Provider
	Encoding       string // Encoding name (e.g., "cl100k_base", "claude", "cohere")
	MaxTokens      int    // Context window size
	TokensPerMsg   int    // Overhead tokens per message
	TokensPerRole  int    // Tokens used by role label
	TokensPerReply int    // Tokens for reply priming
}

// GetModelInfo returns tokenization metadata for a model
func GetModelInfo(modelName string) ModelInfo {
	provider := DetectProvider(modelName)

	switch provider {
	case ProviderOpenAI:
		encoding := GetOpenAIEncoding(modelName)
		return ModelInfo{
			Name:           modelName,
			Provider:       ProviderOpenAI,
			Encoding:       string(encoding),
			TokensPerMsg:   4, // <|im_start|>role\n ... <|im_end|>\n
			TokensPerRole:  1, // Role name typically 1 token
			TokensPerReply: 3, // <|im_start|>assistant<|im_sep|>
		}

	case ProviderAnthropic:
		return ModelInfo{
			Name:           modelName,
			Provider:       ProviderAnthropic,
			Encoding:       "claude",
			TokensPerMsg:   3, // Anthropic message format overhead
			TokensPerRole:  1,
			TokensPerReply: 3,
		}

	case ProviderCohere:
		return ModelInfo{
			Name:           modelName,
			Provider:       ProviderCohere,
			Encoding:       "cohere",
			TokensPerMsg:   3, // Cohere message format overhead
			TokensPerRole:  1,
			TokensPerReply: 3,
		}

	default:
		return ModelInfo{
			Name:           modelName,
			Provider:       ProviderUnknown,
			Encoding:       "estimate",
			TokensPerMsg:   4,
			TokensPerRole:  1,
			TokensPerReply: 3,
		}
	}
}
