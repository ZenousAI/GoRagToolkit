package catalog

import "strings"

// DetectProvider determines the provider from a model name.
// This is the single source of truth for provider detection across all packages.
func DetectProvider(modelName string) ProviderType {
	if modelName == "" {
		return ProviderUnknown
	}

	model := strings.ToLower(modelName)

	// OpenAI models
	if isOpenAIModel(model) {
		return ProviderOpenAI
	}

	// Anthropic models
	if isAnthropicModel(model) {
		return ProviderAnthropic
	}

	// Cohere models
	if isCohereModel(model) {
		return ProviderCohere
	}

	// Jina models
	if isJinaModel(model) {
		return ProviderJina
	}

	// Mixedbread models
	if isMixedbreadModel(model) {
		return ProviderMixedbread
	}

	// Groq models (explicit groq/ prefix only; Groq hosts other providers' models)
	if isGroqModel(model) {
		return ProviderGroq
	}

	// Bedrock models (AWS-hosted models with specific prefixes)
	if isBedrockModel(model) {
		return ProviderBedrock
	}

	// HuggingFace models (usually contain slashes or specific prefixes)
	if isHuggingFaceModel(model) {
		return ProviderHuggingFace
	}

	return ProviderUnknown
}

// isOpenAIModel checks if the model name indicates an OpenAI model
func isOpenAIModel(model string) bool {
	openAIPrefixes := []string{
		// GPT series
		"gpt-3", "gpt-4", "gpt-5",
		// o-series reasoning models
		"o1", "o3", "o4",
		// Embedding models
		"text-embedding",
		// Legacy models
		"davinci", "curie", "babbage", "ada",
		// Codex
		"code-", "codex",
		// ChatGPT
		"chatgpt",
		// Audio models
		"tts-", "whisper",
		// Open source models from OpenAI
		"gpt-oss",
	}

	for _, prefix := range openAIPrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// isAnthropicModel checks if the model name indicates an Anthropic model
func isAnthropicModel(model string) bool {
	return strings.HasPrefix(model, "claude")
}

// isCohereModel checks if the model name indicates a Cohere model
func isCohereModel(model string) bool {
	coherePrefixes := []string{
		"command",  // command-r, command-a, command-nightly
		"c4ai-aya", // Aya multilingual models
		"embed-",   // Cohere embedding models (embed-english-v3, embed-multilingual-v3, etc.)
		"rerank",   // Cohere rerank models
	}

	for _, prefix := range coherePrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// isJinaModel checks if the model name indicates a Jina AI model
func isJinaModel(model string) bool {
	return strings.HasPrefix(model, "jina-") ||
		strings.Contains(model, "jina/") ||
		strings.Contains(model, "jinaai/")
}

// isMixedbreadModel checks if the model name indicates a Mixedbread model
func isMixedbreadModel(model string) bool {
	return strings.HasPrefix(model, "mxbai-") ||
		strings.Contains(model, "mixedbread")
}

// isGroqModel checks if the model name indicates a Groq model
func isGroqModel(model string) bool {
	// Check for Groq-specific model prefixes
	groqPrefixes := []string{
		"groq/",           // Explicit Groq prefix and Compound systems
		"llama-3.3-",      // Llama 3.3 70B versatile
		"llama-3.1-",      // Llama 3.1 8B instant
		"openai/gpt-oss-", // OpenAI GPT OSS models hosted on Groq
	}

	for _, prefix := range groqPrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}

	// Check for Groq-hosted Llama 4 models (meta-llama/llama-4-*)
	// These must be distinguished from HuggingFace's meta-llama/ models
	if strings.HasPrefix(model, "meta-llama/llama-4-") {
		return true
	}

	// Check for Groq-hosted Qwen3 models
	if strings.HasPrefix(model, "qwen/qwen3") {
		return true
	}

	// Check for Groq-hosted Kimi K2 models
	if strings.HasPrefix(model, "moonshotai/kimi-k2") {
		return true
	}

	return false
}

// isBedrockModel checks if the model name indicates an AWS Bedrock model
func isBedrockModel(model string) bool {
	bedrockPrefixes := []string{
		"amazon.",    // Amazon Titan models
		"anthropic.", // Anthropic models on Bedrock
		"cohere.",    // Cohere models on Bedrock
		"meta.",      // Meta Llama models on Bedrock
		"ai21.",      // AI21 models on Bedrock
		"stability.", // Stability AI models on Bedrock
	}

	for _, prefix := range bedrockPrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	return false
}

// isHuggingFaceModel checks if the model name indicates a HuggingFace model
func isHuggingFaceModel(model string) bool {
	// HuggingFace models typically use org/model format
	if strings.Contains(model, "/") {
		huggingFaceOrgs := []string{
			"sentence-transformers/",
			"baai/",
			"intfloat/",
			"meta-llama/",
			"mistralai/",
			"google/",
			"microsoft/",
			"facebook/",
			"huggingface/",
		}
		for _, org := range huggingFaceOrgs {
			if strings.HasPrefix(model, org) {
				return true
			}
		}
	}

	// Also check for common HuggingFace model prefixes without org
	huggingFacePrefixes := []string{
		"bge-",        // BAAI BGE models
		"e5-",         // E5 models
		"instructor-", // Instructor models
		"gte-",        // GTE models
		"nomic-",      // Nomic models
	}
	for _, prefix := range huggingFacePrefixes {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}

	return false
}

// IsReasoningModelByName checks if a model is a reasoning model based on its name.
// This is useful when you don't know the provider.
func IsReasoningModelByName(modelName string) bool {
	model := strings.ToLower(modelName)

	// OpenAI o-series reasoning models
	if strings.HasPrefix(model, "o1") ||
		strings.HasPrefix(model, "o3") ||
		strings.HasPrefix(model, "o4") {
		return true
	}

	// GPT-5 series are reasoning models
	if strings.HasPrefix(model, "gpt-5") {
		return true
	}

	// Cohere reasoning models
	if strings.Contains(model, "reasoning") {
		return true
	}

	// Check the catalog for explicit reasoning flag
	provider := DetectProvider(modelName)
	if provider != ProviderUnknown {
		if m := GetModel(provider, modelName); m != nil {
			return m.IsReasoning
		}
	}

	return false
}

// GetMaxTokensForModel returns the max tokens for a model.
// Returns 0 if the model is not found in the catalog.
func GetMaxTokensForModel(modelName string) int {
	provider := DetectProvider(modelName)
	if provider == ProviderUnknown {
		return 0
	}

	model := GetModel(provider, modelName)
	if model != nil {
		return model.MaxTokens
	}

	// Try to find by prefix match for models with version suffixes
	providerCatalog := GetProviderCatalog(provider)
	if providerCatalog != nil {
		lowerName := strings.ToLower(modelName)
		for _, m := range providerCatalog.Models {
			if strings.HasPrefix(lowerName, strings.ToLower(m.Name)) {
				return m.MaxTokens
			}
		}
	}

	return 0
}

// GetDefaultMaxTokens returns the context window size for a model.
// This is the total input + output token limit (NOT the max output tokens).
// For max output tokens, use GetDefaultMaxOutputTokens instead.
//
// If the model is in the catalog, returns its MaxTokens.
// Otherwise returns a reasonable default based on the provider.
func GetDefaultMaxTokens(modelName string) int {
	// First check catalog
	if maxTokens := GetMaxTokensForModel(modelName); maxTokens > 0 {
		return maxTokens
	}

	// Provider-based defaults
	provider := DetectProvider(modelName)
	switch provider {
	case ProviderOpenAI:
		return 128000 // Modern OpenAI models
	case ProviderAnthropic:
		return 200000 // Claude models
	case ProviderCohere:
		return 128000 // Command models
	case ProviderBedrock:
		return 200000 // Varies, but Anthropic on Bedrock is common
	case ProviderHuggingFace:
		return 8192 // Most HF models are smaller
	case ProviderGroq:
		return 131072 // Groq models support 128K context
	case ProviderLMStudio, ProviderOllama:
		return 8192 // Local models typically smaller
	default:
		return 4096 // Conservative default
	}
}

// GetDefaultMaxOutputTokens returns the max output/completion tokens for a model.
// This is the limit for how many tokens the model can generate in a single response.
// This is different from the context window (GetDefaultMaxTokens).
//
// If the model has MaxOutputTokens set in the catalog, returns that.
// Otherwise returns a reasonable default based on the provider.
func GetDefaultMaxOutputTokens(modelName string) int {
	// First check catalog for explicit MaxOutputTokens
	provider := DetectProvider(modelName)
	if provider != ProviderUnknown {
		if m := GetModel(provider, modelName); m != nil && m.MaxOutputTokens > 0 {
			return m.MaxOutputTokens
		}
	}

	// Provider-based defaults for max output tokens
	switch provider {
	case ProviderOpenAI:
		// GPT-4o: 16384, GPT-4.1: 32768, o1: 32768-100000
		// Use 16384 as safe default for most models
		return 16384
	case ProviderAnthropic:
		// Claude 3.5/4.5: 8192 default, can be increased
		return 8192
	case ProviderCohere:
		// Command models: 4096 default
		return 4096
	case ProviderBedrock:
		// Varies by model, but 8192 is common
		return 8192
	case ProviderHuggingFace:
		return 4096
	case ProviderGroq:
		return 32768 // Groq default max output
	case ProviderLMStudio, ProviderOllama:
		return 4096 // Local models typically have smaller limits
	default:
		return 4096 // Conservative default
	}
}
