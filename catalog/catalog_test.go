package catalog

import (
	"testing"
)

func TestGetCatalog(t *testing.T) {
	catalog := GetCatalog()

	if catalog == nil {
		t.Fatal("GetCatalog returned nil")
	}

	if catalog.Version == "" {
		t.Error("Catalog version is empty")
	}

	if len(catalog.Providers) == 0 {
		t.Error("Catalog has no providers")
	}

	// Verify expected providers exist
	expectedProviders := []ProviderType{
		ProviderOpenAI,
		ProviderAnthropic,
		ProviderCohere,
		ProviderBedrock,
		ProviderHuggingFace,
		ProviderLMStudio,
		ProviderGroq,
	}

	for _, expected := range expectedProviders {
		found := false
		for _, p := range catalog.Providers {
			if p.Provider == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected provider %s not found in catalog", expected)
		}
	}
}

func TestGetProviderCatalog(t *testing.T) {
	tests := []struct {
		name     string
		provider ProviderType
		wantNil  bool
	}{
		{"OpenAI exists", ProviderOpenAI, false},
		{"Anthropic exists", ProviderAnthropic, false},
		{"Cohere exists", ProviderCohere, false},
		{"Bedrock exists", ProviderBedrock, false},
		{"HuggingFace exists", ProviderHuggingFace, false},
		{"LMStudio exists", ProviderLMStudio, false},
		{"Groq exists", ProviderGroq, false},
		{"Unknown provider returns nil", ProviderType("unknown"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetProviderCatalog(tt.provider)
			if tt.wantNil && result != nil {
				t.Errorf("Expected nil for provider %s, got %+v", tt.provider, result)
			}
			if !tt.wantNil && result == nil {
				t.Errorf("Expected non-nil for provider %s", tt.provider)
			}
			if !tt.wantNil && result != nil {
				if result.Provider != tt.provider {
					t.Errorf("Provider mismatch: got %s, want %s", result.Provider, tt.provider)
				}
				if len(result.Models) == 0 {
					t.Errorf("Provider %s has no models", tt.provider)
				}
			}
		})
	}
}

func TestGetModelsByType(t *testing.T) {
	tests := []struct {
		name      string
		modelType ModelType
		wantMin   int
	}{
		{"Embedding models exist", ModelTypeEmbedding, 5},
		{"Chat models exist", ModelTypeChat, 10},
		{"Rerank models exist", ModelTypeRerank, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			models := GetModelsByType(tt.modelType)
			if len(models) < tt.wantMin {
				t.Errorf("Expected at least %d %s models, got %d", tt.wantMin, tt.modelType, len(models))
			}

			// Verify all returned models have the correct type
			for _, m := range models {
				if m.Type != tt.modelType {
					t.Errorf("Model %s has type %s, expected %s", m.Name, m.Type, tt.modelType)
				}
			}
		})
	}
}

func TestGetModel(t *testing.T) {
	tests := []struct {
		name      string
		provider  ProviderType
		modelName string
		wantNil   bool
		wantType  ModelType
	}{
		{"OpenAI embedding model", ProviderOpenAI, "text-embedding-3-small", false, ModelTypeEmbedding},
		{"OpenAI chat model", ProviderOpenAI, "gpt-4o", false, ModelTypeChat},
		{"Anthropic chat model", ProviderAnthropic, "claude-sonnet-4-5", false, ModelTypeChat},
		{"Cohere embedding model", ProviderCohere, "embed-v4.0", false, ModelTypeEmbedding},
		{"Cohere rerank model", ProviderCohere, "rerank-v4.0-pro", false, ModelTypeRerank},
		{"Groq chat model", ProviderGroq, "llama-3.3-70b-versatile", false, ModelTypeChat},
		{"Groq Qwen model", ProviderGroq, "qwen/qwen3-32b", false, ModelTypeChat},
		{"Non-existent model", ProviderOpenAI, "non-existent-model", true, ""},
		{"Non-existent provider", ProviderType("fake"), "gpt-4o", true, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := GetModel(tt.provider, tt.modelName)
			if tt.wantNil && model != nil {
				t.Errorf("Expected nil for %s/%s, got %+v", tt.provider, tt.modelName, model)
			}
			if !tt.wantNil && model == nil {
				t.Errorf("Expected non-nil for %s/%s", tt.provider, tt.modelName)
			}
			if !tt.wantNil && model != nil {
				if model.Name != tt.modelName {
					t.Errorf("Model name mismatch: got %s, want %s", model.Name, tt.modelName)
				}
				if model.Type != tt.wantType {
					t.Errorf("Model type mismatch: got %s, want %s", model.Type, tt.wantType)
				}
			}
		})
	}
}

func TestIsReasoningModel(t *testing.T) {
	tests := []struct {
		name        string
		provider    ProviderType
		modelName   string
		isReasoning bool
	}{
		{"GPT-5 is reasoning", ProviderOpenAI, "gpt-5", true},
		{"GPT-5.2 is reasoning", ProviderOpenAI, "gpt-5.2", true},
		{"o3 is reasoning", ProviderOpenAI, "o3", true},
		{"o1 is reasoning", ProviderOpenAI, "o1", true},
		{"GPT-4o is not reasoning", ProviderOpenAI, "gpt-4o", false},
		{"GPT-4.1 is not reasoning", ProviderOpenAI, "gpt-4.1", false},
		{"Claude is not reasoning", ProviderAnthropic, "claude-sonnet-4-5", false},
		{"Command A reasoning is reasoning", ProviderCohere, "command-a-reasoning-08-2025", true},
		{"Command A is not reasoning", ProviderCohere, "command-a-03-2025", false},
		{"Non-existent model returns false", ProviderOpenAI, "non-existent", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsReasoningModel(tt.provider, tt.modelName)
			if result != tt.isReasoning {
				t.Errorf("IsReasoningModel(%s, %s) = %v, want %v", tt.provider, tt.modelName, result, tt.isReasoning)
			}
		})
	}
}

func TestEmbeddingModelDimensions(t *testing.T) {
	// Verify all embedding models have dimensions set
	embeddingModels := GetModelsByType(ModelTypeEmbedding)

	for _, m := range embeddingModels {
		if m.Dimension == nil {
			t.Errorf("Embedding model %s has no dimension set", m.Name)
		} else if *m.Dimension <= 0 {
			t.Errorf("Embedding model %s has invalid dimension: %d", m.Name, *m.Dimension)
		}
	}
}

func TestChatModelsNoDimension(t *testing.T) {
	// Verify chat models don't have dimensions set (they shouldn't)
	chatModels := GetModelsByType(ModelTypeChat)

	for _, m := range chatModels {
		if m.Dimension != nil {
			t.Errorf("Chat model %s should not have dimension set, got %d", m.Name, *m.Dimension)
		}
	}
}

func TestModelMaxTokens(t *testing.T) {
	catalog := GetCatalog()

	for _, provider := range catalog.Providers {
		for _, m := range provider.Models {
			if m.MaxTokens <= 0 {
				t.Errorf("Model %s/%s has invalid max_tokens: %d", provider.Provider, m.Name, m.MaxTokens)
			}
		}
	}
}

func TestModelMaxChunkSize(t *testing.T) {
	catalog := GetCatalog()

	for _, provider := range catalog.Providers {
		for _, m := range provider.Models {
			if m.MaxChunkSize <= 0 {
				t.Errorf("Model %s/%s has invalid max_chunk_size: %d", provider.Provider, m.Name, m.MaxChunkSize)
			}
		}
	}
}

func TestProviderDisplayNames(t *testing.T) {
	catalog := GetCatalog()

	for _, provider := range catalog.Providers {
		if provider.DisplayName == "" {
			t.Errorf("Provider %s has no display name", provider.Provider)
		}
	}
}

func TestDetectProvider(t *testing.T) {
	tests := []struct {
		name             string
		modelName        string
		expectedProvider ProviderType
	}{
		// OpenAI models
		{"GPT-4 Turbo", "gpt-4-turbo", ProviderOpenAI},
		{"GPT-3.5", "gpt-3.5-turbo", ProviderOpenAI},
		{"o1-preview", "o1-preview", ProviderOpenAI},

		// Anthropic models
		{"Claude Sonnet", "claude-3-5-sonnet-20241022", ProviderAnthropic},
		{"Claude Opus", "claude-opus-4-6", ProviderAnthropic},

		// Cohere models
		{"Command R", "command-r", ProviderCohere},
		{"Embed v3", "embed-english-v3.0", ProviderCohere},

		// Groq models - critical test cases
		{"Groq Llama 3.3 70B", "llama-3.3-70b-versatile", ProviderGroq},
		{"Groq Llama 3.1 8B", "llama-3.1-8b-instant", ProviderGroq},
		{"Groq Llama 4 Scout", "meta-llama/llama-4-scout-17b-16e-instruct", ProviderGroq},
		{"Groq Llama 4 Maverick", "meta-llama/llama-4-maverick-17b-128e-instruct", ProviderGroq},
		{"Groq Qwen3", "qwen/qwen3-32b", ProviderGroq},
		{"Groq with explicit prefix", "groq/llama-3-70b", ProviderGroq},
		{"Groq Kimi K2", "moonshotai/kimi-k2-instruct-0905", ProviderGroq},
		{"Groq GPT OSS 120B", "openai/gpt-oss-120b", ProviderGroq},
		{"Groq GPT OSS 20B", "openai/gpt-oss-20b", ProviderGroq},
		{"Groq Compound", "groq/compound", ProviderGroq},
		{"Groq Compound Mini", "groq/compound-mini", ProviderGroq},

		// HuggingFace models (should NOT conflict with Groq)
		{"HF Llama 3", "meta-llama/llama-3-8b", ProviderHuggingFace},
		{"HF Mistral", "mistralai/mistral-7b", ProviderHuggingFace},
		{"HF BGE", "bge-large-en-v1.5", ProviderHuggingFace},

		// Bedrock models
		{"Bedrock Titan", "amazon.titan-embed-text-v1", ProviderBedrock},
		{"Bedrock Claude", "anthropic.claude-v2", ProviderBedrock},

		// Jina models
		{"Jina Embeddings", "jina-embeddings-v3", ProviderJina},

		// Unknown models
		{"Unknown model", "some-random-model", ProviderUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DetectProvider(tt.modelName)
			if got != tt.expectedProvider {
				t.Errorf("DetectProvider(%q) = %v, want %v", tt.modelName, got, tt.expectedProvider)
			}
		})
	}
}

func TestGroqModelsNotMisdetectedAsHuggingFace(t *testing.T) {
	// Critical regression test: Groq models with meta-llama/ and qwen/ prefixes
	// should be detected as Groq, NOT HuggingFace
	groqModels := []string{
		"meta-llama/llama-4-scout-17b-16e-instruct",
		"meta-llama/llama-4-maverick-17b-128e-instruct",
		"qwen/qwen3-32b",
	}

	for _, model := range groqModels {
		t.Run(model, func(t *testing.T) {
			provider := DetectProvider(model)
			if provider != ProviderGroq {
				t.Errorf("Model %q detected as %v, expected ProviderGroq. This model will fail at runtime!", model, provider)
			}
		})
	}
}
