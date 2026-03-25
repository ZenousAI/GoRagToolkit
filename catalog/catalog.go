// Package catalog provides a comprehensive AI model catalog.
// It is the single source of truth for model metadata across all major providers.
package catalog

// ModelType represents the type of model
type ModelType string

const (
	ModelTypeEmbedding ModelType = "embedding"
	ModelTypeChat      ModelType = "chat"
	ModelTypeRerank    ModelType = "rerank"
)

// ProviderType represents the type of provider
type ProviderType string

const (
	ProviderOpenAI      ProviderType = "openai"
	ProviderAnthropic   ProviderType = "anthropic"
	ProviderCohere      ProviderType = "cohere"
	ProviderBedrock     ProviderType = "bedrock"
	ProviderHuggingFace ProviderType = "huggingface"
	ProviderLMStudio    ProviderType = "lmstudio"
	ProviderOllama      ProviderType = "ollama"
	ProviderCustom     ProviderType = "custom"
	ProviderJina       ProviderType = "jina" // Jina AI (embeddings, reranking)
	ProviderMixedbread  ProviderType = "mixedbread" // Mixedbread (embeddings, reranking)
	ProviderGroq        ProviderType = "groq"       // Groq (LPU-accelerated chat inference)
	ProviderUnknown     ProviderType = "unknown"
)

// Model represents a model in the catalog
type Model struct {
	Name            string    `json:"name"`
	Type            ModelType `json:"type"`
	Dimension       *int      `json:"dimension,omitempty"`         // For embedding models
	MaxTokens       int       `json:"max_tokens"`                  // Context window size (input + output)
	MaxOutputTokens int       `json:"max_output_tokens,omitempty"` // Max output/completion tokens (0 = use default)
	MaxChunkSize    int       `json:"max_chunk_size"`              // Recommended chunk size for this model
	Description     string    `json:"description,omitempty"`       // Optional description
	Deprecated      bool      `json:"deprecated,omitempty"`        // Whether this model is deprecated
	IsReasoning     bool      `json:"is_reasoning,omitempty"`      // Whether this is a reasoning model (requires special handling)
}

// ProviderCatalog contains all models for a provider
type ProviderCatalog struct {
	Provider    ProviderType `json:"provider"`
	DisplayName string       `json:"display_name"`
	Models      []Model      `json:"models"`
}

// Catalog is the complete model catalog
type Catalog struct {
	Version   string            `json:"version"`
	Providers []ProviderCatalog `json:"providers"`
}

// GetCatalog returns the complete model catalog
func GetCatalog() *Catalog {
	return &Catalog{
		Version: "1.0.0",
		Providers: []ProviderCatalog{
			getOpenAICatalog(),
			getCohereeCatalog(),
			getAnthropicCatalog(),
			getBedrockCatalog(),
			getHuggingFaceCatalog(),
			getLMStudioCatalog(),
			getGroqCatalog(),
		},
	}
}

// GetProviderCatalog returns the catalog for a specific provider
func GetProviderCatalog(provider ProviderType) *ProviderCatalog {
	catalog := GetCatalog()
	for _, p := range catalog.Providers {
		if p.Provider == provider {
			return &p
		}
	}
	return nil
}

// GetModelsByType returns all models of a specific type across all providers
func GetModelsByType(modelType ModelType) []Model {
	var models []Model
	catalog := GetCatalog()
	for _, p := range catalog.Providers {
		for _, m := range p.Models {
			if m.Type == modelType {
				models = append(models, m)
			}
		}
	}
	return models
}

// GetModel returns a specific model by provider and name
func GetModel(provider ProviderType, name string) *Model {
	providerCatalog := GetProviderCatalog(provider)
	if providerCatalog == nil {
		return nil
	}
	for _, m := range providerCatalog.Models {
		if m.Name == name {
			return &m
		}
	}
	return nil
}

// IsReasoningModel checks if a model requires special reasoning model handling
func IsReasoningModel(provider ProviderType, name string) bool {
	model := GetModel(provider, name)
	if model != nil {
		return model.IsReasoning
	}
	return false
}

func getOpenAICatalog() ProviderCatalog {
	return ProviderCatalog{
		Provider:    ProviderOpenAI,
		DisplayName: "OpenAI",
		Models: []Model{
			// Embedding models
			{
				Name:         "text-embedding-3-small",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1536),
				MaxTokens:    8192,
				MaxChunkSize: 512,
				Description:  "Smaller, efficient embedding model",
			},
			{
				Name:         "text-embedding-3-large",
				Type:         ModelTypeEmbedding,
				Dimension:    new(3072),
				MaxTokens:    8192,
				MaxChunkSize: 512,
				Description:  "Larger, more capable embedding model",
			},
			{
				Name:         "text-embedding-ada-002",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1536),
				MaxTokens:    8192,
				MaxChunkSize: 512,
				Description:  "Legacy embedding model",
				Deprecated:   true,
			},
			// GPT-5.2 series (Latest flagship - reasoning models)
			{
				Name:         "gpt-5.2",
				Type:         ModelTypeChat,
				MaxTokens:    400000,
				MaxChunkSize: 8192,
				Description:  "Latest flagship reasoning model",
				IsReasoning:  true,
			},
			{
				Name:         "gpt-5.2-pro",
				Type:         ModelTypeChat,
				MaxTokens:    400000,
				MaxChunkSize: 8192,
				Description:  "Professional tier GPT-5.2",
				IsReasoning:  true,
			},
			// GPT-5 series (reasoning models)
			{
				Name:         "gpt-5",
				Type:         ModelTypeChat,
				MaxTokens:    400000,
				MaxChunkSize: 8192,
				Description:  "GPT-5 reasoning model",
				IsReasoning:  true,
			},
			{
				Name:         "gpt-5-mini",
				Type:         ModelTypeChat,
				MaxTokens:    400000,
				MaxChunkSize: 8192,
				Description:  "Smaller GPT-5 reasoning model",
				IsReasoning:  true,
			},
			{
				Name:         "gpt-5-nano",
				Type:         ModelTypeChat,
				MaxTokens:    400000,
				MaxChunkSize: 8192,
				Description:  "Lightweight GPT-5 reasoning model",
				IsReasoning:  true,
			},
			// GPT-4.1 series (non-reasoning, 1M context)
			{
				Name:         "gpt-4.1",
				Type:         ModelTypeChat,
				MaxTokens:    1047576,
				MaxChunkSize: 8192,
				Description:  "GPT-4.1 with 1M context window",
			},
			{
				Name:         "gpt-4.1-mini",
				Type:         ModelTypeChat,
				MaxTokens:    1047576,
				MaxChunkSize: 8192,
				Description:  "Smaller GPT-4.1 with 1M context",
			},
			{
				Name:         "gpt-4.1-nano",
				Type:         ModelTypeChat,
				MaxTokens:    1047576,
				MaxChunkSize: 8192,
				Description:  "Lightweight GPT-4.1 with 1M context",
			},
			// o-series reasoning models
			{
				Name:         "o3",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "OpenAI o3 reasoning model",
				IsReasoning:  true,
			},
			{
				Name:         "o3-mini",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "Smaller o3 reasoning model",
				IsReasoning:  true,
			},
			{
				Name:         "o4-mini",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "OpenAI o4-mini reasoning model",
				IsReasoning:  true,
			},
			{
				Name:         "o1",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "OpenAI o1 reasoning model",
				IsReasoning:  true,
			},
			{
				Name:         "o1-mini",
				Type:         ModelTypeChat,
				MaxTokens:    128000,
				MaxChunkSize: 4096,
				Description:  "Smaller o1 reasoning model",
				IsReasoning:  true,
			},
			// GPT-4o series
			{
				Name:         "gpt-4o",
				Type:         ModelTypeChat,
				MaxTokens:    128000,
				MaxChunkSize: 4096,
				Description:  "GPT-4o multimodal model",
			},
			{
				Name:         "gpt-4o-mini",
				Type:         ModelTypeChat,
				MaxTokens:    128000,
				MaxChunkSize: 4096,
				Description:  "Smaller GPT-4o multimodal model",
			},
			{
				Name:         "gpt-4-turbo",
				Type:         ModelTypeChat,
				MaxTokens:    128000,
				MaxChunkSize: 4096,
				Description:  "GPT-4 Turbo",
			},
			{
				Name:         "gpt-3.5-turbo",
				Type:         ModelTypeChat,
				MaxTokens:    16385,
				MaxChunkSize: 4096,
				Description:  "GPT-3.5 Turbo (legacy)",
				Deprecated:   true,
			},
		},
	}
}

func getCohereeCatalog() ProviderCatalog {
	return ProviderCatalog{
		Provider:    ProviderCohere,
		DisplayName: "Cohere",
		Models: []Model{
			// Embed v4 (latest)
			{
				Name:         "embed-v4.0",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1536),
				MaxTokens:    128000,
				MaxChunkSize: 2048,
				Description:  "Latest Cohere embedding model",
			},
			// Embed v3 series
			{
				Name:         "embed-english-v3.0",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1024),
				MaxTokens:    512,
				MaxChunkSize: 512,
				Description:  "English embedding model v3",
			},
			{
				Name:         "embed-multilingual-v3.0",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1024),
				MaxTokens:    512,
				MaxChunkSize: 512,
				Description:  "Multilingual embedding model v3",
			},
			{
				Name:         "embed-english-light-v3.0",
				Type:         ModelTypeEmbedding,
				Dimension:    new(384),
				MaxTokens:    512,
				MaxChunkSize: 512,
				Description:  "Lightweight English embedding model",
			},
			// Command A series (latest)
			{
				Name:         "command-a-03-2025",
				Type:         ModelTypeChat,
				MaxTokens:    256000,
				MaxChunkSize: 8192,
				Description:  "Command A - latest flagship model",
			},
			{
				Name:         "command-a-reasoning-08-2025",
				Type:         ModelTypeChat,
				MaxTokens:    256000,
				MaxChunkSize: 8192,
				Description:  "Command A with reasoning capabilities",
				IsReasoning:  true,
			},
			{
				Name:         "command-a-vision-07-2025",
				Type:         ModelTypeChat,
				MaxTokens:    128000,
				MaxChunkSize: 8192,
				Description:  "Command A with vision capabilities",
			},
			// Command R series
			{
				Name:         "command-r7b-12-2024",
				Type:         ModelTypeChat,
				MaxTokens:    128000,
				MaxChunkSize: 4096,
				Description:  "Command R 7B model",
			},
			{
				Name:         "command-r-plus-08-2024",
				Type:         ModelTypeChat,
				MaxTokens:    128000,
				MaxChunkSize: 4096,
				Description:  "Command R Plus",
			},
			{
				Name:         "command-r-08-2024",
				Type:         ModelTypeChat,
				MaxTokens:    128000,
				MaxChunkSize: 4096,
				Description:  "Command R",
			},
			// Rerank models
			{
				Name:         "rerank-v4.0-pro",
				Type:         ModelTypeRerank,
				MaxTokens:    32000,
				MaxChunkSize: 2048,
				Description:  "Professional reranking model v4",
			},
			{
				Name:         "rerank-v4.0-fast",
				Type:         ModelTypeRerank,
				MaxTokens:    32000,
				MaxChunkSize: 2048,
				Description:  "Fast reranking model v4",
			},
			{
				Name:         "rerank-v3.5",
				Type:         ModelTypeRerank,
				MaxTokens:    4096,
				MaxChunkSize: 512,
				Description:  "Reranking model v3.5",
			},
			// Aya multilingual models
			{
				Name:         "c4ai-aya-expanse-32b",
				Type:         ModelTypeChat,
				MaxTokens:    128000,
				MaxChunkSize: 4096,
				Description:  "Aya Expanse 32B multilingual model",
			},
			{
				Name:         "c4ai-aya-expanse-8b",
				Type:         ModelTypeChat,
				MaxTokens:    8000,
				MaxChunkSize: 2048,
				Description:  "Aya Expanse 8B multilingual model",
			},
		},
	}
}

func getAnthropicCatalog() ProviderCatalog {
	return ProviderCatalog{
		Provider:    ProviderAnthropic,
		DisplayName: "Anthropic",
		Models: []Model{
			// Claude 4.5 series (latest)
			{
				Name:         "claude-sonnet-4-5",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 8192,
				Description:  "Claude Sonnet 4.5 - balanced performance",
			},
			{
				Name:         "claude-opus-4-5",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 8192,
				Description:  "Claude Opus 4.5 - highest capability",
			},
			{
				Name:         "claude-haiku-4-5",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 8192,
				Description:  "Claude Haiku 4.5 - fast and efficient",
			},
			// Claude 4.5 with snapshot dates
			{
				Name:         "claude-sonnet-4-5-20250929",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 8192,
				Description:  "Claude Sonnet 4.5 (2025-09-29 snapshot)",
			},
			{
				Name:         "claude-opus-4-5-20251101",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 8192,
				Description:  "Claude Opus 4.5 (2025-11-01 snapshot)",
			},
			{
				Name:         "claude-haiku-4-5-20251001",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 8192,
				Description:  "Claude Haiku 4.5 (2025-10-01 snapshot)",
			},
			// Claude 3.5 series (legacy)
			{
				Name:         "claude-3-5-sonnet-20241022",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "Claude 3.5 Sonnet",
				Deprecated:   true,
			},
			{
				Name:         "claude-3-5-haiku-20241022",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "Claude 3.5 Haiku",
				Deprecated:   true,
			},
			// Claude 3 series (legacy)
			{
				Name:         "claude-3-opus-20240229",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "Claude 3 Opus",
				Deprecated:   true,
			},
			{
				Name:         "claude-3-sonnet-20240229",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "Claude 3 Sonnet",
				Deprecated:   true,
			},
			{
				Name:         "claude-3-haiku-20240307",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "Claude 3 Haiku",
				Deprecated:   true,
			},
		},
	}
}

func getBedrockCatalog() ProviderCatalog {
	return ProviderCatalog{
		Provider:    ProviderBedrock,
		DisplayName: "AWS Bedrock",
		Models: []Model{
			// Amazon Titan embedding models
			{
				Name:         "amazon.titan-embed-text-v1",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1536),
				MaxTokens:    8192,
				MaxChunkSize: 512,
				Description:  "Amazon Titan Text Embeddings v1",
			},
			{
				Name:         "amazon.titan-embed-text-v2:0",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1024),
				MaxTokens:    8192,
				MaxChunkSize: 512,
				Description:  "Amazon Titan Text Embeddings v2",
			},
			// Cohere on Bedrock
			{
				Name:         "cohere.embed-english-v3",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1024),
				MaxTokens:    512,
				MaxChunkSize: 512,
				Description:  "Cohere English embeddings on Bedrock",
			},
			{
				Name:         "cohere.embed-multilingual-v3",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1024),
				MaxTokens:    512,
				MaxChunkSize: 512,
				Description:  "Cohere multilingual embeddings on Bedrock",
			},
			{
				Name:         "cohere.rerank-v3-5:0",
				Type:         ModelTypeRerank,
				MaxTokens:    4096,
				MaxChunkSize: 512,
				Description:  "Cohere Rerank v3.5 on Bedrock",
			},
			// Anthropic Claude 4.5 on Bedrock
			{
				Name:         "anthropic.claude-sonnet-4-5-20250929-v1:0",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 8192,
				Description:  "Claude Sonnet 4.5 on Bedrock",
			},
			{
				Name:         "anthropic.claude-haiku-4-5-20251001-v1:0",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 8192,
				Description:  "Claude Haiku 4.5 on Bedrock",
			},
			{
				Name:         "anthropic.claude-opus-4-5-20251101-v1:0",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 8192,
				Description:  "Claude Opus 4.5 on Bedrock",
			},
			// Anthropic Claude 3 on Bedrock (legacy)
			{
				Name:         "anthropic.claude-3-sonnet-20240229-v1:0",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "Claude 3 Sonnet on Bedrock",
				Deprecated:   true,
			},
			{
				Name:         "anthropic.claude-3-haiku-20240307-v1:0",
				Type:         ModelTypeChat,
				MaxTokens:    200000,
				MaxChunkSize: 4096,
				Description:  "Claude 3 Haiku on Bedrock",
				Deprecated:   true,
			},
			// Meta Llama on Bedrock
			{
				Name:         "meta.llama3-70b-instruct-v1:0",
				Type:         ModelTypeChat,
				MaxTokens:    8192,
				MaxChunkSize: 2048,
				Description:  "Meta Llama 3 70B Instruct on Bedrock",
			},
		},
	}
}

func getHuggingFaceCatalog() ProviderCatalog {
	return ProviderCatalog{
		Provider:    ProviderHuggingFace,
		DisplayName: "Hugging Face",
		Models: []Model{
			{
				Name:         "sentence-transformers/all-MiniLM-L6-v2",
				Type:         ModelTypeEmbedding,
				Dimension:    new(384),
				MaxTokens:    512,
				MaxChunkSize: 256,
				Description:  "Lightweight sentence transformer",
			},
			{
				Name:         "sentence-transformers/all-mpnet-base-v2",
				Type:         ModelTypeEmbedding,
				Dimension:    new(768),
				MaxTokens:    512,
				MaxChunkSize: 512,
				Description:  "MPNet-based sentence transformer",
			},
			{
				Name:         "BAAI/bge-large-en-v1.5",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1024),
				MaxTokens:    512,
				MaxChunkSize: 512,
				Description:  "BGE Large English embeddings",
			},
			{
				Name:         "BAAI/bge-base-en-v1.5",
				Type:         ModelTypeEmbedding,
				Dimension:    new(768),
				MaxTokens:    512,
				MaxChunkSize: 512,
				Description:  "BGE Base English embeddings",
			},
			{
				Name:         "BAAI/bge-small-en-v1.5",
				Type:         ModelTypeEmbedding,
				Dimension:    new(384),
				MaxTokens:    512,
				MaxChunkSize: 512,
				Description:  "BGE Small English embeddings",
			},
			{
				Name:         "intfloat/e5-large-v2",
				Type:         ModelTypeEmbedding,
				Dimension:    new(1024),
				MaxTokens:    512,
				MaxChunkSize: 512,
				Description:  "E5 Large embeddings",
			},
			{
				Name:         "meta-llama/Meta-Llama-3-8B-Instruct",
				Type:         ModelTypeChat,
				MaxTokens:    8192,
				MaxChunkSize: 2048,
				Description:  "Meta Llama 3 8B Instruct",
			},
			{
				Name:         "mistralai/Mistral-7B-Instruct-v0.2",
				Type:         ModelTypeChat,
				MaxTokens:    32768,
				MaxChunkSize: 4096,
				Description:  "Mistral 7B Instruct v0.2",
			},
		},
	}
}

func getLMStudioCatalog() ProviderCatalog {
	return ProviderCatalog{
		Provider:    ProviderLMStudio,
		DisplayName: "LM Studio",
		Models: []Model{
			{
				Name:         "nomic-embed-text-v1.5",
				Type:         ModelTypeEmbedding,
				Dimension:    new(768),
				MaxTokens:    8192,
				MaxChunkSize: 512,
				Description:  "Nomic embeddings for local use",
			},
			{
				Name:         "llama-3-8b-instruct",
				Type:         ModelTypeChat,
				MaxTokens:    8192,
				MaxChunkSize: 2048,
				Description:  "Llama 3 8B for local inference",
			},
			{
				Name:         "mistral-7b-instruct",
				Type:         ModelTypeChat,
				MaxTokens:    8192,
				MaxChunkSize: 2048,
				Description:  "Mistral 7B for local inference",
			},
			{
				Name:         "phi-3-mini",
				Type:         ModelTypeChat,
				MaxTokens:    4096,
				MaxChunkSize: 1024,
				Description:  "Phi-3 Mini for local inference",
			},
		},
	}
}

func getGroqCatalog() ProviderCatalog {
	return ProviderCatalog{
		Provider:    ProviderGroq,
		DisplayName: "Groq",
		Models: []Model{
			// Production Models
			{
				Name:            "llama-3.3-70b-versatile",
				Type:            ModelTypeChat,
				MaxTokens:       131072,
				MaxOutputTokens: 32768,
				MaxChunkSize:    4096,
				Description:     "General-purpose, tool calling",
			},
			{
				Name:            "llama-3.1-8b-instant",
				Type:            ModelTypeChat,
				MaxTokens:       131072,
				MaxOutputTokens: 131072,
				MaxChunkSize:    4096,
				Description:     "Ultra-fast, low-cost",
			},
			{
				Name:            "openai/gpt-oss-120b",
				Type:            ModelTypeChat,
				MaxTokens:       131072,
				MaxOutputTokens: 65536,
				MaxChunkSize:    4096,
				Description:     "OpenAI GPT OSS 120B",
			},
			{
				Name:            "openai/gpt-oss-20b",
				Type:            ModelTypeChat,
				MaxTokens:       131072,
				MaxOutputTokens: 65536,
				MaxChunkSize:    4096,
				Description:     "OpenAI GPT OSS 20B",
			},
			{
				Name:            "groq/compound",
				Type:            ModelTypeChat,
				MaxTokens:       131072,
				MaxOutputTokens: 8192,
				MaxChunkSize:    4096,
				Description:     "Groq Compound system with built-in tools (web search, code execution)",
			},
			{
				Name:            "groq/compound-mini",
				Type:            ModelTypeChat,
				MaxTokens:       131072,
				MaxOutputTokens: 8192,
				MaxChunkSize:    4096,
				Description:     "Groq Compound Mini system with built-in tools",
			},
			// Preview Models (evaluation only, not production-ready)
			{
				Name:            "meta-llama/llama-4-scout-17b-16e-instruct",
				Type:            ModelTypeChat,
				MaxTokens:       131072,
				MaxOutputTokens: 8192,
				MaxChunkSize:    4096,
				Description:     "Preview: Vision + tool calling (17Bx16E)",
			},
			{
				Name:            "meta-llama/llama-4-maverick-17b-128e-instruct",
				Type:            ModelTypeChat,
				MaxTokens:       131072,
				MaxOutputTokens: 8192,
				MaxChunkSize:    4096,
				Description:     "Preview: Vision + tool calling (17Bx128E)",
			},
			{
				Name:            "qwen/qwen3-32b",
				Type:            ModelTypeChat,
				MaxTokens:       131072,
				MaxOutputTokens: 40960,
				MaxChunkSize:    4096,
				Description:     "Preview: Multilingual (100+ languages)",
			},
			{
				Name:            "moonshotai/kimi-k2-instruct-0905",
				Type:            ModelTypeChat,
				MaxTokens:       262144,
				MaxOutputTokens: 16384,
				MaxChunkSize:    4096,
				Description:     "Preview: Kimi K2 with 256K context",
			},
		},
	}
}
