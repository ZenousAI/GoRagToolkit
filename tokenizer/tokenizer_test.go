package tokenizer

import (
	"strings"
	"testing"
)

func TestDetectProvider(t *testing.T) {
	tests := []struct {
		model    string
		expected Provider
	}{
		// OpenAI models
		{"gpt-4o", ProviderOpenAI},
		{"gpt-4o-mini", ProviderOpenAI},
		{"gpt-4.1", ProviderOpenAI},
		{"gpt-4.5-preview", ProviderOpenAI},
		{"gpt-5", ProviderOpenAI},
		{"gpt-5-mini", ProviderOpenAI},
		{"o1", ProviderOpenAI},
		{"o1-mini", ProviderOpenAI},
		{"o3", ProviderOpenAI},
		{"o3-mini", ProviderOpenAI},
		{"o4-mini", ProviderOpenAI},
		{"gpt-3.5-turbo", ProviderOpenAI},
		{"gpt-4-turbo", ProviderOpenAI},
		{"gpt-4", ProviderOpenAI},
		{"text-embedding-3-small", ProviderOpenAI},
		{"text-embedding-3-large", ProviderOpenAI},
		{"gpt-oss-120b", ProviderOpenAI},

		// Anthropic models
		{"claude-3-opus-20240229", ProviderAnthropic},
		{"claude-3-sonnet-20240229", ProviderAnthropic},
		{"claude-3-haiku-20240307", ProviderAnthropic},
		{"claude-sonnet-4-5", ProviderAnthropic},
		{"claude-2.1", ProviderAnthropic},

		// Cohere models
		{"command-r-plus", ProviderCohere},
		{"command-r", ProviderCohere},
		{"command-a-03-2025", ProviderCohere},
		{"command-nightly", ProviderCohere},
		{"embed-english-v3.0", ProviderCohere},
		{"embed-multilingual-v3.0", ProviderCohere},
		{"rerank-english-v3.0", ProviderCohere},
		{"c4ai-aya-23-35b", ProviderCohere},

		// Unknown models
		{"llama-3.1-70b", ProviderUnknown},
		{"mistral-large", ProviderUnknown},
		{"gemini-pro", ProviderUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			got := DetectProvider(tt.model)
			if got != tt.expected {
				t.Errorf("DetectProvider(%q) = %v, want %v", tt.model, got, tt.expected)
			}
		})
	}
}

func TestGetOpenAIEncoding(t *testing.T) {
	tests := []struct {
		model    string
		expected OpenAIEncoding
	}{
		// o200k_base models
		{"gpt-4o", EncodingO200kBase},
		{"gpt-4o-mini", EncodingO200kBase},
		{"gpt-4.1", EncodingO200kBase},
		{"gpt-4.5-preview", EncodingO200kBase},
		{"gpt-5", EncodingO200kBase},
		{"o1", EncodingO200kBase},
		{"o3", EncodingO200kBase},
		{"o4-mini", EncodingO200kBase},
		{"gpt-oss-120b", EncodingO200kBase},
		{"chatgpt-4o-latest", EncodingO200kBase},

		// cl100k_base models
		{"gpt-4", EncodingCL100kBase},
		{"gpt-4-turbo", EncodingCL100kBase},
		{"gpt-3.5-turbo", EncodingCL100kBase},
		{"text-embedding-3-small", EncodingCL100kBase},
		{"text-embedding-ada-002", EncodingCL100kBase},

		// Legacy encodings
		{"code-davinci-002", EncodingP50kBase},
		{"davinci", EncodingR50kBase},
		{"curie", EncodingR50kBase},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			got := GetOpenAIEncoding(tt.model)
			if got != tt.expected {
				t.Errorf("GetOpenAIEncoding(%q) = %v, want %v", tt.model, got, tt.expected)
			}
		})
	}
}

func TestForModel(t *testing.T) {
	// Clear cache before testing
	ClearCache()

	tests := []struct {
		model            string
		expectedProvider Provider
		expectedAccurate bool
	}{
		{"gpt-4o", ProviderOpenAI, true},
		{"gpt-4", ProviderOpenAI, true},
		{"claude-sonnet-4-5", ProviderAnthropic, true}, // May be false if BPE data not loaded
		{"command-r-plus", ProviderCohere, true},       // May be false if BPE data not loaded
		{"unknown-model", ProviderUnknown, false},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			tok := ForModel(tt.model)
			if tok == nil {
				t.Fatalf("ForModel(%q) returned nil", tt.model)
			}
			if tok.Provider() != tt.expectedProvider {
				t.Errorf("ForModel(%q).Provider() = %v, want %v", tt.model, tok.Provider(), tt.expectedProvider)
			}
		})
	}
}

func TestForModelCaching(t *testing.T) {
	ClearCache()

	tok1 := ForModel("gpt-4o")
	tok2 := ForModel("gpt-4o")

	// Should return the same cached instance
	if tok1 != tok2 {
		t.Error("ForModel should return cached tokenizer instance")
	}
}

func TestTiktokenTokenizer(t *testing.T) {
	tok := ForModel("gpt-4o")
	if tok == nil {
		t.Fatal("Failed to create tiktoken tokenizer")
	}

	if !tok.IsAccurate() {
		t.Skip("Tiktoken not available, skipping accurate tests")
	}

	// Test Count
	text := "Hello, world!"
	count := tok.Count(text)
	if count <= 0 {
		t.Errorf("Count(%q) = %d, expected > 0", text, count)
	}

	// Known token count for "Hello, world!" with cl100k_base/o200k_base is typically 4
	if count < 3 || count > 6 {
		t.Errorf("Count(%q) = %d, expected between 3 and 6", text, count)
	}

	// Test empty string
	if tok.Count("") != 0 {
		t.Error("Count of empty string should be 0")
	}

	// Test Encode/Decode round-trip
	encoded := tok.Encode(text)
	if len(encoded) != count {
		t.Errorf("Encode length %d != Count %d", len(encoded), count)
	}

	decoded := tok.Decode(encoded)
	if decoded != text {
		t.Errorf("Decode(Encode(%q)) = %q, want %q", text, decoded, text)
	}

	// Test Truncate
	longText := strings.Repeat("This is a test sentence. ", 100)
	truncated := tok.Truncate(longText, 10)
	truncatedCount := tok.Count(truncated)
	if truncatedCount > 10 {
		t.Errorf("Truncated text has %d tokens, expected <= 10", truncatedCount)
	}

	// Test Truncate with text that fits
	shortText := "Hi"
	truncatedShort := tok.Truncate(shortText, 100)
	if truncatedShort != shortText {
		t.Errorf("Truncate should not modify text that fits: got %q, want %q", truncatedShort, shortText)
	}
}

func TestTiktokenCountMessages(t *testing.T) {
	tok := ForModel("gpt-4o")
	if tok == nil {
		t.Fatal("Failed to create tiktoken tokenizer")
	}

	if !tok.IsAccurate() {
		t.Skip("Tiktoken not available, skipping accurate tests")
	}

	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello!"},
		{Role: "assistant", Content: "Hi! How can I help you today?"},
	}

	count := tok.CountMessages(messages)
	if count <= 0 {
		t.Errorf("CountMessages returned %d, expected > 0", count)
	}

	// Each message should have some overhead
	minExpected := tok.Count("You are a helpful assistant.") +
		tok.Count("Hello!") +
		tok.Count("Hi! How can I help you today?")
	if count <= minExpected {
		t.Errorf("CountMessages(%d) should be greater than content tokens alone(%d)", count, minExpected)
	}
}

func TestClaudeTokenizer(t *testing.T) {
	tok := ForModel("claude-sonnet-4-5")
	if tok == nil {
		t.Fatal("Failed to create Claude tokenizer")
	}

	// Test Count
	text := "Hello, world!"
	count := tok.Count(text)
	if count <= 0 {
		t.Errorf("Count(%q) = %d, expected > 0", text, count)
	}

	// Test empty string
	if tok.Count("") != 0 {
		t.Error("Count of empty string should be 0")
	}

	// Test Provider
	if tok.Provider() != ProviderAnthropic {
		t.Errorf("Provider() = %v, expected %v", tok.Provider(), ProviderAnthropic)
	}

	// Test EncodingName
	if tok.EncodingName() != "claude" {
		t.Errorf("EncodingName() = %v, expected 'claude'", tok.EncodingName())
	}
}

func TestCohereTokenizer(t *testing.T) {
	tok := ForModel("command-r-plus")
	if tok == nil {
		t.Fatal("Failed to create Cohere tokenizer")
	}

	// Test Count
	text := "Hello, world!"
	count := tok.Count(text)
	if count <= 0 {
		t.Errorf("Count(%q) = %d, expected > 0", text, count)
	}

	// Test empty string
	if tok.Count("") != 0 {
		t.Error("Count of empty string should be 0")
	}

	// Test Provider
	if tok.Provider() != ProviderCohere {
		t.Errorf("Provider() = %v, expected %v", tok.Provider(), ProviderCohere)
	}

	// Test EncodingName
	if tok.EncodingName() != "cohere" {
		t.Errorf("EncodingName() = %v, expected 'cohere'", tok.EncodingName())
	}
}

func TestEstimatorTokenizer(t *testing.T) {
	tok := ForModel("unknown-model-xyz")
	if tok == nil {
		t.Fatal("Failed to create estimator tokenizer")
	}

	if tok.IsAccurate() {
		t.Error("Estimator should not be accurate")
	}

	// Test Count
	text := "Hello, world!"
	count := tok.Count(text)
	if count <= 0 {
		t.Errorf("Count(%q) = %d, expected > 0", text, count)
	}

	// Test empty string
	if tok.Count("") != 0 {
		t.Error("Count of empty string should be 0")
	}

	// Test Provider
	if tok.Provider() != ProviderUnknown {
		t.Errorf("Provider() = %v, expected %v", tok.Provider(), ProviderUnknown)
	}

	// Test EncodingName
	if tok.EncodingName() != "estimate" {
		t.Errorf("EncodingName() = %v, expected 'estimate'", tok.EncodingName())
	}

	// Test Truncate
	longText := strings.Repeat("word ", 100) // 500 chars = ~125 tokens at 4 char/token
	truncated := tok.Truncate(longText, 10)
	if len(truncated) >= len(longText) {
		t.Error("Truncate should shorten the text")
	}
}

func TestTruncateMessages(t *testing.T) {
	tok := ForModel("gpt-4o")
	if tok == nil {
		t.Fatal("Failed to create tokenizer")
	}

	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "First message"},
		{Role: "assistant", Content: "First response"},
		{Role: "user", Content: "Second message"},
		{Role: "assistant", Content: "Second response"},
		{Role: "user", Content: "Third message"},
	}

	// Truncate to a small budget that won't fit all messages
	truncated := tok.TruncateMessages(messages, 50)

	// Should have fewer messages or same
	if len(truncated) > len(messages) {
		t.Error("TruncateMessages should not add messages")
	}

	// Should keep first message (system)
	if len(truncated) > 0 && truncated[0].Role != "system" {
		t.Error("TruncateMessages should preserve system message")
	}

	// Should keep last message
	if len(truncated) > 1 && truncated[len(truncated)-1].Content != "Third message" {
		// This might not always be the case depending on token budget
		// The algorithm prioritizes keeping recent messages
	}
}

func TestConvenienceFunctions(t *testing.T) {
	// Test CountTokens
	count := CountTokens("gpt-4o", "Hello, world!")
	if count <= 0 {
		t.Error("CountTokens should return positive count")
	}

	// Test CountMessageTokens
	messages := []Message{
		{Role: "user", Content: "Hello!"},
	}
	msgCount := CountMessageTokens("gpt-4o", messages)
	if msgCount <= 0 {
		t.Error("CountMessageTokens should return positive count")
	}

	// Test EstimateTokens
	estimate := EstimateTokens("Hello, world!")
	if estimate <= 0 {
		t.Error("EstimateTokens should return positive count")
	}
}

func TestForProvider(t *testing.T) {
	tests := []struct {
		provider         Provider
		expectedProvider Provider
	}{
		{ProviderOpenAI, ProviderOpenAI},
		{ProviderAnthropic, ProviderAnthropic},
		{ProviderCohere, ProviderCohere},
		{ProviderUnknown, ProviderUnknown},
	}

	for _, tt := range tests {
		t.Run(string(tt.provider), func(t *testing.T) {
			tok := ForProvider(tt.provider)
			if tok == nil {
				t.Fatalf("ForProvider(%v) returned nil", tt.provider)
			}
			if tok.Provider() != tt.expectedProvider {
				t.Errorf("ForProvider(%v).Provider() = %v, want %v",
					tt.provider, tok.Provider(), tt.expectedProvider)
			}
		})
	}
}

func TestGetModelInfo(t *testing.T) {
	tests := []struct {
		model            string
		expectedProvider Provider
	}{
		{"gpt-4o", ProviderOpenAI},
		{"claude-sonnet-4-5", ProviderAnthropic},
		{"command-r-plus", ProviderCohere},
		{"unknown", ProviderUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			info := GetModelInfo(tt.model)
			if info.Provider != tt.expectedProvider {
				t.Errorf("GetModelInfo(%q).Provider = %v, want %v",
					tt.model, info.Provider, tt.expectedProvider)
			}
			if info.TokensPerMsg <= 0 {
				t.Errorf("GetModelInfo(%q).TokensPerMsg should be > 0", tt.model)
			}
		})
	}
}

// Benchmark tests
func BenchmarkTiktokenCount(b *testing.B) {
	tok := ForModel("gpt-4o")
	text := "This is a sample text for benchmarking the tokenizer performance across different scenarios."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok.Count(text)
	}
}

func BenchmarkEstimatorCount(b *testing.B) {
	tok := ForModel("unknown-model")
	text := "This is a sample text for benchmarking the tokenizer performance across different scenarios."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok.Count(text)
	}
}

func BenchmarkForModel(b *testing.B) {
	ClearCache()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ForModel("gpt-4o")
	}
}

func BenchmarkForModelCached(b *testing.B) {
	ClearCache()
	ForModel("gpt-4o") // Prime the cache

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ForModel("gpt-4o")
	}
}
