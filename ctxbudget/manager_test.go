package ctxbudget

import (
	"strings"
	"testing"
)

// TestDefaultConfig tests that default configuration has sensible values
func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Strategy != StrategySourceFirst {
		t.Errorf("expected Strategy=%s, got %s", StrategySourceFirst, cfg.Strategy)
	}
	if cfg.SourceTruncation != SourceTruncateProportional {
		t.Errorf("expected SourceTruncation=%s, got %s", SourceTruncateProportional, cfg.SourceTruncation)
	}
	if cfg.MaxTokensPerMessage != 4000 {
		t.Errorf("expected MaxTokensPerMessage=4000, got %d", cfg.MaxTokensPerMessage)
	}
	if cfg.MinHistoryMessages != 2 {
		t.Errorf("expected MinHistoryMessages=2, got %d", cfg.MinHistoryMessages)
	}
	if cfg.SafetyMargin != 100 {
		t.Errorf("expected SafetyMargin=100, got %d", cfg.SafetyMargin)
	}
}

// TestNewManager tests manager creation
func TestNewManager(t *testing.T) {
	counter := NewMockCounter(nil, 10)
	mgr := NewManager("gpt-4o", counter, DefaultConfig())

	if mgr.GetModelID() != "gpt-4o" {
		t.Errorf("expected model=gpt-4o, got %s", mgr.GetModelID())
	}
	if mgr.GetConfig().Strategy != StrategySourceFirst {
		t.Errorf("expected Strategy=source_first, got %s", mgr.GetConfig().Strategy)
	}
}

// TestAssemble_WithinBudget tests assembly when everything fits
func TestAssemble_WithinBudget(t *testing.T) {
	// Mock counter: 10 tokens per call
	counter := NewMockCounter(nil, 10)

	cfg := Config{
		MaxContextTokens:    1000,
		ResponseReserve:     200,
		MaxTokensPerMessage: 100,
		Strategy:            StrategySourceFirst,
		SourceTruncation:    SourceTruncateDropLowest,
		MinHistoryMessages:  2,
		SafetyMargin:        50,
	}
	mgr := NewManager("gpt-4o", counter, cfg)

	input := AssembleInput{
		SystemPrompt: "You are helpful.",
		Sources: []Source{
			{ID: "src1", Content: "Source 1 content", Score: 0.9},
			{ID: "src2", Content: "Source 2 content", Score: 0.8},
		},
		History: []Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there!"},
		},
		UserQuery: "What is the answer?",
	}

	result, err := mgr.Assemble(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check that we have messages
	if len(result.Messages) == 0 {
		t.Error("expected messages, got none")
	}

	// Check that no truncation occurred (everything fits)
	if len(result.Truncations) > 0 {
		t.Errorf("expected no truncations, got %d", len(result.Truncations))
	}

	// Check budget usage is reasonable
	if result.BudgetUsed > 1.0 {
		t.Errorf("budget usage should not exceed 1.0, got %f", result.BudgetUsed)
	}
}

// TestAssemble_SourceFirst_TruncatesHistory tests that source-first drops history before sources
func TestAssemble_SourceFirst_TruncatesHistory(t *testing.T) {
	// Use mock counter with specific values to force truncation
	counter := NewMockCounter(map[string]int{
		"System prompt here.":                             20,
		"Important source content that must be included.": 50,
		"Old message 1":                                   20,
		"Old response 1":                                  20,
		"Old message 2":                                   20,
		"Old response 2":                                  20,
		"Recent message":                                  20,
		"Recent response":                                 20,
		"What now?":                                       5,
	}, 20) // Default 20 tokens

	cfg := Config{
		MaxContextTokens:    300, // Very small budget to force truncation
		ResponseReserve:     50,
		MaxTokensPerMessage: 100,
		Strategy:            StrategySourceFirst,
		SourceTruncation:    SourceTruncateDropLowest,
		MinHistoryMessages:  1,
		SafetyMargin:        20,
	}
	mgr := NewManager("gpt-4o", counter, cfg)

	input := AssembleInput{
		SystemPrompt: "System prompt here.",
		Sources: []Source{
			{ID: "src1", Content: "Important source content that must be included.", Score: 0.9},
		},
		History: []Message{
			{Role: "user", Content: "Old message 1"},
			{Role: "assistant", Content: "Old response 1"},
			{Role: "user", Content: "Old message 2"},
			{Role: "assistant", Content: "Old response 2"},
			{Role: "user", Content: "Recent message"},
			{Role: "assistant", Content: "Recent response"},
		},
		UserQuery: "What now?",
	}

	result, err := mgr.Assemble(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// With source-first, sources should be preserved and history reduced
	// Count non-system, non-final-user messages (the history)
	historyCount := 0
	for i, m := range result.Messages {
		// Skip system message (first) and user query (last)
		if i == 0 || i == len(result.Messages)-1 {
			continue
		}
		if m.Role == "user" || m.Role == "assistant" {
			historyCount++
		}
	}

	// Original history was 6 messages, should be reduced
	if historyCount >= 6 {
		t.Errorf("expected history to be truncated (was 6), got %d. Budget constraints should force truncation.", historyCount)
	}
}

// TestAssemble_HistoryFirst_TruncatesSources tests that history-first drops sources before history
func TestAssemble_HistoryFirst_TruncatesSources(t *testing.T) {
	counter := NewEstimatorCounter()

	cfg := Config{
		MaxContextTokens:    600, // Small budget
		ResponseReserve:     100,
		MaxTokensPerMessage: 100,
		Strategy:            StrategyHistoryFirst,
		SourceTruncation:    SourceTruncateDropLowest,
		MinHistoryMessages:  2,
		SafetyMargin:        50,
	}
	mgr := NewManager("gpt-4o", counter, cfg)

	input := AssembleInput{
		SystemPrompt: "System.",
		Sources: []Source{
			{ID: "src1", Content: "Source 1 with lots of content here.", Score: 0.9},
			{ID: "src2", Content: "Source 2 with lots of content here.", Score: 0.8},
			{ID: "src3", Content: "Source 3 with lots of content here.", Score: 0.7},
		},
		History: []Message{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi!"},
		},
		UserQuery: "Question?",
	}

	result, err := mgr.Assemble(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// History should be preserved since we use history-first
	// Check that history messages are present
	hasHello := false
	for _, m := range result.Messages {
		if strings.Contains(m.Content, "Hello") {
			hasHello = true
			break
		}
	}
	if !hasHello {
		t.Error("expected history 'Hello' to be preserved with history-first strategy")
	}
}

// TestAssemble_Balanced tests balanced allocation
func TestAssemble_Balanced(t *testing.T) {
	counter := NewEstimatorCounter()

	cfg := Config{
		MaxContextTokens:    800,
		ResponseReserve:     100,
		MaxTokensPerMessage: 100,
		Strategy:            StrategyBalanced,
		SourceTruncation:    SourceTruncateDropLowest,
		MinHistoryMessages:  2,
		SafetyMargin:        50,
	}
	mgr := NewManager("gpt-4o", counter, cfg)

	input := AssembleInput{
		SystemPrompt: "Sys.",
		Sources: []Source{
			{ID: "src1", Content: "Source content 1.", Score: 0.9},
		},
		History: []Message{
			{Role: "user", Content: "Msg"},
		},
		UserQuery: "Q?",
	}

	result, err := mgr.Assemble(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Just verify it assembles successfully
	if len(result.Messages) == 0 {
		t.Error("expected messages")
	}
}

// TestAssemble_ImmutablesExceedBudget tests error when system prompt + query exceed budget
func TestAssemble_ImmutablesExceedBudget(t *testing.T) {
	counter := NewMockCounter(map[string]int{
		"Very long system prompt": 500,
		"Very long query":         400,
	}, 10)

	cfg := Config{
		MaxContextTokens: 600, // Total available will be ~400 after reserves
		ResponseReserve:  200,
		Strategy:         StrategySourceFirst,
		SafetyMargin:     50,
	}
	mgr := NewManager("gpt-4o", counter, cfg)

	input := AssembleInput{
		SystemPrompt: "Very long system prompt",
		Sources:      nil,
		History:      nil,
		UserQuery:    "Very long query",
	}

	_, err := mgr.Assemble(input)
	if err == nil {
		t.Error("expected error when immutables exceed budget")
	}
}

// TestAssemble_EmptyInputs tests handling of empty sources and history
func TestAssemble_EmptyInputs(t *testing.T) {
	counter := NewMockCounter(nil, 10)

	cfg := DefaultConfig()
	cfg.MaxContextTokens = 1000
	cfg.ResponseReserve = 200
	mgr := NewManager("gpt-4o", counter, cfg)

	input := AssembleInput{
		SystemPrompt: "You are helpful.",
		Sources:      nil,
		History:      nil,
		UserQuery:    "Hello?",
	}

	result, err := mgr.Assemble(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have system message and user query
	if len(result.Messages) != 2 {
		t.Errorf("expected 2 messages (system + user), got %d", len(result.Messages))
	}
}

// TestDropLowestSources tests that low-score sources are dropped first
func TestDropLowestSources(t *testing.T) {
	counter := NewMockCounter(nil, 100) // 100 tokens per source

	cfg := Config{
		MaxContextTokens:    500,
		ResponseReserve:     100,
		MaxTokensPerMessage: 200,
		Strategy:            StrategySourceFirst,
		SourceTruncation:    SourceTruncateDropLowest,
		SafetyMargin:        50,
	}
	mgr := NewManager("gpt-4o", counter, cfg)

	input := AssembleInput{
		SystemPrompt: "S",
		Sources: []Source{
			{ID: "low", Content: "Low score", Score: 0.1},
			{ID: "mid", Content: "Mid score", Score: 0.5},
			{ID: "high", Content: "High score", Score: 0.9},
		},
		History:   nil,
		UserQuery: "Q",
	}

	result, err := mgr.Assemble(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Check that high-scoring source is kept
	systemMsg := result.Messages[0].Content
	if !strings.Contains(systemMsg, "High score") {
		t.Error("expected high-scoring source to be kept")
	}
}

// TestReserveForShape tests shape-aware response reservation
func TestReserveForShape(t *testing.T) {
	tests := []struct {
		shape     ShapeType
		itemCount int
		minTokens int
	}{
		{ShapeFactual, 0, 500},
		{ShapeEnumerative, 5, 1000 + 5*300}, // base + per-item
		{ShapeExhaustive, 10, 2000 + 10*300},
		{ShapeExploratory, 0, 3000},
		{ShapeProcedural, 0, 2000},
	}

	for _, tc := range tests {
		t.Run(string(tc.shape), func(t *testing.T) {
			reserve := ReserveForShape(tc.shape, tc.itemCount)
			// Reserve includes +500 safety margin
			expected := tc.minTokens + 500
			if reserve != expected {
				t.Errorf("ReserveForShape(%s, %d) = %d, want %d",
					tc.shape, tc.itemCount, reserve, expected)
			}
		})
	}
}

// TestSetResponseReserveFromShape tests dynamic reserve setting
func TestSetResponseReserveFromShape(t *testing.T) {
	counter := NewMockCounter(nil, 10)
	mgr := NewManager("gpt-4o", counter, DefaultConfig())

	mgr.SetResponseReserveFromShape(ShapeEnumerative, 10)

	expected := ReserveForShape(ShapeEnumerative, 10)
	if mgr.GetConfig().ResponseReserve != expected {
		t.Errorf("expected ResponseReserve=%d, got %d", expected, mgr.GetConfig().ResponseReserve)
	}
}

// TestTruncateContent tests head+tail truncation
func TestTruncateContent(t *testing.T) {
	counter := NewEstimatorCounter()
	mgr := NewManager("gpt-4o", counter, DefaultConfig())

	// Create a long content
	longContent := strings.Repeat("word ", 500) // ~2500 chars, ~625 tokens

	// Truncate to 100 tokens
	truncated := mgr.truncateContent(longContent, 100)

	// Should contain truncation marker
	if !strings.Contains(truncated, "[... content truncated for length ...]") {
		t.Error("expected truncation marker in truncated content")
	}

	// Should be shorter than original
	if len(truncated) >= len(longContent) {
		t.Error("truncated content should be shorter than original")
	}
}

// TestMockCounter tests the mock counter
func TestMockCounter(t *testing.T) {
	counts := map[string]int{
		"hello": 5,
		"world": 7,
	}
	counter := NewMockCounter(counts, 10)

	if counter.Count("hello") != 5 {
		t.Errorf("expected Count(hello)=5, got %d", counter.Count("hello"))
	}
	if counter.Count("world") != 7 {
		t.Errorf("expected Count(world)=7, got %d", counter.Count("world"))
	}
	if counter.Count("unknown") != 10 {
		t.Errorf("expected Count(unknown)=10 (default), got %d", counter.Count("unknown"))
	}
}

// TestEstimatorCounter tests the estimator counter
func TestEstimatorCounter(t *testing.T) {
	counter := NewEstimatorCounter()

	// ~4 chars per token
	text := "Hello world!" // 12 chars -> ~3 tokens
	count := counter.Count(text)

	if count < 2 || count > 4 {
		t.Errorf("expected Count(%q) to be ~3, got %d", text, count)
	}
}

// TestAssemble_MessagesOrder tests that messages are in correct order
func TestAssemble_MessagesOrder(t *testing.T) {
	counter := NewMockCounter(nil, 10)
	cfg := DefaultConfig()
	cfg.MaxContextTokens = 2000
	cfg.ResponseReserve = 200
	mgr := NewManager("gpt-4o", counter, cfg)

	input := AssembleInput{
		SystemPrompt: "System prompt",
		Sources: []Source{
			{ID: "s1", Content: "Source content", Score: 0.9},
		},
		History: []Message{
			{Role: "user", Content: "First user message"},
			{Role: "assistant", Content: "First response"},
		},
		UserQuery: "Final question",
	}

	result, err := mgr.Assemble(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Order should be: system, history..., user query
	if len(result.Messages) < 4 {
		t.Fatalf("expected at least 4 messages, got %d", len(result.Messages))
	}

	if result.Messages[0].Role != "system" {
		t.Errorf("first message should be system, got %s", result.Messages[0].Role)
	}

	lastMsg := result.Messages[len(result.Messages)-1]
	if lastMsg.Role != "user" || lastMsg.Content != "Final question" {
		t.Errorf("last message should be user query, got role=%s content=%s",
			lastMsg.Role, lastMsg.Content)
	}
}

// TestEffectiveScore tests the EffectiveScore method
func TestEffectiveScore(t *testing.T) {
	tests := []struct {
		name     string
		source   Source
		expected float64
	}{
		{"default weight (zero)", Source{Score: 0.8, Weight: 0}, 0.8},
		{"explicit weight 1.0", Source{Score: 0.8, Weight: 1.0}, 0.8},
		{"engagement boost 1.5x", Source{Score: 0.8, Weight: 1.5}, 1.2},
		{"older engagement 1.2x", Source{Score: 0.5, Weight: 1.2}, 0.6},
		{"negative weight defaults to 1.0", Source{Score: 0.9, Weight: -1.0}, 0.9},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.source.EffectiveScore()
			if got < tc.expected-0.001 || got > tc.expected+0.001 {
				t.Errorf("EffectiveScore() = %f, want %f", got, tc.expected)
			}
		})
	}
}

// TestDropLowestSources_WithWeights tests that engagement-weighted sources are kept
func TestDropLowestSources_WithWeights(t *testing.T) {
	counter := NewMockCounter(nil, 100) // 100 tokens per source

	// Budget: 600 - 100 (reserve) - 50 (safety) = 450 tokens
	// System "S" + user "Q" = 200 tokens → 250 for sources → fits 2 of 3
	cfg := Config{
		MaxContextTokens:    600,
		ResponseReserve:     100,
		MaxTokensPerMessage: 200,
		Strategy:            StrategySourceFirst,
		SourceTruncation:    SourceTruncateDropLowest,
		SafetyMargin:        50,
	}
	mgr := NewManager("gpt-4o", counter, cfg)

	input := AssembleInput{
		SystemPrompt: "S",
		Sources: []Source{
			{ID: "low-score-high-weight", Content: "Engaged source", Score: 0.3, Weight: 1.5}, // effective: 0.45
			{ID: "mid-score", Content: "Normal source", Score: 0.5},                            // effective: 0.5
			{ID: "high-score", Content: "Best source", Score: 0.9},                              // effective: 0.9
		},
		History:   nil,
		UserQuery: "Q",
	}

	result, err := mgr.Assemble(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// The high-score source should definitely be kept
	systemMsg := result.Messages[0].Content
	if !strings.Contains(systemMsg, "Best source") {
		t.Error("expected high-scoring source to be kept")
	}

	// With weight, low-score-high-weight has effective 0.45, mid-score has 0.5
	// Budget allows 2 of 3 sources, so mid-score should be kept over low-score-high-weight
	if len(result.Truncations) == 0 {
		t.Fatal("expected budget to force truncation (budget math may have changed)")
	}
	{
		if !strings.Contains(systemMsg, "Normal source") {
			t.Error("expected mid-score source (effective 0.5) to be kept over low-score-high-weight (effective 0.45)")
		}
		if strings.Contains(systemMsg, "Engaged source") {
			t.Error("expected low-score-high-weight source (effective 0.45) to be dropped")
		}
	}
}

// TestProportionalTruncateSources_WithWeights tests proportional truncation with weights
func TestProportionalTruncateSources_WithWeights(t *testing.T) {
	// Create sources where weight changes which gets more content
	counter := NewEstimatorCounter()

	cfg := Config{
		MaxContextTokens:    400,
		ResponseReserve:     50,
		MaxTokensPerMessage: 200,
		Strategy:            StrategySourceFirst,
		SourceTruncation:    SourceTruncateProportional,
		SafetyMargin:        50,
	}
	mgr := NewManager("gpt-4o", counter, cfg)

	input := AssembleInput{
		SystemPrompt: "S",
		Sources: []Source{
			{ID: "engaged", Content: strings.Repeat("Engaged content. ", 30), Score: 0.5, Weight: 1.5},
			{ID: "normal", Content: strings.Repeat("Normal content. ", 30), Score: 0.5},
		},
		History:   nil,
		UserQuery: "Q",
	}

	result, err := mgr.Assemble(input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Both sources should be present (proportional keeps all, just truncated)
	systemMsg := result.Messages[0].Content
	if !strings.Contains(systemMsg, "Engaged") {
		t.Error("expected engaged source to be present")
	}
	if !strings.Contains(systemMsg, "Normal") {
		t.Error("expected normal source to be present")
	}
}

// BenchmarkAssemble benchmarks context assembly
func BenchmarkAssemble(b *testing.B) {
	counter := NewEstimatorCounter()
	cfg := DefaultConfig()
	cfg.MaxContextTokens = 32000
	cfg.ResponseReserve = 4000
	mgr := NewManager("gpt-4o", counter, cfg)

	// Create realistic input
	sources := make([]Source, 20)
	for i := range sources {
		sources[i] = Source{
			ID:      string(rune('a' + i)),
			Content: strings.Repeat("Content for source. ", 100), // ~2000 chars
			Score:   float64(20-i) / 20.0,
		}
	}

	history := make([]Message, 10)
	for i := range history {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		history[i] = Message{
			Role:    role,
			Content: strings.Repeat("History message content. ", 20),
		}
	}

	input := AssembleInput{
		SystemPrompt: strings.Repeat("System instructions. ", 50),
		Sources:      sources,
		History:      history,
		UserQuery:    "What is the answer to the question?",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := mgr.Assemble(input)
		if err != nil {
			b.Fatalf("unexpected error: %v", err)
		}
	}
}
