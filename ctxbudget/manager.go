package ctxbudget

import (
	"fmt"
	"sort"
	"strings"

	"github.com/zenousai/goragtoolkit/catalog"
)

// Manager handles token-aware context assembly
type Manager struct {
	counter TokenCounter
	modelID string
	config  Config
}

// NewManager creates a context manager for the given model
func NewManager(modelID string, counter TokenCounter, config Config) *Manager {
	// Apply defaults for zero values
	if config.Strategy == "" {
		config.Strategy = StrategySourceFirst
	}
	if config.SourceTruncation == "" {
		config.SourceTruncation = SourceTruncateProportional
	}
	if config.SafetyMargin == 0 {
		config.SafetyMargin = 100
	}
	if config.MinHistoryMessages == 0 {
		config.MinHistoryMessages = 2
	}

	return &Manager{
		counter: counter,
		modelID: modelID,
		config:  config,
	}
}

// NewManagerWithDefaults creates a manager with default configuration
func NewManagerWithDefaults(modelID string) *Manager {
	return NewManager(modelID, NewTokenCounter(modelID), DefaultConfig())
}

// Assemble builds context within token budget using the configured strategy.
// It returns the assembled messages along with metadata about token usage and truncations.
func (m *Manager) Assemble(input AssembleInput) (*AssembleResult, error) {
	result := &AssembleResult{
		Truncations: []TruncationEvent{},
		Warnings:    []string{},
	}

	// Step 1: Measure inputs for metrics tracking
	result.InputSourceCount = len(input.Sources)
	result.InputHistoryCount = len(input.History)
	result.InputSourceTokens = m.measureTotalSourceTokens(input.Sources)
	result.InputHistoryTokens = m.countHistoryTokens(input.History)

	// Step 2: Determine total available budget
	maxContext := m.getMaxContextTokens()
	responseReserve := m.getResponseReserve()
	available := maxContext - responseReserve - m.config.SafetyMargin

	if available <= 0 {
		return nil, fmt.Errorf("no budget available: maxContext=%d, responseReserve=%d, safety=%d",
			maxContext, responseReserve, m.config.SafetyMargin)
	}

	result.ResponseBudget = responseReserve

	// Step 3: Measure immutables (system prompt + user query - never truncated)
	systemTokens := m.counter.Count(input.SystemPrompt)
	queryTokens := m.counter.Count(input.UserQuery)
	immutableTokens := systemTokens + queryTokens + 8 // +8 for message overhead (2 messages * 4)

	if immutableTokens >= available {
		return nil, fmt.Errorf("immutable content exceeds budget: immutables=%d, available=%d",
			immutableTokens, available)
	}

	flexibleBudget := available - immutableTokens

	// Step 4: Allocate flexible budget based on strategy
	// Note: The returned budget values are estimates used during allocation.
	// Actual token counts are computed from the allocated content later.
	var allocatedSources []Source
	var allocatedHistory []Message

	switch m.config.Strategy {
	case StrategySourceFirst:
		allocatedSources, _, allocatedHistory, _ =
			m.allocateSourceFirst(input.Sources, input.History, flexibleBudget, result)

	case StrategyHistoryFirst:
		allocatedHistory, _, allocatedSources, _ =
			m.allocateHistoryFirst(input.History, input.Sources, flexibleBudget, result)

	case StrategyBalanced:
		allocatedSources, _, allocatedHistory, _ =
			m.allocateBalanced(input.Sources, input.History, flexibleBudget, result)

	default:
		// Fall back to source-first
		allocatedSources, _, allocatedHistory, _ =
			m.allocateSourceFirst(input.Sources, input.History, flexibleBudget, result)
	}

	// Step 5: Build final message list
	messages := m.buildMessages(input.SystemPrompt, allocatedSources, allocatedHistory, input.UserQuery)
	result.Messages = messages

	// Step 6: Record output counts for metrics - always use ACTUAL token counts
	result.OutputSourceCount = len(allocatedSources)
	result.OutputSourceTokens = m.measureTotalSourceTokens(allocatedSources)
	result.OutputHistoryCount = len(allocatedHistory)
	result.OutputHistoryTokens = m.countHistoryTokens(allocatedHistory)

	// Step 7: Calculate final token count by measuring the actual assembled messages
	// This accounts for ALL formatting overhead including source formatting in system message
	result.TotalTokens = m.countMessagesTokens(messages)
	result.BudgetUsed = float64(result.TotalTokens) / float64(available)

	return result, nil
}

// measureTotalSourceTokens calculates total tokens for all sources
func (m *Manager) measureTotalSourceTokens(sources []Source) int {
	total := 0
	for _, s := range sources {
		tokens := m.counter.Count(s.Content)
		if s.Title != "" {
			tokens += m.counter.Count(s.Title) + 2
		}
		tokens += 10 // overhead
		total += tokens
	}
	return total
}

// getMaxContextTokens returns the context window size for the model
func (m *Manager) getMaxContextTokens() int {
	if m.config.MaxContextTokens > 0 {
		return m.config.MaxContextTokens
	}

	// Use catalog as single source of truth for model context windows
	contextWindow := catalog.GetDefaultMaxTokens(m.modelID)
	if contextWindow > 0 {
		return contextWindow
	}

	return DefaultContextWindow
}

// getResponseReserve returns tokens to reserve for response
func (m *Manager) getResponseReserve() int {
	if m.config.ResponseReserve > 0 {
		return m.config.ResponseReserve
	}
	return DefaultResponseReserve
}

// allocateSourceFirst allocates budget prioritizing sources over history
func (m *Manager) allocateSourceFirst(
	sources []Source,
	history []Message,
	flexibleBudget int,
	result *AssembleResult,
) ([]Source, int, []Message, int) {
	// Calculate total source tokens
	sourcesWithTokens := m.measureSources(sources)
	totalSourceTokens := 0
	for _, s := range sourcesWithTokens {
		totalSourceTokens += s.tokens
	}

	var allocatedSources []Source
	var sourceBudget int

	if totalSourceTokens <= flexibleBudget {
		// All sources fit
		allocatedSources = sources
		sourceBudget = totalSourceTokens
	} else {
		// Need to truncate sources
		allocatedSources, sourceBudget = m.truncateSources(sourcesWithTokens, flexibleBudget, result)
	}

	// Allocate remaining to history
	historyBudget := flexibleBudget - sourceBudget
	allocatedHistory := m.truncateHistory(history, historyBudget, result)

	actualHistoryTokens := m.countHistoryTokens(allocatedHistory)

	return allocatedSources, sourceBudget, allocatedHistory, actualHistoryTokens
}

// allocateHistoryFirst allocates budget prioritizing history over sources
func (m *Manager) allocateHistoryFirst(
	history []Message,
	sources []Source,
	flexibleBudget int,
	result *AssembleResult,
) ([]Message, int, []Source, int) {
	// Calculate total history tokens
	totalHistoryTokens := m.countHistoryTokens(history)

	var allocatedHistory []Message
	var historyBudget int

	if totalHistoryTokens <= flexibleBudget {
		// All history fits
		allocatedHistory = history
		historyBudget = totalHistoryTokens
	} else {
		// Need to truncate history
		allocatedHistory = m.truncateHistory(history, flexibleBudget, result)
		historyBudget = m.countHistoryTokens(allocatedHistory)
	}

	// Allocate remaining to sources
	sourceBudget := flexibleBudget - historyBudget
	sourcesWithTokens := m.measureSources(sources)
	allocatedSources, actualSourceTokens := m.truncateSources(sourcesWithTokens, sourceBudget, result)

	return allocatedHistory, historyBudget, allocatedSources, actualSourceTokens
}

// allocateBalanced splits budget evenly between sources and history
func (m *Manager) allocateBalanced(
	sources []Source,
	history []Message,
	flexibleBudget int,
	result *AssembleResult,
) ([]Source, int, []Message, int) {
	halfBudget := flexibleBudget / 2

	// Allocate sources
	sourcesWithTokens := m.measureSources(sources)
	allocatedSources, sourceBudget := m.truncateSources(sourcesWithTokens, halfBudget, result)

	// Allocate history - can use leftover from sources
	historyBudget := flexibleBudget - sourceBudget
	allocatedHistory := m.truncateHistory(history, historyBudget, result)
	actualHistoryTokens := m.countHistoryTokens(allocatedHistory)

	return allocatedSources, sourceBudget, allocatedHistory, actualHistoryTokens
}

// sourceWithTokens pairs a source with its token count
type sourceWithTokens struct {
	source Source
	tokens int
}

// measureSources calculates token counts for all sources
func (m *Manager) measureSources(sources []Source) []sourceWithTokens {
	result := make([]sourceWithTokens, len(sources))
	for i, s := range sources {
		// Count content + title + formatting overhead
		tokens := m.counter.Count(s.Content)
		if s.Title != "" {
			tokens += m.counter.Count(s.Title) + 2 // title + newlines
		}
		tokens += 10 // overhead for [SU-xxx] marker and formatting
		result[i] = sourceWithTokens{source: s, tokens: tokens}
	}
	return result
}

// truncateSources reduces sources to fit within budget using configured strategy
func (m *Manager) truncateSources(
	sources []sourceWithTokens,
	budget int,
	result *AssembleResult,
) ([]Source, int) {
	if len(sources) == 0 {
		return nil, 0
	}

	// Calculate total
	total := 0
	for _, s := range sources {
		total += s.tokens
	}

	if total <= budget {
		// All sources fit
		out := make([]Source, len(sources))
		for i, s := range sources {
			out[i] = s.source
		}
		return out, total
	}

	// Need to truncate based on strategy
	switch m.config.SourceTruncation {
	case SourceTruncateDropLowest:
		return m.dropLowestSources(sources, budget, result)
	case SourceTruncateProportional:
		return m.proportionalTruncateSources(sources, budget, result)
	case SourceTruncateDiversity:
		return m.diversityTruncateSources(sources, budget, result)
	default:
		return m.proportionalTruncateSources(sources, budget, result)
	}
}

// dropLowestSources drops entire sources starting from lowest score
func (m *Manager) dropLowestSources(
	sources []sourceWithTokens,
	budget int,
	result *AssembleResult,
) ([]Source, int) {
	// Sort by effective score descending (highest first)
	sorted := make([]sourceWithTokens, len(sources))
	copy(sorted, sources)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].source.EffectiveScore() > sorted[j].source.EffectiveScore()
	})

	var kept []Source
	usedTokens := 0

	for _, s := range sorted {
		if usedTokens+s.tokens <= budget {
			kept = append(kept, s.source)
			usedTokens += s.tokens
		} else {
			// Record truncation
			result.Truncations = append(result.Truncations, TruncationEvent{
				Component:    "source",
				ItemID:       s.source.ID,
				TokensBefore: s.tokens,
				TokensAfter:  0,
				Reason:       "dropped: over budget",
			})
		}
	}

	if len(kept) < len(sources) {
		result.Warnings = append(result.Warnings,
			fmt.Sprintf("%d sources dropped due to token budget", len(sources)-len(kept)))
	}

	return kept, usedTokens
}

// proportionalTruncateSources truncates all sources proportionally by score
func (m *Manager) proportionalTruncateSources(
	sources []sourceWithTokens,
	budget int,
	result *AssembleResult,
) ([]Source, int) {
	// Calculate total and effective score sum
	total := 0
	scoreSum := 0.0
	for _, s := range sources {
		total += s.tokens
		scoreSum += s.source.EffectiveScore()
	}

	if scoreSum == 0 {
		// Equal weights if no scores
		scoreSum = float64(len(sources))
	}

	// Calculate compression ratio needed
	ratio := float64(budget) / float64(total)

	var kept []Source
	usedTokens := 0
	truncatedCount := 0

	for _, s := range sources {
		// Higher-scored sources get proportionally more of their content
		scoreWeight := s.source.EffectiveScore() / scoreSum * float64(len(sources))
		adjustedRatio := ratio * (0.5 + 0.5*scoreWeight) // Scale between 50-150% of base ratio

		targetTokens := max(int(float64(s.tokens)*adjustedRatio),
			// Minimum tokens to keep
			50)
		if targetTokens > s.tokens {
			targetTokens = s.tokens
		}

		if usedTokens+targetTokens > budget {
			targetTokens = budget - usedTokens
		}

		if targetTokens <= 0 {
			result.Truncations = append(result.Truncations, TruncationEvent{
				Component:    "source",
				ItemID:       s.source.ID,
				TokensBefore: s.tokens,
				TokensAfter:  0,
				Reason:       "dropped: no budget remaining",
			})
			continue
		}

		// Truncate content if needed
		if targetTokens < s.tokens {
			truncatedContent := m.truncateContent(s.source.Content, targetTokens-10) // Reserve for overhead
			kept = append(kept, Source{
				ID:         s.source.ID,
				Content:    truncatedContent,
				Title:      s.source.Title,
				Score:      s.source.Score,
				DocumentID: s.source.DocumentID,
				Weight:     s.source.Weight,
			})
			result.Truncations = append(result.Truncations, TruncationEvent{
				Component:    "source",
				ItemID:       s.source.ID,
				TokensBefore: s.tokens,
				TokensAfter:  targetTokens,
				Reason:       "proportionally reduced",
			})
			truncatedCount++
		} else {
			kept = append(kept, s.source)
		}

		usedTokens += targetTokens
	}

	if truncatedCount > 0 {
		result.Warnings = append(result.Warnings,
			fmt.Sprintf("%d sources truncated to fit token budget", truncatedCount))
	}

	return kept, usedTokens
}

// diversityTruncateSources keeps sources from different documents
func (m *Manager) diversityTruncateSources(
	sources []sourceWithTokens,
	budget int,
	result *AssembleResult,
) ([]Source, int) {
	// Group by document
	byDoc := make(map[string][]sourceWithTokens)
	noDoc := []sourceWithTokens{}

	for _, s := range sources {
		if s.source.DocumentID != "" {
			byDoc[s.source.DocumentID] = append(byDoc[s.source.DocumentID], s)
		} else {
			noDoc = append(noDoc, s)
		}
	}

	// Sort each group by effective score
	for _, group := range byDoc {
		sort.Slice(group, func(i, j int) bool {
			return group[i].source.EffectiveScore() > group[j].source.EffectiveScore()
		})
	}
	sort.Slice(noDoc, func(i, j int) bool {
		return noDoc[i].source.EffectiveScore() > noDoc[j].source.EffectiveScore()
	})

	// Round-robin selection: take highest from each doc, then repeat
	var kept []Source
	usedTokens := 0
	docIDs := make([]string, 0, len(byDoc))
	for docID := range byDoc {
		docIDs = append(docIDs, docID)
	}
	sort.Strings(docIDs) // Deterministic order

	// Index tracking for each document group
	indices := make(map[string]int)

	for {
		added := false

		// Try to add one from each document
		for _, docID := range docIDs {
			group := byDoc[docID]
			idx := indices[docID]
			if idx >= len(group) {
				continue
			}
			s := group[idx]
			if usedTokens+s.tokens <= budget {
				kept = append(kept, s.source)
				usedTokens += s.tokens
				indices[docID] = idx + 1
				added = true
			}
		}

		// Add from sources without document
		if len(noDoc) > 0 {
			s := noDoc[0]
			if usedTokens+s.tokens <= budget {
				kept = append(kept, s.source)
				usedTokens += s.tokens
				noDoc = noDoc[1:]
				added = true
			}
		}

		if !added {
			break
		}
	}

	droppedCount := len(sources) - len(kept)
	if droppedCount > 0 {
		result.Warnings = append(result.Warnings,
			fmt.Sprintf("%d sources dropped for diversity (keeping sources from different documents)", droppedCount))
	}

	return kept, usedTokens
}

// truncateHistory reduces history to fit within budget, keeping most recent
func (m *Manager) truncateHistory(
	history []Message,
	budget int,
	result *AssembleResult,
) []Message {
	if len(history) == 0 {
		return nil
	}

	// Measure all messages
	type msgWithTokens struct {
		msg    Message
		tokens int
	}

	measured := make([]msgWithTokens, len(history))
	total := 0
	for i, msg := range history {
		tokens := m.counter.Count(msg.Content) + 4 // +4 for role overhead
		measured[i] = msgWithTokens{msg: msg, tokens: tokens}
		total += tokens
	}

	if total <= budget {
		return history
	}

	// Keep most recent messages that fit
	// Start from the end and work backwards
	var kept []Message
	usedTokens := 0
	keptCount := 0

	for i := len(measured) - 1; i >= 0; i-- {
		item := measured[i]
		if usedTokens+item.tokens <= budget {
			kept = append([]Message{item.msg}, kept...) // Prepend to maintain order
			usedTokens += item.tokens
			keptCount++
		} else if keptCount < m.config.MinHistoryMessages && item.tokens <= budget-usedTokens+100 {
			// Try to keep minimum messages by truncating this one
			availableForThis := budget - usedTokens
			if availableForThis > 50 {
				truncatedContent := m.truncateMessage(item.msg.Content, availableForThis-4)
				// Recompute actual tokens after truncation to avoid overcounting
				actualTruncatedTokens := m.counter.Count(truncatedContent) + 4
				kept = append([]Message{{Role: item.msg.Role, Content: truncatedContent}}, kept...)
				usedTokens += actualTruncatedTokens
				keptCount++

				result.Truncations = append(result.Truncations, TruncationEvent{
					Component:    "message",
					TokensBefore: item.tokens,
					TokensAfter:  actualTruncatedTokens,
					Reason:       "truncated to preserve minimum history",
				})
			}
		}
	}

	droppedCount := len(history) - len(kept)
	if droppedCount > 0 {
		result.Truncations = append(result.Truncations, TruncationEvent{
			Component:    "history",
			TokensBefore: total,
			TokensAfter:  usedTokens,
			Reason:       fmt.Sprintf("%d messages dropped", droppedCount),
		})
		result.Warnings = append(result.Warnings,
			fmt.Sprintf("%d history messages dropped to fit token budget", droppedCount))
	}

	return kept
}

// countHistoryTokens calculates total tokens for history messages
func (m *Manager) countHistoryTokens(history []Message) int {
	total := 0
	for _, msg := range history {
		total += m.counter.Count(msg.Content) + 4
	}
	return total
}

// countMessagesTokens calculates total tokens for a list of messages.
// This accounts for all content including formatting and role overhead.
func (m *Manager) countMessagesTokens(messages []Message) int {
	total := 0
	for _, msg := range messages {
		total += m.counter.Count(msg.Content) + 4 // +4 for role overhead per message
	}
	return total
}

// truncateContent truncates text content to fit within token limit
// Uses head+tail strategy per design doc: 40% head, 40% tail
func (m *Manager) truncateContent(content string, maxTokens int) string {
	currentTokens := m.counter.Count(content)
	if currentTokens <= maxTokens {
		return content
	}

	// Calculate target lengths
	headTokens := maxTokens * 40 / 100
	tailTokens := maxTokens * 40 / 100
	// Middle 20% is for the truncation marker

	// Estimate character positions (rough: 4 chars per token)
	headChars := headTokens * 4
	tailChars := tailTokens * 4

	if headChars >= len(content) {
		headChars = len(content) / 2
	}
	if tailChars >= len(content) {
		tailChars = len(content) / 2
	}

	head := content[:headChars]
	tail := content[len(content)-tailChars:]

	return head + "\n\n[... content truncated for length ...]\n\n" + tail
}

// truncateMessage truncates a message using head+tail strategy
func (m *Manager) truncateMessage(content string, maxTokens int) string {
	return m.truncateContent(content, maxTokens)
}

// buildMessages assembles the final message list
func (m *Manager) buildMessages(
	systemPrompt string,
	sources []Source,
	history []Message,
	userQuery string,
) []Message {
	var messages []Message

	// System message includes sources
	var systemContent strings.Builder
	systemContent.WriteString(systemPrompt)
	if len(sources) > 0 {
		systemContent.WriteString("\n\n=== REFERENCE MATERIAL ===\n")
		for _, s := range sources {
			idShort := s.ID
			if len(idShort) > 8 {
				idShort = idShort[:8]
			}
			systemContent.WriteString(fmt.Sprintf("\n[SU-%s]", idShort))
			if s.Title != "" {
				systemContent.WriteString(fmt.Sprintf(" %s", s.Title))
			}
			systemContent.WriteString(fmt.Sprintf("\n%s\n", s.Content))
		}
	}

	messages = append(messages, Message{
		Role:    "system",
		Content: systemContent.String(),
	})

	// Add history
	messages = append(messages, history...)

	// Add user query
	messages = append(messages, Message{
		Role:    "user",
		Content: userQuery,
	})

	return messages
}

// ReserveForShape calculates response token reservation based on answer shape.
// This provides shape-aware defaults when ResponseReserve is not explicitly set.
func ReserveForShape(shape ShapeType, itemCount int) int {
	base := map[ShapeType]int{
		ShapeFactual:      500,
		ShapeEnumerative:  1000,
		ShapeExhaustive:   2000,
		ShapeExploratory:  3000,
		ShapeProcedural:   2000,
		ShapeComparative:  2500,
		ShapeHierarchical: 3000,
	}[shape]

	if base == 0 {
		base = DefaultResponseReserve
	}

	// Add per-item budget for list-type responses
	if shape == ShapeEnumerative || shape == ShapeExhaustive {
		base += itemCount * 300 // ~300 tokens per list item
	}

	// Safety margin
	return base + 500
}

// SetResponseReserveFromShape updates the config with shape-aware response reservation
func (m *Manager) SetResponseReserveFromShape(shape ShapeType, itemCount int) {
	m.config.ResponseReserve = ReserveForShape(shape, itemCount)
}

// GetConfig returns the current configuration
func (m *Manager) GetConfig() Config {
	return m.config
}

// GetModelID returns the model ID
func (m *Manager) GetModelID() string {
	return m.modelID
}

// Helper to check if string contains another (case-insensitive)
func containsIgnoreCase(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}
