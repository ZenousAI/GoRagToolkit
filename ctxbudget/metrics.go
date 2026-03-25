package ctxbudget

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Prometheus metrics for context budget management
var (
	// contextTokensTotal tracks the total tokens in assembled contexts
	contextTokensTotal = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "tokens_total",
			Help:      "Total tokens in assembled context",
			Buckets:   prometheus.ExponentialBuckets(1000, 2, 10), // 1K to 512K
		},
		[]string{"model", "strategy"},
	)

	// contextUtilization tracks context budget utilization ratio
	contextUtilization = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "utilization_ratio",
			Help:      "Context budget utilization (0-1)",
			Buckets:   prometheus.LinearBuckets(0, 0.1, 11), // 0 to 1.0
		},
		[]string{"model", "strategy"},
	)

	// contextTruncationEvents counts truncation events by component
	contextTruncationEvents = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "truncation_events_total",
			Help:      "Number of truncation events",
		},
		[]string{"component", "reason"},
	)

	// contextAssemblyDuration tracks time to assemble context
	contextAssemblyDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "assembly_duration_seconds",
			Help:      "Time to assemble context",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 10), // 1ms to 1s
		},
		[]string{"model"},
	)

	// contextSourcesIncluded tracks how many sources fit in context
	contextSourcesIncluded = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "sources_included",
			Help:      "Number of sources included in context",
			Buckets:   prometheus.LinearBuckets(0, 5, 11), // 0 to 50
		},
		[]string{"strategy"},
	)

	// contextHistoryIncluded tracks how many history messages fit in context
	contextHistoryIncluded = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "history_included",
			Help:      "Number of history messages included in context",
			Buckets:   prometheus.LinearBuckets(0, 5, 11), // 0 to 50
		},
		[]string{"strategy"},
	)

	// === New Phase 1 Metrics for Truncation Observability ===

	// historyMessagesDropped tracks how many history messages were dropped per request
	historyMessagesDropped = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "history_messages_dropped",
			Help:      "Number of history messages dropped due to token budget",
			Buckets:   prometheus.LinearBuckets(0, 2, 16), // 0 to 30
		},
		[]string{"strategy", "shape"},
	)

	// historyTokensDropped tracks total tokens lost from dropped history
	historyTokensDropped = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "history_tokens_dropped",
			Help:      "Total tokens dropped from conversation history",
			Buckets:   prometheus.ExponentialBuckets(100, 2, 12), // 100 to 200K
		},
		[]string{"strategy"},
	)

	// sourcesDropped tracks how many sources were dropped per request
	sourcesDropped = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "sources_dropped",
			Help:      "Number of sources dropped due to token budget",
			Buckets:   prometheus.LinearBuckets(0, 2, 16), // 0 to 30
		},
		[]string{"strategy", "truncation_mode"},
	)

	// sourceTokensDropped tracks total tokens lost from dropped/truncated sources
	sourceTokensDropped = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "source_tokens_dropped",
			Help:      "Total tokens dropped from sources",
			Buckets:   prometheus.ExponentialBuckets(100, 2, 12), // 100 to 200K
		},
		[]string{"strategy", "truncation_mode"},
	)

	// conversationLength tracks the total conversation length (input messages)
	conversationLength = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "conversation_length",
			Help:      "Total number of messages in conversation before truncation",
			Buckets:   prometheus.LinearBuckets(0, 5, 21), // 0 to 100
		},
		[]string{"strategy"},
	)

	// conversationTruncated tracks whether truncation occurred (0 or 1)
	conversationTruncated = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "conversation_truncated_total",
			Help:      "Number of conversations that required truncation",
		},
		[]string{"strategy", "component"}, // component: history, sources, both
	)

	// contextBudgetExhausted tracks when budget is nearly full (>90% utilization)
	contextBudgetExhausted = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "budget_exhausted_total",
			Help:      "Number of times context budget was >90% utilized",
		},
		[]string{"model", "strategy"},
	)

	// longConversationTruncated tracks truncation in long conversations (>10 messages)
	longConversationTruncated = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "goragtoolkit",
			Subsystem: "ctxbudget",
			Name:      "long_conversation_truncated_total",
			Help:      "Number of long conversations (>10 messages) that were truncated",
		},
		[]string{"strategy"},
	)
)

// AssemblyMetrics holds detailed metrics about a context assembly operation
type AssemblyMetrics struct {
	// Input counts
	InputHistoryMessages int
	InputSources         int
	InputHistoryTokens   int
	InputSourceTokens    int

	// Output counts
	OutputHistoryMessages int
	OutputSources         int
	OutputHistoryTokens   int
	OutputSourceTokens    int

	// Shape info
	Shape          string
	TruncationMode string

	// Derived
	HistoryTruncated bool
	SourcesTruncated bool
}

// MetricsRecorder records metrics for context assembly
type MetricsRecorder interface {
	RecordAssembly(result *AssembleResult, modelID string, strategy AllocationStrategy, durationSeconds float64)
	RecordDetailedAssembly(result *AssembleResult, metrics *AssemblyMetrics, modelID string, strategy AllocationStrategy, durationSeconds float64)
}

// prometheusRecorder implements MetricsRecorder using Prometheus
type prometheusRecorder struct{}

// DefaultMetricsRecorder is the default Prometheus-based metrics recorder
var DefaultMetricsRecorder MetricsRecorder = &prometheusRecorder{}

// RecordAssembly records metrics for a context assembly operation
func (p *prometheusRecorder) RecordAssembly(
	result *AssembleResult,
	modelID string,
	strategy AllocationStrategy,
	durationSeconds float64,
) {
	strategyStr := string(strategy)

	// Record token totals
	contextTokensTotal.WithLabelValues(modelID, strategyStr).Observe(float64(result.TotalTokens))

	// Record utilization
	contextUtilization.WithLabelValues(modelID, strategyStr).Observe(result.BudgetUsed)

	// Record duration
	contextAssemblyDuration.WithLabelValues(modelID).Observe(durationSeconds)

	// Record truncation events
	for _, t := range result.Truncations {
		contextTruncationEvents.WithLabelValues(t.Component, t.Reason).Inc()
	}

	// Track budget exhaustion (>90% utilization)
	if result.BudgetUsed > 0.9 {
		contextBudgetExhausted.WithLabelValues(modelID, strategyStr).Inc()
	}

	// Count sources and history from result messages
	sourceCount := 0
	historyCount := 0
	for _, msg := range result.Messages {
		if msg.Role == "system" {
			// Sources are embedded in system message - count by [SU-] markers
			for i := 0; i < len(msg.Content); i++ {
				if i+4 < len(msg.Content) && msg.Content[i:i+4] == "[SU-" {
					sourceCount++
				}
			}
		} else if msg.Role == "assistant" || msg.Role == "user" {
			historyCount++
		}
	}
	// Subtract the final user query from history count
	if historyCount > 0 {
		historyCount--
	}

	contextSourcesIncluded.WithLabelValues(strategyStr).Observe(float64(sourceCount))
	contextHistoryIncluded.WithLabelValues(strategyStr).Observe(float64(historyCount))
}

// RecordDetailedAssembly records comprehensive metrics including truncation details
func (p *prometheusRecorder) RecordDetailedAssembly(
	result *AssembleResult,
	metrics *AssemblyMetrics,
	modelID string,
	strategy AllocationStrategy,
	durationSeconds float64,
) {
	// Record basic metrics first
	p.RecordAssembly(result, modelID, strategy, durationSeconds)

	strategyStr := string(strategy)
	shape := metrics.Shape
	if shape == "" {
		shape = "unknown"
	}
	truncationMode := metrics.TruncationMode
	if truncationMode == "" {
		truncationMode = "proportional"
	}

	// Record conversation length
	conversationLength.WithLabelValues(strategyStr).Observe(float64(metrics.InputHistoryMessages))

	// Record history truncation metrics
	messagesDropped := metrics.InputHistoryMessages - metrics.OutputHistoryMessages
	if messagesDropped > 0 {
		historyMessagesDropped.WithLabelValues(strategyStr, shape).Observe(float64(messagesDropped))

		tokensDropped := metrics.InputHistoryTokens - metrics.OutputHistoryTokens
		if tokensDropped > 0 {
			historyTokensDropped.WithLabelValues(strategyStr).Observe(float64(tokensDropped))
		}
	}

	// Record source truncation metrics
	sourcesDroppedCount := metrics.InputSources - metrics.OutputSources
	if sourcesDroppedCount > 0 {
		sourcesDropped.WithLabelValues(strategyStr, truncationMode).Observe(float64(sourcesDroppedCount))

		sourceTokensLost := metrics.InputSourceTokens - metrics.OutputSourceTokens
		if sourceTokensLost > 0 {
			sourceTokensDropped.WithLabelValues(strategyStr, truncationMode).Observe(float64(sourceTokensLost))
		}
	}

	// Track truncation occurrence
	if metrics.HistoryTruncated && metrics.SourcesTruncated {
		conversationTruncated.WithLabelValues(strategyStr, "both").Inc()
	} else if metrics.HistoryTruncated {
		conversationTruncated.WithLabelValues(strategyStr, "history").Inc()
	} else if metrics.SourcesTruncated {
		conversationTruncated.WithLabelValues(strategyStr, "sources").Inc()
	}

	// Track long conversation truncation
	if metrics.InputHistoryMessages > 10 && metrics.HistoryTruncated {
		longConversationTruncated.WithLabelValues(strategyStr).Inc()
	}
}

// noopRecorder is a no-op metrics recorder for testing
type noopRecorder struct{}

// NoopMetricsRecorder is a metrics recorder that does nothing
var NoopMetricsRecorder MetricsRecorder = &noopRecorder{}

// RecordAssembly does nothing
func (n *noopRecorder) RecordAssembly(*AssembleResult, string, AllocationStrategy, float64) {}

// RecordDetailedAssembly does nothing
func (n *noopRecorder) RecordDetailedAssembly(*AssembleResult, *AssemblyMetrics, string, AllocationStrategy, float64) {
}
