// Package answershape provides answer shape inference for RAG retrieval optimization.
package answershape

import (
	"context"
	"regexp"
	"strings"
	"sync"
	"time"
)

// cacheEntry holds a cached inference result with its expiration time.
type cacheEntry struct {
	result    *InferenceResult
	expiresAt time.Time
}

// ttlCache is a simple TTL-aware cache with a max size limit.
type ttlCache struct {
	mu      sync.RWMutex
	entries map[string]cacheEntry
	ttl     time.Duration
	maxSize int
}

const defaultMaxCacheSize = 10000

func newTTLCache(ttlSeconds int, maxSize int) *ttlCache {
	if maxSize <= 0 {
		maxSize = defaultMaxCacheSize
	}
	return &ttlCache{
		entries: make(map[string]cacheEntry),
		ttl:     time.Duration(ttlSeconds) * time.Second,
		maxSize: maxSize,
	}
}

func (c *ttlCache) Load(key string) (*InferenceResult, bool) {
	c.mu.RLock()
	entry, ok := c.entries[key]
	c.mu.RUnlock()
	if !ok {
		return nil, false
	}
	if time.Now().After(entry.expiresAt) {
		// Expired — remove lazily
		c.mu.Lock()
		delete(c.entries, key)
		c.mu.Unlock()
		return nil, false
	}
	return entry.result, true
}

func (c *ttlCache) Store(key string, result *InferenceResult) {
	c.mu.Lock()
	// Evict all entries when cache is full to keep it simple and bounded
	if len(c.entries) >= c.maxSize {
		c.entries = make(map[string]cacheEntry)
	}
	c.entries[key] = cacheEntry{
		result:    result,
		expiresAt: time.Now().Add(c.ttl),
	}
	c.mu.Unlock()
}

func (c *ttlCache) Clear() {
	c.mu.Lock()
	c.entries = make(map[string]cacheEntry)
	c.mu.Unlock()
}

// PatternInferrer uses pattern matching to infer answer shapes
type PatternInferrer struct {
	config   *Config
	patterns *patternSet
	cache    *ttlCache
}

// patternSet contains compiled regex patterns for shape detection
type patternSet struct {
	enumerativePatterns  []*compiledPattern
	exhaustivePatterns   []*compiledPattern
	hierarchicalPatterns []*compiledPattern
	comparativePatterns  []*compiledPattern
	proceduralPatterns   []*compiledPattern
	factualPatterns      []*compiledPattern
}

// compiledPattern holds a compiled regex and its metadata
type compiledPattern struct {
	regex        *regexp.Regexp
	weight       float32
	signalType   SignalType
	patternLabel string
}

// NewPatternInferrer creates a new pattern-based inferrer
func NewPatternInferrer(cfg *Config) *PatternInferrer {
	if cfg == nil {
		cfg = DefaultConfig()
	}
	return &PatternInferrer{
		config:   cfg,
		patterns: buildPatternSet(),
		cache:    newTTLCache(cfg.CacheTTLSeconds, defaultMaxCacheSize),
	}
}

// Infer analyzes a query and returns the expected answer shape
func (p *PatternInferrer) Infer(ctx context.Context, query string) (*InferenceResult, error) {
	return p.InferWithContext(ctx, query, nil)
}

// InferWithContext analyzes a query with conversation context
func (p *PatternInferrer) InferWithContext(ctx context.Context, query string, history []Message) (*InferenceResult, error) {
	// Check cache first
	if p.config.EnableCaching {
		if cached, ok := p.cache.Load(strings.ToLower(query)); ok {
			return cached, nil
		}
	}

	// Normalize query for analysis
	normalizedQuery := strings.ToLower(strings.TrimSpace(query))

	// FIX 1: DETERMINISTIC OVERRIDES - Force shape classification for unambiguous question types
	// These patterns MUST NOT fall back to exploratory - they have definite, closed answers
	if deterministicResult := p.checkDeterministicPatterns(normalizedQuery); deterministicResult != nil {
		// Cache and return immediately - no inference needed
		if p.config.EnableCaching {
			p.cache.Store(strings.ToLower(query), deterministicResult)
		}
		return deterministicResult, nil
	}

	// Calculate scores for each shape
	scores := make(map[Shape]float32)
	signals := make(map[Shape][]InferenceSignal)

	// Check enumerative patterns
	score, sigs := p.matchPatterns(normalizedQuery, p.patterns.enumerativePatterns)
	scores[ShapeEnumerative] = score
	signals[ShapeEnumerative] = sigs

	// Check exhaustive patterns
	score, sigs = p.matchPatterns(normalizedQuery, p.patterns.exhaustivePatterns)
	scores[ShapeExhaustive] = score
	signals[ShapeExhaustive] = sigs

	// Check hierarchical patterns
	score, sigs = p.matchPatterns(normalizedQuery, p.patterns.hierarchicalPatterns)
	scores[ShapeHierarchical] = score
	signals[ShapeHierarchical] = sigs

	// Check comparative patterns
	score, sigs = p.matchPatterns(normalizedQuery, p.patterns.comparativePatterns)
	scores[ShapeComparative] = score
	signals[ShapeComparative] = sigs

	// Check procedural patterns
	score, sigs = p.matchPatterns(normalizedQuery, p.patterns.proceduralPatterns)
	scores[ShapeProcedural] = score
	signals[ShapeProcedural] = sigs

	// Check factual patterns
	score, sigs = p.matchPatterns(normalizedQuery, p.patterns.factualPatterns)
	scores[ShapeFactual] = score
	signals[ShapeFactual] = sigs

	// Find the best matching shape
	bestShape := p.config.DefaultShape
	bestScore := float32(0)

	for shape, score := range scores {
		if score > bestScore {
			bestScore = score
			bestShape = shape
		}
	}

	// If no strong match, use exploratory as default
	if bestScore < 0.3 {
		bestShape = ShapeExploratory
		bestScore = 0.5
	}

	// Build result
	result := &InferenceResult{
		Shape:                      bestShape,
		CoverageExpectation:        ShapeToCoverageExpectation[bestShape],
		Depth:                      p.inferDepth(normalizedQuery),
		MergeAllowed:               p.inferMergeAllowed(bestShape, normalizedQuery),
		ExpectedMinItems:           p.inferMinItems(bestShape, normalizedQuery),
		Confidence:                 p.normalizeConfidence(bestScore),
		Signals:                    signals[bestShape],
		SuggestedRetrievalStrategy: ShapeToRetrievalStrategy[bestShape],
	}

	// Cache the result
	if p.config.EnableCaching {
		p.cache.Store(strings.ToLower(query), result)
	}

	return result, nil
}

// ========== FIX 1: DETERMINISTIC SHAPE OVERRIDES ==========
// These patterns bypass inference entirely for unambiguous question types.
// The system MUST NOT classify these as "exploratory" and tell the LLM to elaborate.

// deterministicPattern represents a pattern that forces a specific shape
type deterministicPattern struct {
	regex       *regexp.Regexp
	shape       Shape
	minItems    int
	description string
}

// deterministicPatterns contains patterns that MUST NOT fall back to exploratory
var deterministicPatterns = []deterministicPattern{
	// Business segment questions - ALWAYS enumerative
	// Handles: "what are the segments", "what are the primary business segments", "what are the main segments"
	{regexp.MustCompile(`(?i)what\s+are\s+(the\s+)?(main\s+|primary\s+|key\s+|major\s+)?(business\s+)?segments?`), ShapeEnumerative, 2, "business segments"},
	{regexp.MustCompile(`(?i)what\s+are\s+(the\s+)?(reportable\s+)?segments?`), ShapeEnumerative, 2, "reportable segments"},
	{regexp.MustCompile(`(?i)(business\s+)?segments?\s+(reported|disclosed|identified)\s+by`), ShapeEnumerative, 2, "reported segments"},
	{regexp.MustCompile(`(?i)how\s+many\s+segments?`), ShapeFactual, 0, "segment count"},

	// Revenue source questions - ALWAYS enumerative
	{regexp.MustCompile(`(?i)what\s+are\s+(the\s+)?(main\s+|primary\s+|key\s+)?revenue\s+sources?`), ShapeEnumerative, 2, "revenue sources"},
	{regexp.MustCompile(`(?i)how\s+does\s+.*\s+generate\s+revenue`), ShapeEnumerative, 2, "revenue generation"},
	{regexp.MustCompile(`(?i)what\s+are\s+(the\s+)?sources?\s+of\s+(revenue|income)`), ShapeEnumerative, 2, "income sources"},

	// Employee/headcount questions - ALWAYS factual
	// Handles: "how many employees", "how many employees did X have", "how many employees does X have"
	{regexp.MustCompile(`(?i)how\s+many\s+employees?(\s+(did|does|do|has|have|had)\s+)?`), ShapeFactual, 0, "employee count"},
	{regexp.MustCompile(`(?i)what\s+(is|was)\s+(the\s+)?(total\s+)?(number\s+of\s+)?employees?`), ShapeFactual, 0, "employee count"},
	{regexp.MustCompile(`(?i)what\s+is\s+(the\s+)?(total\s+)?headcount`), ShapeFactual, 0, "headcount"},
	{regexp.MustCompile(`(?i)number\s+of\s+employees?`), ShapeFactual, 0, "employee number"},
	{regexp.MustCompile(`(?i)employees?\s+(count|total|number)`), ShapeFactual, 0, "employee count"},

	// Incident/event questions - ALWAYS factual or enumerative
	// Handles: "what were the incidents", "what incidents", "any incidents", "were there incidents"
	{regexp.MustCompile(`(?i)(any|were\s+there)\s+(cybersecurity\s+)?incidents?`), ShapeFactual, 0, "incidents"},
	{regexp.MustCompile(`(?i)what\s+(were|are)\s+(\w+('s|'s)?\s+)?(cybersecurity\s+)?incidents?`), ShapeEnumerative, 1, "incident list"},
	{regexp.MustCompile(`(?i)what\s+(cybersecurity\s+)?incidents?\s+(did|does|has|have|were|are)`), ShapeEnumerative, 1, "incident list"},
	{regexp.MustCompile(`(?i)list\s+(the\s+)?(cybersecurity\s+)?incidents?`), ShapeEnumerative, 1, "incident enumeration"},
	{regexp.MustCompile(`(?i)(cybersecurity\s+)?incidents?\s+in\s+\d{4}`), ShapeEnumerative, 1, "incident year query"},

	// Financial metrics - ALWAYS factual
	{regexp.MustCompile(`(?i)what\s+(is|was)\s+(the\s+)?(total\s+)?revenue`), ShapeFactual, 0, "revenue amount"},
	{regexp.MustCompile(`(?i)what\s+(is|was)\s+(the\s+)?net\s+income`), ShapeFactual, 0, "net income"},
	{regexp.MustCompile(`(?i)what\s+(is|was)\s+(the\s+)?profit`), ShapeFactual, 0, "profit"},
	{regexp.MustCompile(`(?i)what\s+(is|was)\s+(the\s+)?market\s+cap`), ShapeFactual, 0, "market cap"},

	// Risk factor questions - ALWAYS enumerative
	{regexp.MustCompile(`(?i)what\s+are\s+(the\s+)?(main\s+|primary\s+|key\s+|major\s+)?risk\s+factors?`), ShapeEnumerative, 3, "risk factors"},
	{regexp.MustCompile(`(?i)what\s+(are\s+)?(the\s+)?(main\s+|primary\s+|key\s+|major\s+)?risks?\s+(does|did|that)\s+`), ShapeEnumerative, 2, "company risks"},
	{regexp.MustCompile(`(?i)what\s+risks?\s+.*\s+(face|identify|identifies|identified|highlight|highlights)`), ShapeEnumerative, 2, "company risks"},
	{regexp.MustCompile(`(?i)(major|main|primary|key)\s+risks?\s+(related\s+to|associated\s+with|regarding)`), ShapeEnumerative, 2, "risk enumeration"},
	{regexp.MustCompile(`(?i)what\s+are\s+(the\s+)?(major|main|primary|key)\s+risks?\s+\w+\s+identifies?`), ShapeEnumerative, 2, "identified risks"},

	// Regulatory/compliance/scrutiny questions - ALWAYS enumerative
	// Handles: "what regulatory areas", "areas of scrutiny", "compliance areas"
	{regexp.MustCompile(`(?i)what\s+(regulatory|compliance|legal)\s+(areas?|issues?|concerns?|matters?)`), ShapeEnumerative, 2, "regulatory areas"},
	{regexp.MustCompile(`(?i)(areas?|topics?)\s+of\s+(increased\s+)?(regulatory\s+)?(scrutiny|focus|concern|attention)`), ShapeEnumerative, 2, "scrutiny areas"},
	{regexp.MustCompile(`(?i)what\s+.*\s+(highlight|highlights|identifies?|mentions?)\s+as\s+(areas?|topics?)\s+of`), ShapeEnumerative, 2, "highlighted areas"},
	{regexp.MustCompile(`(?i)(regulatory|compliance|legal)\s+(risks?|concerns?|issues?|challenges?)`), ShapeEnumerative, 2, "regulatory concerns"},

	// Product/service enumeration - ALWAYS enumerative
	{regexp.MustCompile(`(?i)what\s+(products?|services?)\s+does\s+.*\s+(offer|provide|sell)`), ShapeEnumerative, 2, "products/services"},
	{regexp.MustCompile(`(?i)what\s+are\s+(the\s+)?(main\s+|primary\s+|key\s+)?(products?|services?)`), ShapeEnumerative, 2, "product list"},

	// Geographic/market questions - ALWAYS enumerative
	{regexp.MustCompile(`(?i)what\s+(markets?|regions?|countries?)\s+does\s+.*\s+operate`), ShapeEnumerative, 2, "markets"},
	{regexp.MustCompile(`(?i)where\s+does\s+.*\s+operate`), ShapeEnumerative, 2, "operating regions"},

	// Generic "what are the X" enumerative patterns - catch common enumeration questions
	// These are lower priority but catch things like "what are the challenges", "what are the opportunities"
	{regexp.MustCompile(`(?i)what\s+are\s+(the\s+)?(main\s+|primary\s+|key\s+|major\s+)?(challenges?|opportunities?|priorities?|initiatives?|strategies?|objectives?|goals?)`), ShapeEnumerative, 2, "enumeration"},
	{regexp.MustCompile(`(?i)what\s+.*\s+(identifies?|highlights?|lists?|mentions?|describes?)\s+as\s+(the\s+)?(main|key|primary|major)`), ShapeEnumerative, 2, "identified items"},

	// ========== FIX: EXHAUSTIVE QUERY DETECTION FOR COMPREHENSIVE RETRIEVAL ==========
	// These patterns trigger exhaustive retrieval to ensure all relevant information is found
	// NOTE: Only generic patterns here - no domain-specific keywords

	// Follow-up questions requesting more information - ALWAYS exhaustive
	// These are universal patterns that indicate user wants comprehensive coverage
	{regexp.MustCompile(`(?i)^(is\s+)?that\s+(all|everything|it)\??$`), ShapeExhaustive, 0, "is that all"},
	{regexp.MustCompile(`(?i)^(is\s+there\s+)?(anything|something)\s+(else|more)\??$`), ShapeExhaustive, 0, "anything else"},
	{regexp.MustCompile(`(?i)^(is\s+there|are\s+there)\s+(more|any\s+more|others?)\??$`), ShapeExhaustive, 0, "is there more"},
	{regexp.MustCompile(`(?i)^what\s+else\??$`), ShapeExhaustive, 0, "what else"},
	{regexp.MustCompile(`(?i)^any\s+(other|more)\s+`), ShapeExhaustive, 0, "any other"},
	{regexp.MustCompile(`(?i)^did\s+(i|you)\s+miss\s+(anything|something)`), ShapeExhaustive, 0, "did I miss"},
	{regexp.MustCompile(`(?i)(complete|comprehensive|exhaustive|full)\s+(list|coverage|information|answer)`), ShapeExhaustive, 0, "complete list"},
}

// checkDeterministicPatterns checks if a query matches a deterministic pattern
// Returns a pre-built InferenceResult if matched, nil otherwise
func (p *PatternInferrer) checkDeterministicPatterns(normalizedQuery string) *InferenceResult {
	for _, dp := range deterministicPatterns {
		if dp.regex.MatchString(normalizedQuery) {
			return &InferenceResult{
				Shape:               dp.shape,
				CoverageExpectation: ShapeToCoverageExpectation[dp.shape],
				Depth:               DepthMedium,
				MergeAllowed:        false, // NEVER allow merging for deterministic patterns
				ExpectedMinItems:    dp.minItems,
				Confidence:          0.95, // High confidence - this is deterministic
				Signals: []InferenceSignal{
					{
						Type:         SignalKeywordMatch,
						Pattern:      dp.description,
						Contribution: 0.95,
					},
				},
				SuggestedRetrievalStrategy: ShapeToRetrievalStrategy[dp.shape],
			}
		}
	}
	return nil
}

// matchPatterns checks a query against a set of patterns and returns the total score
func (p *PatternInferrer) matchPatterns(normalizedQuery string, patterns []*compiledPattern) (float32, []InferenceSignal) {
	var totalScore float32
	var signals []InferenceSignal

	for _, pat := range patterns {
		if pat.regex.MatchString(normalizedQuery) {
			totalScore += pat.weight
			signals = append(signals, InferenceSignal{
				Type:         pat.signalType,
				Pattern:      pat.patternLabel,
				Contribution: pat.weight,
			})
		}
	}

	return totalScore, signals
}

// inferDepth determines the expected depth of the answer
func (p *PatternInferrer) inferDepth(query string) Depth {
	// Deep depth indicators
	deepPatterns := []string{
		`detail`,
		`explain.*in.*depth`,
		`comprehensive`,
		`thorough`,
		`complete.*explanation`,
		`everything.*about`,
		`all.*details`,
	}

	// Shallow depth indicators
	shallowPatterns := []string{
		`brief`,
		`quick`,
		`summary`,
		`overview`,
		`short`,
		`simple`,
		`just`,
		`only`,
	}

	for _, pat := range deepPatterns {
		if matched, _ := regexp.MatchString(pat, query); matched {
			return DepthDeep
		}
	}

	for _, pat := range shallowPatterns {
		if matched, _ := regexp.MatchString(pat, query); matched {
			return DepthShallow
		}
	}

	return DepthMedium
}

// inferMergeAllowed determines if similar items can be merged
func (p *PatternInferrer) inferMergeAllowed(shape Shape, query string) bool {
	// Enumerative and exhaustive shapes should not merge by default
	if shape == ShapeEnumerative || shape == ShapeExhaustive {
		return false
	}

	// Check for explicit "each" or "every" which indicate no merging
	noMergePatterns := []string{
		`\beach\b`,
		`\bevery\b`,
		`\bseparate`,
		`\bdistinct`,
		`\bindividual`,
		`\bone by one`,
	}

	for _, pat := range noMergePatterns {
		if matched, _ := regexp.MatchString(pat, query); matched {
			return false
		}
	}

	return true
}

// inferMinItems estimates minimum expected items for list-type answers
func (p *PatternInferrer) inferMinItems(shape Shape, query string) int {
	if shape != ShapeEnumerative && shape != ShapeExhaustive {
		return 0
	}

	// Look for explicit number mentions
	numberPatterns := map[string]int{
		`top\s*(\d+)`:     -1, // Will extract the number
		`first\s*(\d+)`:   -1,
		`(\d+)\s*main`:    -1,
		`(\d+)\s*key`:     -1,
		`(\d+)\s*primary`: -1,
	}

	for pat := range numberPatterns {
		re := regexp.MustCompile(pat)
		if matches := re.FindStringSubmatch(query); len(matches) > 1 {
			// Parse the captured number (simplified - just check common values)
			switch matches[1] {
			case "3":
				return 3
			case "5":
				return 5
			case "10":
				return 10
			}
		}
	}

	// Default minimums based on shape
	if shape == ShapeExhaustive {
		return 5 // Exhaustive implies more items
	}
	return 3 // Enumerative default
}

// normalizeConfidence converts raw score to 0-1 confidence
func (p *PatternInferrer) normalizeConfidence(score float32) float32 {
	// Sigmoid-like normalization
	if score <= 0 {
		return 0.3 // Base confidence for default shape
	}
	if score >= 2.0 {
		return 0.95
	}
	// Linear interpolation for mid-range
	return 0.3 + (score/2.0)*0.65
}

// buildPatternSet creates the compiled pattern set
func buildPatternSet() *patternSet {
	return &patternSet{
		enumerativePatterns:  buildEnumerativePatterns(),
		exhaustivePatterns:   buildExhaustivePatterns(),
		hierarchicalPatterns: buildHierarchicalPatterns(),
		comparativePatterns:  buildComparativePatterns(),
		proceduralPatterns:   buildProceduralPatterns(),
		factualPatterns:      buildFactualPatterns(),
	}
}

func buildEnumerativePatterns() []*compiledPattern {
	patterns := []struct {
		pattern    string
		weight     float32
		signalType SignalType
		label      string
	}{
		// List indicators
		{`\blist\b`, 0.8, SignalListIndicator, "list"},
		{`\bwhat\s+are\s+(the\s+)?(\w+\s+)?types?\b`, 0.7, SignalListIndicator, "what are types"},
		{`\bwhat\s+are\s+(the\s+)?(\w+\s+)?kinds?\b`, 0.7, SignalListIndicator, "what are kinds"},
		{`\bwhat\s+are\s+(the\s+)?(\w+\s+)?categories\b`, 0.7, SignalListIndicator, "what are categories"},
		{`\bwhat\s+are\s+(the\s+)?(\w+\s+)?features?\b`, 0.6, SignalListIndicator, "what are features"},
		{`\bwhat\s+are\s+(the\s+)?(\w+\s+)?benefits?\b`, 0.6, SignalListIndicator, "what are benefits"},
		{`\bwhat\s+are\s+(the\s+)?(\w+\s+)?risks?\b`, 0.6, SignalListIndicator, "what are risks"},
		{`\bwhat\s+are\s+(the\s+)?(\w+\s+)?components?\b`, 0.6, SignalListIndicator, "what are components"},
		{`\bwhat\s+are\s+(the\s+)?(\w+\s+)?options?\b`, 0.6, SignalListIndicator, "what are options"},
		{`\bwhat\s+are\s+(the\s+)?(\w+\s+)?examples?\b`, 0.6, SignalListIndicator, "what are examples"},

		// Enumeration keywords
		{`\benumerate\b`, 0.9, SignalKeywordMatch, "enumerate"},
		{`\bidentify\s+(the\s+)?(\w+\s+)?(different|various|multiple)\b`, 0.7, SignalKeywordMatch, "identify different"},
		{`\bname\s+(the\s+)?(\w+\s+)?(different|various|multiple|all)\b`, 0.7, SignalKeywordMatch, "name multiple"},
		{`\bdescribe\s+(the\s+)?(\w+\s+)?(different|various|multiple)\b`, 0.6, SignalKeywordMatch, "describe multiple"},

		// Plurality indicators
		{`\bwhat\s+.*\s+are\s+available\b`, 0.5, SignalListIndicator, "what are available"},
		{`\bwhat\s+.*\s+exist\b`, 0.4, SignalListIndicator, "what exist"},
		{`\btypes\s+of\b`, 0.6, SignalListIndicator, "types of"},
		{`\bkinds\s+of\b`, 0.6, SignalListIndicator, "kinds of"},
		{`\bforms\s+of\b`, 0.5, SignalListIndicator, "forms of"},
		{`\bcategories\s+of\b`, 0.6, SignalListIndicator, "categories of"},

		// Quantifier indicators
		{`\bhow\s+many\b`, 0.5, SignalQuantifier, "how many"},
		{`\bmultiple\b`, 0.4, SignalListIndicator, "multiple"},
		{`\bseveral\b`, 0.4, SignalListIndicator, "several"},
		{`\bvarious\b`, 0.4, SignalListIndicator, "various"},
		{`\bdifferent\b`, 0.3, SignalListIndicator, "different"},
	}

	return compilePatterns(patterns)
}

func buildExhaustivePatterns() []*compiledPattern {
	patterns := []struct {
		pattern    string
		weight     float32
		signalType SignalType
		label      string
	}{
		// "All" quantifiers
		{`\ball\s+(the\s+)?(\w+\s+)?`, 0.8, SignalQuantifier, "all"},
		{`\bevery\s+`, 0.7, SignalQuantifier, "every"},
		{`\beach\s+`, 0.6, SignalQuantifier, "each"},
		{`\bcomplete\s+list\b`, 0.9, SignalQuantifier, "complete list"},
		{`\bfull\s+list\b`, 0.9, SignalQuantifier, "full list"},
		{`\bentire\b`, 0.6, SignalQuantifier, "entire"},
		{`\bexhaustive\b`, 0.9, SignalKeywordMatch, "exhaustive"},
		{`\bcomprehensive\b`, 0.7, SignalKeywordMatch, "comprehensive"},

		// Totality indicators
		{`\bwhat\s+are\s+all\b`, 0.8, SignalQuantifier, "what are all"},
		{`\blist\s+all\b`, 0.9, SignalQuantifier, "list all"},
		{`\bname\s+all\b`, 0.9, SignalQuantifier, "name all"},
		{`\bidentify\s+all\b`, 0.8, SignalQuantifier, "identify all"},
		{`\bdescribe\s+all\b`, 0.7, SignalQuantifier, "describe all"},

		// Requirement keywords
		{`\ball\s+requirements?\b`, 0.8, SignalKeywordMatch, "all requirements"},
		{`\ball\s+rules?\b`, 0.8, SignalKeywordMatch, "all rules"},
		{`\ball\s+regulations?\b`, 0.8, SignalKeywordMatch, "all regulations"},
		{`\ball\s+policies?\b`, 0.8, SignalKeywordMatch, "all policies"},
		{`\ball\s+conditions?\b`, 0.7, SignalKeywordMatch, "all conditions"},
		{`\ball\s+criteria\b`, 0.7, SignalKeywordMatch, "all criteria"},

		// Nothing missing indicators
		{`\bmissing\s+any\b`, 0.5, SignalKeywordMatch, "missing any"},
		{`\bno\s+exceptions?\b`, 0.5, SignalKeywordMatch, "no exceptions"},
		{`\bwithout\s+exception\b`, 0.5, SignalKeywordMatch, "without exception"},

		// ========== ADDITIONAL EXHAUSTIVE PATTERNS ==========
		// NOTE: Only generic patterns - no domain-specific keywords

		// Follow-up patterns (user asking for more) - universal indicators
		{`\bis\s+that\s+(all|everything)\b`, 0.8, SignalKeywordMatch, "is that all"},
		{`\banything\s+(else|more)\b`, 0.7, SignalKeywordMatch, "anything else"},
		{`\bis\s+there\s+(more|any\s+more)\b`, 0.7, SignalKeywordMatch, "is there more"},
		{`\bwhat\s+else\b`, 0.7, SignalKeywordMatch, "what else"},
		{`\bany\s+other\b`, 0.6, SignalKeywordMatch, "any other"},
		{`\bdid\s+(i|you)\s+miss\b`, 0.7, SignalKeywordMatch, "did I miss"},
	}

	return compilePatterns(patterns)
}

func buildHierarchicalPatterns() []*compiledPattern {
	patterns := []struct {
		pattern    string
		weight     float32
		signalType SignalType
		label      string
	}{
		// Structure keywords
		{`\barchitecture\b`, 0.8, SignalStructure, "architecture"},
		{`\bstructure\b`, 0.7, SignalStructure, "structure"},
		{`\bhierarchy\b`, 0.9, SignalStructure, "hierarchy"},
		{`\borganization\b`, 0.6, SignalStructure, "organization"},
		{`\bframework\b`, 0.6, SignalStructure, "framework"},
		{`\btree\b`, 0.5, SignalStructure, "tree"},

		// Relationship keywords
		{`\bhow\s+.*\s+organized\b`, 0.7, SignalStructure, "how organized"},
		{`\bhow\s+.*\s+structured\b`, 0.7, SignalStructure, "how structured"},
		{`\bparent.*child\b`, 0.7, SignalStructure, "parent child"},
		{`\bnested\b`, 0.6, SignalStructure, "nested"},
		{`\blevels?\s+of\b`, 0.5, SignalStructure, "levels of"},
		{`\blayers?\s+of\b`, 0.5, SignalStructure, "layers of"},

		// Component relationships
		{`\bcomponents?\s+and\s+.*\s+relationships?\b`, 0.7, SignalStructure, "component relationships"},
		{`\bhow\s+.*\s+relate\b`, 0.5, SignalStructure, "how relate"},
		{`\bbreakdown\b`, 0.6, SignalStructure, "breakdown"},
		{`\bdecomposition\b`, 0.7, SignalStructure, "decomposition"},

		// System explanation
		{`\bexplain\s+the\s+(system|architecture|structure)\b`, 0.7, SignalStructure, "explain system"},
		{`\boverview\s+of\s+(the\s+)?(system|architecture|structure)\b`, 0.6, SignalStructure, "overview of system"},
	}

	return compilePatterns(patterns)
}

func buildComparativePatterns() []*compiledPattern {
	patterns := []struct {
		pattern    string
		weight     float32
		signalType SignalType
		label      string
	}{
		// Comparison keywords
		{`\bcompare\b`, 0.9, SignalComparison, "compare"},
		{`\bcomparison\b`, 0.9, SignalComparison, "comparison"},
		{`\bcontrast\b`, 0.8, SignalComparison, "contrast"},
		{`\bdifferences?\s+between\b`, 0.9, SignalComparison, "difference between"},
		{`\bsimilarities?\s+between\b`, 0.8, SignalComparison, "similarity between"},

		// VS patterns
		{`\bvs\.?\b`, 0.8, SignalComparison, "vs"},
		{`\bversus\b`, 0.8, SignalComparison, "versus"},
		{`\bor\s+(?:the\s+)?other\b`, 0.5, SignalComparison, "or the other"},

		// Comparative adjectives
		{`\bbetter\s+than\b`, 0.6, SignalComparison, "better than"},
		{`\bworse\s+than\b`, 0.6, SignalComparison, "worse than"},
		{`\bmore\s+\w+\s+than\b`, 0.5, SignalComparison, "more than"},
		{`\bless\s+\w+\s+than\b`, 0.5, SignalComparison, "less than"},

		// Choice patterns
		{`\bwhich\s+is\s+(better|best|preferred)\b`, 0.7, SignalComparison, "which is better"},
		{`\bshould\s+I\s+(choose|use|pick)\b`, 0.5, SignalComparison, "should I choose"},
		{`\badvantages?\s+and\s+disadvantages?\b`, 0.8, SignalComparison, "advantages and disadvantages"},
		{`\bpros?\s+and\s+cons?\b`, 0.8, SignalComparison, "pros and cons"},
	}

	return compilePatterns(patterns)
}

func buildProceduralPatterns() []*compiledPattern {
	patterns := []struct {
		pattern    string
		weight     float32
		signalType SignalType
		label      string
	}{
		// How-to patterns
		{`\bhow\s+to\b`, 0.8, SignalSequence, "how to"},
		{`\bhow\s+do\s+(I|you|we)\b`, 0.7, SignalSequence, "how do I"},
		{`\bhow\s+can\s+(I|you|we)\b`, 0.6, SignalSequence, "how can I"},
		{`\bsteps?\s+to\b`, 0.9, SignalSequence, "steps to"},
		{`\bprocess\s+(for|of|to)\b`, 0.7, SignalSequence, "process for"},
		{`\bprocedure\s+(for|of|to)\b`, 0.8, SignalSequence, "procedure for"},

		// Instruction keywords
		{`\binstructions?\b`, 0.8, SignalSequence, "instructions"},
		{`\bguide\b`, 0.5, SignalSequence, "guide"},
		{`\btutorial\b`, 0.6, SignalSequence, "tutorial"},
		{`\bwalkthrough\b`, 0.7, SignalSequence, "walkthrough"},

		// Action sequence indicators
		{`\bstep\s*-?\s*by\s*-?\s*step\b`, 0.9, SignalSequence, "step by step"},
		{`\bfirst.*then\b`, 0.6, SignalSequence, "first then"},
		{`\bsequence\s+of\b`, 0.7, SignalSequence, "sequence of"},
		{`\border\s+of\s+(operations?|steps?|actions?)\b`, 0.7, SignalSequence, "order of operations"},

		// Task-oriented
		{`\bimplement\b`, 0.5, SignalKeywordMatch, "implement"},
		{`\bconfigure\b`, 0.5, SignalKeywordMatch, "configure"},
		{`\bsetup\b`, 0.5, SignalKeywordMatch, "setup"},
		{`\binstall\b`, 0.5, SignalKeywordMatch, "install"},
		{`\bcreate\b`, 0.4, SignalKeywordMatch, "create"},
	}

	return compilePatterns(patterns)
}

func buildFactualPatterns() []*compiledPattern {
	patterns := []struct {
		pattern    string
		weight     float32
		signalType SignalType
		label      string
	}{
		// Definition questions
		{`\bwhat\s+is\s+(a|an|the)?\s*\w+\s*\??\s*$`, 0.8, SignalQuestionWord, "what is"},
		{`\bdefine\b`, 0.9, SignalKeywordMatch, "define"},
		{`\bdefinition\s+of\b`, 0.9, SignalKeywordMatch, "definition of"},
		{`\bmeaning\s+of\b`, 0.7, SignalKeywordMatch, "meaning of"},
		{`\bwhat\s+does\s+.*\s+mean\b`, 0.7, SignalKeywordMatch, "what does mean"},

		// Single fact patterns
		{`\bwhen\s+(was|did|is|will)\b`, 0.7, SignalQuestionWord, "when"},
		{`\bwhere\s+(is|was|are|were)\b`, 0.6, SignalQuestionWord, "where"},
		{`\bwho\s+(is|was|are|were)\b`, 0.7, SignalQuestionWord, "who"},
		{`\bwhich\s+\w+\s+(is|was|does)\b`, 0.5, SignalQuestionWord, "which one"},

		// Value/number questions
		{`\bhow\s+much\b`, 0.7, SignalQuestionWord, "how much"},
		{`\bhow\s+long\b`, 0.6, SignalQuestionWord, "how long"},
		{`\bhow\s+old\b`, 0.6, SignalQuestionWord, "how old"},
		{`\bwhat\s+(is|was)\s+the\s+(date|time|number|value|amount|price|cost)\b`, 0.8, SignalQuestionWord, "what is the value"},

		// Specific lookup
		{`\bwhat\s+(is|was)\s+the\s+name\b`, 0.7, SignalQuestionWord, "what is the name"},
		{`\bwhat\s+(is|was)\s+the\s+status\b`, 0.7, SignalQuestionWord, "what is the status"},
	}

	return compilePatterns(patterns)
}

// compilePatterns creates compiled pattern structs from pattern definitions
func compilePatterns(patterns []struct {
	pattern    string
	weight     float32
	signalType SignalType
	label      string
}) []*compiledPattern {
	compiled := make([]*compiledPattern, 0, len(patterns))
	for _, p := range patterns {
		re, err := regexp.Compile(p.pattern)
		if err != nil {
			continue // Skip invalid patterns
		}
		compiled = append(compiled, &compiledPattern{
			regex:        re,
			weight:       p.weight,
			signalType:   p.signalType,
			patternLabel: p.label,
		})
	}
	return compiled
}

// ClearCache clears the inference cache
func (p *PatternInferrer) ClearCache() {
	p.cache.Clear()
}
