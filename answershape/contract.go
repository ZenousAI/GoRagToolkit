// Package answershape provides answer shape inference and response contracts.
package answershape

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
)

// ResponseContract defines machine-enforceable constraints for LLM generation.
// It ensures that responses adhere to expected structure, coverage, and format.
type ResponseContract struct {
	// ID is the unique identifier for this contract
	ID uuid.UUID `json:"id"`

	// Shape is the expected answer shape
	Shape Shape `json:"shape"`

	// Coverage constraints
	Coverage CoverageContract `json:"coverage"`

	// Structure constraints
	Structure StructureContract `json:"structure"`

	// Citation constraints
	Citation CitationContract `json:"citation"`

	// Format constraints
	Format FormatContract `json:"format"`
}

// CoverageContract specifies coverage requirements for the response
type CoverageContract struct {
	// MinItems is the minimum number of distinct items required
	MinItems int `json:"min_items,omitempty"`

	// MaxItems is the maximum number of items allowed (0 = unlimited)
	MaxItems int `json:"max_items,omitempty"`

	// MustCoverAllSUs requires all provided SUs to be addressed
	MustCoverAllSUs bool `json:"must_cover_all_sus"`

	// MustCoverSUIDs is a list of SU IDs that must be covered
	MustCoverSUIDs []uuid.UUID `json:"must_cover_su_ids,omitempty"`

	// NoMerging prohibits combining similar items
	NoMerging bool `json:"no_merging"`

	// NoOmitting prohibits skipping any provided items
	NoOmitting bool `json:"no_omitting"`

	// RequiredSemanticRoles lists semantic roles that must be covered
	RequiredSemanticRoles []string `json:"required_semantic_roles,omitempty"`
}

// StructureContract specifies structural requirements for the response
type StructureContract struct {
	// RequireNumberedList requires a numbered list format
	RequireNumberedList bool `json:"require_numbered_list,omitempty"`

	// RequireBulletList requires a bullet list format
	RequireBulletList bool `json:"require_bullet_list,omitempty"`

	// RequireHeadings requires section headings
	RequireHeadings bool `json:"require_headings,omitempty"`

	// PreserveHierarchy requires maintaining SU hierarchy in response
	PreserveHierarchy bool `json:"preserve_hierarchy,omitempty"`

	// PreserveOrder requires maintaining SU order in response
	PreserveOrder bool `json:"preserve_order,omitempty"`

	// MaxDepth limits nesting depth for hierarchical responses
	MaxDepth int `json:"max_depth,omitempty"`
}

// CitationContract specifies citation requirements for the response
type CitationContract struct {
	// CitationRequired requires citations for all claims
	CitationRequired bool `json:"citation_required"`

	// CitationFormat specifies how citations should be formatted
	CitationFormat CitationFormat `json:"citation_format,omitempty"`

	// InlineCitations requires citations inline with text
	InlineCitations bool `json:"inline_citations,omitempty"`

	// EndCitations requires citations at the end
	EndCitations bool `json:"end_citations,omitempty"`

	// MaxUnsourcedSentences is the max sentences without citation (0 = all must be cited)
	MaxUnsourcedSentences int `json:"max_unsourced_sentences,omitempty"`
}

// CitationFormat specifies how citations should be formatted
type CitationFormat string

const (
	// CitationFormatBracket uses [1], [2] style citations
	CitationFormatBracket CitationFormat = "bracket"
	// CitationFormatSuperscript uses ¹, ² style citations
	CitationFormatSuperscript CitationFormat = "superscript"
	// CitationFormatInline uses (Source: Document Name) style
	CitationFormatInline CitationFormat = "inline"
	// CitationFormatFootnote uses footnote-style citations
	CitationFormatFootnote CitationFormat = "footnote"
)

// PresentationDepth controls how much context and explanation to include
type PresentationDepth string

const (
	// PresentationDepthMinimal provides just the essential facts
	PresentationDepthMinimal PresentationDepth = "minimal"
	// PresentationDepthStandard provides facts with helpful context (default)
	PresentationDepthStandard PresentationDepth = "standard"
	// PresentationDepthDetailed provides comprehensive explanations with full context
	PresentationDepthDetailed PresentationDepth = "detailed"
)

// FormatContract specifies general formatting requirements
type FormatContract struct {
	// MaxLength is the maximum response length in characters
	MaxLength int `json:"max_length,omitempty"`

	// MaxTokens is the maximum response length in tokens
	MaxTokens int `json:"max_tokens,omitempty"`

	// Language is the required response language (ISO 639-1)
	Language string `json:"language,omitempty"`

	// Tone specifies the required tone (formal, casual, technical)
	Tone string `json:"tone,omitempty"`

	// IncludeSummary requires a summary section
	IncludeSummary bool `json:"include_summary,omitempty"`

	// SummaryPosition specifies where the summary should appear
	SummaryPosition SummaryPosition `json:"summary_position,omitempty"`

	// PresentationDepth controls how much context/explanation to include
	// "minimal" = essential facts only, "standard" = facts with context, "detailed" = comprehensive
	PresentationDepth PresentationDepth `json:"presentation_depth,omitempty"`

	// MinSentences is the minimum number of sentences for non-list responses
	MinSentences int `json:"min_sentences,omitempty"`

	// IncludeContext indicates whether to include contextual information from sources
	IncludeContext bool `json:"include_context,omitempty"`
}

// SummaryPosition specifies where summaries should appear
type SummaryPosition string

const (
	// SummaryAtStart places summary at the beginning
	SummaryAtStart SummaryPosition = "start"
	// SummaryAtEnd places summary at the end
	SummaryAtEnd SummaryPosition = "end"
)

// ShapeToTokenBudget defines base token budgets per answer shape.
// These represent the expected response length for each shape type.
var ShapeToTokenBudget = map[Shape]int{
	ShapeFactual:      100,
	ShapeEnumerative:  300,
	ShapeProcedural:   400,
	ShapeComparative:  450,
	ShapeHierarchical: 500,
	ShapeExploratory:  600,
	ShapeExhaustive:   600,
}

// DepthToMultiplier maps inferred depth to a token budget multiplier.
var DepthToMultiplier = map[Depth]float64{
	DepthShallow: 0.6,
	DepthMedium:  1.0,
	DepthDeep:    1.5,
}

// TokenBudgetForShape returns the token budget for a given shape and depth.
// It applies the depth multiplier to the shape's base budget.
func TokenBudgetForShape(shape Shape, depth Depth) int {
	base := ShapeToTokenBudget[shape]
	if base == 0 {
		base = 800 // fallback for unknown shapes
	}
	mult := DepthToMultiplier[depth]
	if mult == 0 {
		mult = 1.0 // default to medium depth
	}
	return int(float64(base) * mult)
}

// ContractBuilder provides a fluent API for building response contracts
type ContractBuilder struct {
	contract *ResponseContract
}

// NewContractBuilder creates a new contract builder
func NewContractBuilder() *ContractBuilder {
	return &ContractBuilder{
		contract: &ResponseContract{
			ID: uuid.New(),
		},
	}
}

// WithShape sets the expected answer shape
func (b *ContractBuilder) WithShape(shape Shape) *ContractBuilder {
	b.contract.Shape = shape
	return b
}

// WithMinItems sets the minimum number of items
func (b *ContractBuilder) WithMinItems(min int) *ContractBuilder {
	b.contract.Coverage.MinItems = min
	return b
}

// WithMaxItems sets the maximum number of items
func (b *ContractBuilder) WithMaxItems(max int) *ContractBuilder {
	b.contract.Coverage.MaxItems = max
	return b
}

// RequireCoverAllSUs requires all SUs to be addressed
func (b *ContractBuilder) RequireCoverAllSUs() *ContractBuilder {
	b.contract.Coverage.MustCoverAllSUs = true
	return b
}

// RequireCoverSUs requires specific SUs to be covered
func (b *ContractBuilder) RequireCoverSUs(suIDs ...uuid.UUID) *ContractBuilder {
	b.contract.Coverage.MustCoverSUIDs = append(b.contract.Coverage.MustCoverSUIDs, suIDs...)
	return b
}

// DisableMerging prohibits merging similar items
func (b *ContractBuilder) DisableMerging() *ContractBuilder {
	b.contract.Coverage.NoMerging = true
	return b
}

// DisableOmitting prohibits skipping items
func (b *ContractBuilder) DisableOmitting() *ContractBuilder {
	b.contract.Coverage.NoOmitting = true
	return b
}

// RequireSemanticRoles requires coverage of specific semantic roles
func (b *ContractBuilder) RequireSemanticRoles(roles ...string) *ContractBuilder {
	b.contract.Coverage.RequiredSemanticRoles = append(
		b.contract.Coverage.RequiredSemanticRoles, roles...)
	return b
}

// RequireNumberedList requires numbered list format
func (b *ContractBuilder) RequireNumberedList() *ContractBuilder {
	b.contract.Structure.RequireNumberedList = true
	b.contract.Structure.RequireBulletList = false
	return b
}

// RequireBulletList requires bullet list format
func (b *ContractBuilder) RequireBulletList() *ContractBuilder {
	b.contract.Structure.RequireBulletList = true
	b.contract.Structure.RequireNumberedList = false
	return b
}

// RequireHeadings requires section headings
func (b *ContractBuilder) RequireHeadings() *ContractBuilder {
	b.contract.Structure.RequireHeadings = true
	return b
}

// PreserveHierarchy requires maintaining SU hierarchy
func (b *ContractBuilder) PreserveHierarchy() *ContractBuilder {
	b.contract.Structure.PreserveHierarchy = true
	return b
}

// PreserveOrder requires maintaining SU order
func (b *ContractBuilder) PreserveOrder() *ContractBuilder {
	b.contract.Structure.PreserveOrder = true
	return b
}

// WithMaxDepth limits nesting depth
func (b *ContractBuilder) WithMaxDepth(depth int) *ContractBuilder {
	b.contract.Structure.MaxDepth = depth
	return b
}

// RequireCitations requires citations
func (b *ContractBuilder) RequireCitations() *ContractBuilder {
	b.contract.Citation.CitationRequired = true
	return b
}

// WithCitationFormat sets the citation format
func (b *ContractBuilder) WithCitationFormat(format CitationFormat) *ContractBuilder {
	b.contract.Citation.CitationFormat = format
	return b
}

// WithInlineCitations requires inline citations
func (b *ContractBuilder) WithInlineCitations() *ContractBuilder {
	b.contract.Citation.InlineCitations = true
	b.contract.Citation.EndCitations = false
	return b
}

// WithEndCitations requires citations at the end
func (b *ContractBuilder) WithEndCitations() *ContractBuilder {
	b.contract.Citation.EndCitations = true
	b.contract.Citation.InlineCitations = false
	return b
}

// WithMaxLength sets maximum response length
func (b *ContractBuilder) WithMaxLength(length int) *ContractBuilder {
	b.contract.Format.MaxLength = length
	return b
}

// WithMaxTokens sets maximum token count
func (b *ContractBuilder) WithMaxTokens(tokens int) *ContractBuilder {
	b.contract.Format.MaxTokens = tokens
	return b
}

// WithLanguage sets required response language
func (b *ContractBuilder) WithLanguage(lang string) *ContractBuilder {
	b.contract.Format.Language = lang
	return b
}

// WithTone sets required response tone
func (b *ContractBuilder) WithTone(tone string) *ContractBuilder {
	b.contract.Format.Tone = tone
	return b
}

// WithSummary requires a summary
func (b *ContractBuilder) WithSummary(position SummaryPosition) *ContractBuilder {
	b.contract.Format.IncludeSummary = true
	b.contract.Format.SummaryPosition = position
	return b
}

// WithPresentationDepth sets the presentation depth
func (b *ContractBuilder) WithPresentationDepth(depth PresentationDepth) *ContractBuilder {
	b.contract.Format.PresentationDepth = depth
	return b
}

// WithMinSentences sets the minimum number of sentences
func (b *ContractBuilder) WithMinSentences(min int) *ContractBuilder {
	b.contract.Format.MinSentences = min
	return b
}

// WithContext enables including contextual information
func (b *ContractBuilder) WithContext() *ContractBuilder {
	b.contract.Format.IncludeContext = true
	return b
}

// Build returns the completed contract
func (b *ContractBuilder) Build() *ResponseContract {
	return b.contract
}

// ContractFromShape creates a contract from an inference result
func ContractFromShape(result *InferenceResult) *ResponseContract {
	builder := NewContractBuilder().WithShape(result.Shape)

	// Set presentation depth based on shape
	// More complex shapes get more detailed presentation by default
	switch result.Shape {
	case ShapeEnumerative:
		builder.
			WithMinItems(result.ExpectedMinItems).
			DisableMerging().
			RequireCitations().
			RequireNumberedList().
			WithPresentationDepth(PresentationDepthStandard)
		if result.CoverageExpectation == CoverageHigh || result.CoverageExpectation == CoverageComplete {
			builder.DisableOmitting()
		}

	case ShapeExhaustive:
		builder.
			RequireCoverAllSUs().
			DisableMerging().
			DisableOmitting().
			RequireCitations().
			RequireNumberedList().
			WithPresentationDepth(PresentationDepthStandard).
			WithContext()
		if result.ExpectedMinItems > 0 {
			builder.WithMinItems(result.ExpectedMinItems)
		}

	case ShapeHierarchical:
		builder.
			PreserveHierarchy().
			RequireHeadings().
			RequireCitations().
			WithPresentationDepth(PresentationDepthStandard).
			WithContext()

	case ShapeComparative:
		builder.
			DisableMerging().
			RequireHeadings().
			RequireCitations().
			WithPresentationDepth(PresentationDepthStandard).
			WithContext()

	case ShapeProcedural:
		builder.
			PreserveOrder().
			RequireNumberedList().
			RequireCitations().
			WithPresentationDepth(PresentationDepthStandard).
			WithContext()

	case ShapeFactual:
		builder.
			RequireCitations().
			WithPresentationDepth(PresentationDepthMinimal).
			WithMinSentences(1)

	case ShapeExploratory:
		builder.
			RequireCitations().
			WithPresentationDepth(PresentationDepthDetailed).
			RequireHeadings().
			WithContext()
	}

	// Apply merge settings from inference
	if !result.MergeAllowed {
		builder.DisableMerging()
	}

	// Set token budget based on shape and inferred depth
	budget := TokenBudgetForShape(result.Shape, result.Depth)
	builder.WithMaxTokens(budget)

	return builder.Build()
}

// Validate checks if a contract is internally consistent
func (c *ResponseContract) Validate() error {
	if c.Coverage.MinItems > 0 && c.Coverage.MaxItems > 0 {
		if c.Coverage.MinItems > c.Coverage.MaxItems {
			return fmt.Errorf("min_items (%d) cannot exceed max_items (%d)",
				c.Coverage.MinItems, c.Coverage.MaxItems)
		}
	}

	if c.Structure.RequireNumberedList && c.Structure.RequireBulletList {
		return fmt.Errorf("cannot require both numbered and bullet lists")
	}

	if c.Citation.InlineCitations && c.Citation.EndCitations {
		return fmt.Errorf("cannot require both inline and end citations")
	}

	return nil
}

// ToJSON serializes the contract to JSON
func (c *ResponseContract) ToJSON() ([]byte, error) {
	return json.Marshal(c)
}

// ContractFromJSON deserializes a contract from JSON
func ContractFromJSON(data []byte) (*ResponseContract, error) {
	var contract ResponseContract
	if err := json.Unmarshal(data, &contract); err != nil {
		return nil, err
	}
	return &contract, nil
}

// ToPromptInstructions converts the contract to prompt instructions
func (c *ResponseContract) ToPromptInstructions() string {
	var instructions []string

	// Shape instruction
	// Balance truthfulness with complete, well-presented answers
	// Allow natural language presentation while staying grounded in sources
	switch c.Shape {
	case ShapeEnumerative:
		instructions = append(instructions, "You are answering an ENUMERATIVE question.")
		instructions = append(instructions, "List each item from the reference units.")
		instructions = append(instructions, "Present items clearly — you may rephrase for readability while preserving meaning.")
	case ShapeExhaustive:
		instructions = append(instructions, "You are answering an EXHAUSTIVE question.")
		instructions = append(instructions, "Include ALL relevant items from the reference units.")
		instructions = append(instructions, "Do NOT omit any items, but you may organize them logically.")
	case ShapeHierarchical:
		instructions = append(instructions, "You are answering a HIERARCHICAL question.")
		instructions = append(instructions, "Maintain the structural relationships from the sources.")
		instructions = append(instructions, "Use headings and subheadings to show hierarchy clearly.")
	case ShapeComparative:
		instructions = append(instructions, "You are answering a COMPARATIVE question.")
		instructions = append(instructions, "Present balanced information for each option being compared.")
		instructions = append(instructions, "Highlight key differences and similarities.")
	case ShapeProcedural:
		instructions = append(instructions, "You are answering a PROCEDURAL question.")
		instructions = append(instructions, "Provide clear, ordered steps as described in the sources.")
		instructions = append(instructions, "Maintain the correct sequence of operations.")
	case ShapeFactual:
		instructions = append(instructions, "You are answering a FACTUAL question.")
		instructions = append(instructions, "State the specific fact clearly and directly.")
	case ShapeExploratory:
		instructions = append(instructions, "You are answering an EXPLORATORY question.")
		instructions = append(instructions, "Cover key aspects from the sources.")
		instructions = append(instructions, "Organize with sections if the topic has multiple facets.")
	}

	// Coverage instructions - only require covering RELEVANT reference units
	if c.Coverage.MinItems > 0 {
		instructions = append(instructions, fmt.Sprintf("Use reference units that are RELEVANT to answer the question (up to %d units available).", c.Coverage.MinItems))
		instructions = append(instructions, "You may skip reference units that are not relevant to the specific question.")
		instructions = append(instructions, "If NONE of the reference units contain relevant information, output INSUFFICIENT_INPUT. If some reference units contain partial or related information, provide what is available and note what is missing.")
	}
	if c.Coverage.MustCoverAllSUs {
		instructions = append(instructions, "Prioritize covering all provided source units IF they are relevant to the question.")
		instructions = append(instructions, "Skip source units that do not relate to the question being asked.")
	}
	if c.Coverage.NoMerging {
		instructions = append(instructions, "Do NOT merge or combine similar items. Keep each item separate.")
		// Fix 4: Enforce 1 SU → 1 bullet rule
		instructions = append(instructions, "Each bullet must be supported by exactly one Reference Unit.")
		instructions = append(instructions, "Do not combine multiple concepts into a single bullet.")
	}
	if c.Coverage.NoOmitting {
		instructions = append(instructions, "Do NOT omit any items from the sources. Include everything.")
	}

	// Structure instructions
	if c.Structure.RequireNumberedList {
		instructions = append(instructions, "Format your response as a numbered list.")
	}
	if c.Structure.RequireBulletList {
		instructions = append(instructions, "Format your response as a bullet list.")
	}
	if c.Structure.RequireHeadings {
		instructions = append(instructions, "Use section headings to organize your response.")
	}
	if c.Structure.PreserveOrder {
		instructions = append(instructions, "Maintain the order of items as they appear in the sources.")
	}
	if c.Structure.PreserveHierarchy {
		instructions = append(instructions, "Preserve the hierarchical structure from the sources.")
	}

	// Citation instructions
	if c.Citation.CitationRequired {
		instructions = append(instructions, "Cite your sources for each claim or item.")
		if c.Citation.InlineCitations {
			switch c.Citation.CitationFormat {
			case CitationFormatBracket:
				instructions = append(instructions, "Use inline citations in [1], [2] format.")
			case CitationFormatSuperscript:
				instructions = append(instructions, "Use superscript citation markers.")
			default:
				instructions = append(instructions, "Use inline citations.")
			}
		}
	}

	// Format instructions
	if c.Format.MaxLength > 0 {
		instructions = append(instructions, fmt.Sprintf("Keep your response under %d characters.", c.Format.MaxLength))
	}
	if c.Format.MaxTokens > 0 {
		instructions = append(instructions,
			fmt.Sprintf("Target approximately %d tokens in your response. Be concise for simple facts, thorough for complex explanations.", c.Format.MaxTokens))
	}
	if c.Format.Language != "" {
		instructions = append(instructions, fmt.Sprintf("Respond in %s.", c.Format.Language))
	}
	if c.Format.Tone != "" {
		instructions = append(instructions, fmt.Sprintf("Use a %s tone.", c.Format.Tone))
	}
	if c.Format.IncludeSummary {
		if c.Format.SummaryPosition == SummaryAtStart {
			instructions = append(instructions, "Start with a brief summary.")
		} else {
			instructions = append(instructions, "End with a brief summary.")
		}
	}

	// Presentation depth instructions
	switch c.Format.PresentationDepth {
	case PresentationDepthMinimal:
		instructions = append(instructions, "Be concise. State the essential facts only.")
	case PresentationDepthStandard:
		instructions = append(instructions, "Be direct. Include necessary context but avoid unnecessary elaboration.")
	case PresentationDepthDetailed:
		instructions = append(instructions, "Provide a thorough response with supporting context and examples where relevant.")
	}

	// Minimum sentences for non-list responses
	if c.Format.MinSentences > 0 {
		instructions = append(instructions, fmt.Sprintf("Provide at least %d sentences in your response.", c.Format.MinSentences))
	}

	// Context inclusion
	if c.Format.IncludeContext {
		instructions = append(instructions, "Include relevant context from the sources that helps explain or support each point.")
	}

	// Join instructions
	var result strings.Builder
	for i, inst := range instructions {
		if i > 0 {
			result.WriteString("\n")
		}
		result.WriteString(inst)
	}
	return result.String()
}
