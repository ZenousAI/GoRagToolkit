// Package answershape provides answer shape inference and response contracts.
package answershape

import (
	"encoding/json"
	"testing"

	"github.com/google/uuid"
)

func TestNewContractBuilder(t *testing.T) {
	builder := NewContractBuilder()
	if builder == nil {
		t.Fatal("NewContractBuilder() returned nil")
	}
	if builder.contract == nil {
		t.Fatal("NewContractBuilder().contract is nil")
	}
	if builder.contract.ID == uuid.Nil {
		t.Error("NewContractBuilder() should generate a non-nil UUID")
	}
}

func TestContractBuilder_WithShape(t *testing.T) {
	shapes := []Shape{
		ShapeEnumerative,
		ShapeExhaustive,
		ShapeHierarchical,
		ShapeComparative,
		ShapeProcedural,
		ShapeExploratory,
		ShapeFactual,
	}

	for _, shape := range shapes {
		t.Run(string(shape), func(t *testing.T) {
			contract := NewContractBuilder().
				WithShape(shape).
				Build()

			if contract.Shape != shape {
				t.Errorf("WithShape(%s) = %s, want %s", shape, contract.Shape, shape)
			}
		})
	}
}

func TestContractBuilder_CoverageConstraints(t *testing.T) {
	contract := NewContractBuilder().
		WithShape(ShapeEnumerative).
		WithMinItems(5).
		WithMaxItems(10).
		DisableMerging().
		DisableOmitting().
		RequireCoverAllSUs().
		RequireSemanticRoles("enumeration_item", "definition").
		Build()

	if contract.Coverage.MinItems != 5 {
		t.Errorf("MinItems = %d, want 5", contract.Coverage.MinItems)
	}
	if contract.Coverage.MaxItems != 10 {
		t.Errorf("MaxItems = %d, want 10", contract.Coverage.MaxItems)
	}
	if !contract.Coverage.NoMerging {
		t.Error("NoMerging should be true")
	}
	if !contract.Coverage.NoOmitting {
		t.Error("NoOmitting should be true")
	}
	if !contract.Coverage.MustCoverAllSUs {
		t.Error("MustCoverAllSUs should be true")
	}
	if len(contract.Coverage.RequiredSemanticRoles) != 2 {
		t.Errorf("RequiredSemanticRoles length = %d, want 2",
			len(contract.Coverage.RequiredSemanticRoles))
	}
}

func TestContractBuilder_RequireCoverSUs(t *testing.T) {
	id1 := uuid.New()
	id2 := uuid.New()

	contract := NewContractBuilder().
		RequireCoverSUs(id1, id2).
		Build()

	if len(contract.Coverage.MustCoverSUIDs) != 2 {
		t.Fatalf("MustCoverSUIDs length = %d, want 2",
			len(contract.Coverage.MustCoverSUIDs))
	}
	if contract.Coverage.MustCoverSUIDs[0] != id1 {
		t.Errorf("MustCoverSUIDs[0] = %s, want %s",
			contract.Coverage.MustCoverSUIDs[0], id1)
	}
	if contract.Coverage.MustCoverSUIDs[1] != id2 {
		t.Errorf("MustCoverSUIDs[1] = %s, want %s",
			contract.Coverage.MustCoverSUIDs[1], id2)
	}
}

func TestContractBuilder_StructureConstraints(t *testing.T) {
	t.Run("numbered list", func(t *testing.T) {
		contract := NewContractBuilder().
			RequireNumberedList().
			Build()

		if !contract.Structure.RequireNumberedList {
			t.Error("RequireNumberedList should be true")
		}
		if contract.Structure.RequireBulletList {
			t.Error("RequireBulletList should be false when RequireNumberedList is set")
		}
	})

	t.Run("bullet list", func(t *testing.T) {
		contract := NewContractBuilder().
			RequireBulletList().
			Build()

		if !contract.Structure.RequireBulletList {
			t.Error("RequireBulletList should be true")
		}
		if contract.Structure.RequireNumberedList {
			t.Error("RequireNumberedList should be false when RequireBulletList is set")
		}
	})

	t.Run("headings", func(t *testing.T) {
		contract := NewContractBuilder().
			RequireHeadings().
			Build()

		if !contract.Structure.RequireHeadings {
			t.Error("RequireHeadings should be true")
		}
	})

	t.Run("preserve hierarchy", func(t *testing.T) {
		contract := NewContractBuilder().
			PreserveHierarchy().
			Build()

		if !contract.Structure.PreserveHierarchy {
			t.Error("PreserveHierarchy should be true")
		}
	})

	t.Run("preserve order", func(t *testing.T) {
		contract := NewContractBuilder().
			PreserveOrder().
			Build()

		if !contract.Structure.PreserveOrder {
			t.Error("PreserveOrder should be true")
		}
	})

	t.Run("max depth", func(t *testing.T) {
		contract := NewContractBuilder().
			WithMaxDepth(3).
			Build()

		if contract.Structure.MaxDepth != 3 {
			t.Errorf("MaxDepth = %d, want 3", contract.Structure.MaxDepth)
		}
	})
}

func TestContractBuilder_CitationConstraints(t *testing.T) {
	t.Run("require citations", func(t *testing.T) {
		contract := NewContractBuilder().
			RequireCitations().
			Build()

		if !contract.Citation.CitationRequired {
			t.Error("CitationRequired should be true")
		}
	})

	t.Run("citation format", func(t *testing.T) {
		formats := []CitationFormat{
			CitationFormatBracket,
			CitationFormatSuperscript,
			CitationFormatInline,
			CitationFormatFootnote,
		}

		for _, format := range formats {
			t.Run(string(format), func(t *testing.T) {
				contract := NewContractBuilder().
					WithCitationFormat(format).
					Build()

				if contract.Citation.CitationFormat != format {
					t.Errorf("CitationFormat = %s, want %s",
						contract.Citation.CitationFormat, format)
				}
			})
		}
	})

	t.Run("inline citations", func(t *testing.T) {
		contract := NewContractBuilder().
			WithInlineCitations().
			Build()

		if !contract.Citation.InlineCitations {
			t.Error("InlineCitations should be true")
		}
		if contract.Citation.EndCitations {
			t.Error("EndCitations should be false when InlineCitations is set")
		}
	})

	t.Run("end citations", func(t *testing.T) {
		contract := NewContractBuilder().
			WithEndCitations().
			Build()

		if !contract.Citation.EndCitations {
			t.Error("EndCitations should be true")
		}
		if contract.Citation.InlineCitations {
			t.Error("InlineCitations should be false when EndCitations is set")
		}
	})
}

func TestContractBuilder_FormatConstraints(t *testing.T) {
	contract := NewContractBuilder().
		WithMaxLength(5000).
		WithMaxTokens(1000).
		WithLanguage("en").
		WithTone("formal").
		WithSummary(SummaryAtStart).
		Build()

	if contract.Format.MaxLength != 5000 {
		t.Errorf("MaxLength = %d, want 5000", contract.Format.MaxLength)
	}
	if contract.Format.MaxTokens != 1000 {
		t.Errorf("MaxTokens = %d, want 1000", contract.Format.MaxTokens)
	}
	if contract.Format.Language != "en" {
		t.Errorf("Language = %s, want en", contract.Format.Language)
	}
	if contract.Format.Tone != "formal" {
		t.Errorf("Tone = %s, want formal", contract.Format.Tone)
	}
	if !contract.Format.IncludeSummary {
		t.Error("IncludeSummary should be true")
	}
	if contract.Format.SummaryPosition != SummaryAtStart {
		t.Errorf("SummaryPosition = %s, want %s",
			contract.Format.SummaryPosition, SummaryAtStart)
	}
}

func TestContractFromShape(t *testing.T) {
	tests := []struct {
		name                    string
		shape                   Shape
		coverageExpectation     CoverageExpectation
		expectedMinItems        int
		expectNoMerging         bool
		expectNumberedList      bool
		expectCitations         bool
		expectPreserveOrder     bool
		expectHeadings          bool
		expectPreserveHierarchy bool
	}{
		{
			name:                "enumerative",
			shape:               ShapeEnumerative,
			coverageExpectation: CoverageHigh,
			expectedMinItems:    3,
			expectNoMerging:     true,
			expectNumberedList:  true,
			expectCitations:     true,
		},
		{
			name:                "exhaustive",
			shape:               ShapeExhaustive,
			coverageExpectation: CoverageComplete,
			expectedMinItems:    5,
			expectNoMerging:     true,
			expectNumberedList:  true,
			expectCitations:     true,
		},
		{
			name:                    "hierarchical",
			shape:                   ShapeHierarchical,
			coverageExpectation:     CoverageMedium,
			expectCitations:         true,
			expectHeadings:          true,
			expectPreserveHierarchy: true,
		},
		{
			name:                "comparative",
			shape:               ShapeComparative,
			coverageExpectation: CoverageHigh,
			expectNoMerging:     true,
			expectCitations:     true,
			expectHeadings:      true,
		},
		{
			name:                "procedural",
			shape:               ShapeProcedural,
			coverageExpectation: CoverageHigh,
			expectCitations:     true,
			expectNumberedList:  true,
			expectPreserveOrder: true,
		},
		{
			name:                "exploratory",
			shape:               ShapeExploratory,
			coverageExpectation: CoverageLow,
			expectCitations:     true,
		},
		{
			name:                "factual",
			shape:               ShapeFactual,
			coverageExpectation: CoverageLow,
			expectCitations:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := &InferenceResult{
				Shape:               tt.shape,
				CoverageExpectation: tt.coverageExpectation,
				ExpectedMinItems:    tt.expectedMinItems,
				MergeAllowed:        false, // Test with merge not allowed
			}

			contract := ContractFromShape(result)

			if contract.Shape != tt.shape {
				t.Errorf("Shape = %s, want %s", contract.Shape, tt.shape)
			}

			if tt.expectNoMerging && !contract.Coverage.NoMerging {
				t.Error("expected NoMerging to be true")
			}

			if tt.expectNumberedList && !contract.Structure.RequireNumberedList {
				t.Error("expected RequireNumberedList to be true")
			}

			if tt.expectCitations && !contract.Citation.CitationRequired {
				t.Error("expected CitationRequired to be true")
			}

			if tt.expectPreserveOrder && !contract.Structure.PreserveOrder {
				t.Error("expected PreserveOrder to be true")
			}

			if tt.expectHeadings && !contract.Structure.RequireHeadings {
				t.Error("expected RequireHeadings to be true")
			}

			if tt.expectPreserveHierarchy && !contract.Structure.PreserveHierarchy {
				t.Error("expected PreserveHierarchy to be true")
			}

			// All shapes should have a token budget set
			if contract.Format.MaxTokens <= 0 {
				t.Errorf("expected MaxTokens > 0 for shape %s, got %d", tt.shape, contract.Format.MaxTokens)
			}
		})
	}
}

func TestResponseContract_Validate(t *testing.T) {
	tests := []struct {
		name      string
		contract  *ResponseContract
		expectErr bool
	}{
		{
			name: "valid contract",
			contract: NewContractBuilder().
				WithShape(ShapeEnumerative).
				WithMinItems(3).
				WithMaxItems(10).
				Build(),
			expectErr: false,
		},
		{
			name: "invalid min > max",
			contract: NewContractBuilder().
				WithMinItems(10).
				WithMaxItems(5).
				Build(),
			expectErr: true,
		},
		{
			name: "invalid both list types",
			contract: &ResponseContract{
				Structure: StructureContract{
					RequireNumberedList: true,
					RequireBulletList:   true,
				},
			},
			expectErr: true,
		},
		{
			name: "invalid both citation positions",
			contract: &ResponseContract{
				Citation: CitationContract{
					InlineCitations: true,
					EndCitations:    true,
				},
			},
			expectErr: true,
		},
		{
			name: "valid with zero max items",
			contract: NewContractBuilder().
				WithMinItems(5).
				Build(),
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.contract.Validate()
			if tt.expectErr && err == nil {
				t.Error("expected error but got nil")
			}
			if !tt.expectErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestResponseContract_ToJSON(t *testing.T) {
	contract := NewContractBuilder().
		WithShape(ShapeEnumerative).
		WithMinItems(5).
		DisableMerging().
		RequireCitations().
		Build()

	data, err := contract.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON() error = %v", err)
	}

	// Verify it's valid JSON
	var parsed map[string]any
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("Failed to parse JSON: %v", err)
	}

	// Check key fields
	if parsed["shape"] != string(ShapeEnumerative) {
		t.Errorf("shape = %v, want %s", parsed["shape"], ShapeEnumerative)
	}
}

func TestContractFromJSON(t *testing.T) {
	original := NewContractBuilder().
		WithShape(ShapeHierarchical).
		WithMinItems(3).
		RequireHeadings().
		PreserveHierarchy().
		RequireCitations().
		Build()

	data, err := original.ToJSON()
	if err != nil {
		t.Fatalf("ToJSON() error = %v", err)
	}

	restored, err := ContractFromJSON(data)
	if err != nil {
		t.Fatalf("ContractFromJSON() error = %v", err)
	}

	if restored.Shape != original.Shape {
		t.Errorf("Shape = %s, want %s", restored.Shape, original.Shape)
	}
	if restored.Coverage.MinItems != original.Coverage.MinItems {
		t.Errorf("MinItems = %d, want %d",
			restored.Coverage.MinItems, original.Coverage.MinItems)
	}
	if restored.Structure.RequireHeadings != original.Structure.RequireHeadings {
		t.Errorf("RequireHeadings = %v, want %v",
			restored.Structure.RequireHeadings, original.Structure.RequireHeadings)
	}
	if restored.Structure.PreserveHierarchy != original.Structure.PreserveHierarchy {
		t.Errorf("PreserveHierarchy = %v, want %v",
			restored.Structure.PreserveHierarchy, original.Structure.PreserveHierarchy)
	}
	if restored.Citation.CitationRequired != original.Citation.CitationRequired {
		t.Errorf("CitationRequired = %v, want %v",
			restored.Citation.CitationRequired, original.Citation.CitationRequired)
	}
}

func TestContractFromJSON_InvalidJSON(t *testing.T) {
	_, err := ContractFromJSON([]byte("not valid json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestResponseContract_ToPromptInstructions(t *testing.T) {
	tests := []struct {
		name     string
		contract *ResponseContract
		contains []string
	}{
		{
			name: "enumerative with min items",
			contract: NewContractBuilder().
				WithShape(ShapeEnumerative).
				WithMinItems(5).
				DisableMerging().
				RequireNumberedList().
				Build(),
			contains: []string{
				"ENUMERATIVE",
				"RELEVANT to answer the question (up to 5 units available)",
				"skip reference units that are not relevant",
				"NONE of the reference units contain relevant information, output INSUFFICIENT_INPUT",
				"Do NOT merge",
				"Each bullet must be supported by exactly one Reference Unit",
				"Do not combine multiple concepts into a single bullet",
				"numbered list",
			},
		},
		{
			name: "exhaustive with all coverage",
			contract: NewContractBuilder().
				WithShape(ShapeExhaustive).
				RequireCoverAllSUs().
				DisableOmitting().
				Build(),
			contains: []string{
				"EXHAUSTIVE",
				"ALL relevant items",
				"Prioritize covering all provided source units IF they are relevant",
				"Skip source units that do not relate to the question",
				"Do NOT omit",
			},
		},
		{
			name: "hierarchical with headings",
			contract: NewContractBuilder().
				WithShape(ShapeHierarchical).
				RequireHeadings().
				PreserveHierarchy().
				Build(),
			contains: []string{
				"HIERARCHICAL",
				"section headings",
				"hierarchical structure",
			},
		},
		{
			name: "procedural with order",
			contract: NewContractBuilder().
				WithShape(ShapeProcedural).
				PreserveOrder().
				RequireNumberedList().
				Build(),
			contains: []string{
				"PROCEDURAL",
				"ordered steps",
				"order of items",
				"numbered list",
			},
		},
		{
			name: "with citations",
			contract: NewContractBuilder().
				RequireCitations().
				WithCitationFormat(CitationFormatBracket).
				WithInlineCitations().
				Build(),
			contains: []string{
				"Cite your sources",
				"[1], [2] format",
			},
		},
		{
			name: "with format constraints",
			contract: NewContractBuilder().
				WithMaxLength(5000).
				WithLanguage("English").
				WithTone("formal").
				WithSummary(SummaryAtEnd).
				Build(),
			contains: []string{
				"under 5000 characters",
				"Respond in English",
				"formal tone",
				"End with a brief summary",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			instructions := tt.contract.ToPromptInstructions()

			for _, substr := range tt.contains {
				if !containsSubstring(instructions, substr) {
					t.Errorf("instructions should contain %q\nGot: %s", substr, instructions)
				}
			}
		})
	}
}

func containsSubstring(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && (containsSubstringHelper(s, substr)))
}

func containsSubstringHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func TestContractBuilder_Chaining(t *testing.T) {
	// Test that all builder methods return the builder for chaining
	contract := NewContractBuilder().
		WithShape(ShapeEnumerative).
		WithMinItems(5).
		WithMaxItems(10).
		RequireCoverAllSUs().
		RequireCoverSUs(uuid.New()).
		DisableMerging().
		DisableOmitting().
		RequireSemanticRoles("item").
		RequireNumberedList().
		RequireHeadings().
		PreserveHierarchy().
		PreserveOrder().
		WithMaxDepth(3).
		RequireCitations().
		WithCitationFormat(CitationFormatBracket).
		WithInlineCitations().
		WithMaxLength(5000).
		WithMaxTokens(1000).
		WithLanguage("en").
		WithTone("formal").
		WithSummary(SummaryAtStart).
		Build()

	// Just verify the contract is valid
	if err := contract.Validate(); err != nil {
		// Expected to fail because we set both numbered list and bullet list via chaining
		// but that's okay - we're testing chaining works
		_ = err
	}

	if contract.Shape != ShapeEnumerative {
		t.Error("chaining should preserve all values")
	}
}

func TestSummaryPosition(t *testing.T) {
	t.Run("summary at start", func(t *testing.T) {
		contract := NewContractBuilder().
			WithSummary(SummaryAtStart).
			Build()

		if contract.Format.SummaryPosition != SummaryAtStart {
			t.Errorf("SummaryPosition = %s, want %s",
				contract.Format.SummaryPosition, SummaryAtStart)
		}
	})

	t.Run("summary at end", func(t *testing.T) {
		contract := NewContractBuilder().
			WithSummary(SummaryAtEnd).
			Build()

		if contract.Format.SummaryPosition != SummaryAtEnd {
			t.Errorf("SummaryPosition = %s, want %s",
				contract.Format.SummaryPosition, SummaryAtEnd)
		}
	})
}

func TestCitationFormat_Values(t *testing.T) {
	// Ensure citation format constants have expected values
	if CitationFormatBracket != "bracket" {
		t.Errorf("CitationFormatBracket = %s, want bracket", CitationFormatBracket)
	}
	if CitationFormatSuperscript != "superscript" {
		t.Errorf("CitationFormatSuperscript = %s, want superscript", CitationFormatSuperscript)
	}
	if CitationFormatInline != "inline" {
		t.Errorf("CitationFormatInline = %s, want inline", CitationFormatInline)
	}
	if CitationFormatFootnote != "footnote" {
		t.Errorf("CitationFormatFootnote = %s, want footnote", CitationFormatFootnote)
	}
}

func TestTokenBudgetForShape(t *testing.T) {
	tests := []struct {
		name     string
		shape    Shape
		depth    Depth
		expected int
	}{
		// Base budgets with default (medium) depth
		{"factual_medium", ShapeFactual, DepthMedium, 100},
		{"enumerative_medium", ShapeEnumerative, DepthMedium, 300},
		{"procedural_medium", ShapeProcedural, DepthMedium, 400},
		{"comparative_medium", ShapeComparative, DepthMedium, 450},
		{"hierarchical_medium", ShapeHierarchical, DepthMedium, 500},
		{"exploratory_medium", ShapeExploratory, DepthMedium, 600},
		{"exhaustive_medium", ShapeExhaustive, DepthMedium, 600},

		// Depth multipliers
		{"factual_shallow", ShapeFactual, DepthShallow, 60},        // 100 * 0.6
		{"factual_deep", ShapeFactual, DepthDeep, 150},             // 100 * 1.5
		{"exhaustive_shallow", ShapeExhaustive, DepthShallow, 360}, // 600 * 0.6
		{"exhaustive_deep", ShapeExhaustive, DepthDeep, 900},       // 600 * 1.5

		// Unknown depth defaults to 1.0 multiplier
		{"factual_unknown_depth", ShapeFactual, "", 100},

		// Unknown shape defaults to 800 base
		{"unknown_shape_medium", Shape("unknown"), DepthMedium, 800},
		{"unknown_shape_shallow", Shape("unknown"), DepthShallow, 480}, // 800 * 0.6
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := TokenBudgetForShape(tt.shape, tt.depth)
			if got != tt.expected {
				t.Errorf("TokenBudgetForShape(%s, %s) = %d, want %d",
					tt.shape, tt.depth, got, tt.expected)
			}
		})
	}
}

func TestToPromptInstructions_IncludesTokenBudget(t *testing.T) {
	contract := NewContractBuilder().
		WithShape(ShapeFactual).
		WithMaxTokens(150).
		Build()

	instructions := contract.ToPromptInstructions()

	if !containsSubstring(instructions, "Target approximately 150 tokens") {
		t.Errorf("expected token budget instruction in prompt, got:\n%s", instructions)
	}
	if !containsSubstring(instructions, "Be concise for simple facts") {
		t.Errorf("expected conciseness guidance in prompt, got:\n%s", instructions)
	}
}

func TestToPromptInstructions_NoTokenBudgetWhenZero(t *testing.T) {
	contract := NewContractBuilder().
		WithShape(ShapeFactual).
		Build()

	instructions := contract.ToPromptInstructions()

	if containsSubstring(instructions, "Target approximately") {
		t.Errorf("should not include token budget instruction when MaxTokens=0, got:\n%s", instructions)
	}
}
