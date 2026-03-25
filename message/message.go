// Package message provides shared message types used across goragtoolkit packages.
package message

// Message represents a chat message used throughout the toolkit.
// This is the canonical message type — individual packages accept this type
// or define type aliases to it for backward compatibility.
type Message struct {
	// Role is the message role (e.g., "system", "user", "assistant", "tool")
	Role string `json:"role"`

	// Content is the message text content
	Content string `json:"content"`

	// Name is an optional sender name (for function/tool messages)
	Name string `json:"name,omitempty"`

	// ToolCallID is an optional tool call identifier (for tool result messages)
	ToolCallID string `json:"tool_call_id,omitempty"`
}
