package anthropic

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic/internal/anthropicclient"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name     string
		envToken string
		opts     []Option
		wantErr  bool
	}{
		{
			name:     "with token from env",
			envToken: "test-token",
			opts:     []Option{},
			wantErr:  false,
		},
		{
			name:     "with token option",
			envToken: "",
			opts:     []Option{WithToken("test-token")},
			wantErr:  false,
		},
		{
			name:     "missing token",
			envToken: "",
			opts:     []Option{},
			wantErr:  true,
		},
		{
			name:     "with all options",
			envToken: "test-token",
			opts: []Option{
				WithModel("claude-3-opus-20240229"),
				WithBaseURL("https://api.example.com"),
				WithAnthropicBetaHeader("max-tokens-3-5-sonnet-2024-07-15"),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			os.Setenv("ANTHROPIC_API_KEY", tt.envToken)

			llm, err := New(tt.opts...)
			if (err != nil) != tt.wantErr {
				t.Errorf("New() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && llm == nil {
				t.Error("New() returned nil LLM without error")
			}
		})
	}
}

func TestProcessMessages(t *testing.T) {
	tests := []struct {
		name       string
		messages   []llms.MessageContent
		wantLen    int
		wantSystem string
		wantErr    bool
	}{
		{
			name: "basic text message",
			messages: []llms.MessageContent{
				{
					Role: llms.ChatMessageTypeHuman,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "Hello"},
					},
				},
			},
			wantLen:    1,
			wantSystem: "",
			wantErr:    false,
		},
		{
			name: "system message",
			messages: []llms.MessageContent{
				{
					Role: llms.ChatMessageTypeSystem,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "You are helpful"},
					},
				},
				{
					Role: llms.ChatMessageTypeHuman,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "Hi"},
					},
				},
			},
			wantLen:    1,
			wantSystem: "You are helpful",
			wantErr:    false,
		},
		{
			name: "ai and human messages",
			messages: []llms.MessageContent{
				{
					Role: llms.ChatMessageTypeHuman,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "Hello"},
					},
				},
				{
					Role: llms.ChatMessageTypeAI,
					Parts: []llms.ContentPart{
						llms.TextContent{Text: "Hi there!"},
					},
				},
			},
			wantLen:    2,
			wantSystem: "",
			wantErr:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, systemPrompt, err := processMessages(tt.messages)
			if (err != nil) != tt.wantErr {
				t.Errorf("processMessages() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if len(result) != tt.wantLen {
					t.Errorf("processMessages() returned %d messages, want %d", len(result), tt.wantLen)
				}
				if systemPrompt != tt.wantSystem {
					t.Errorf("processMessages() system prompt = %q, want %q", systemPrompt, tt.wantSystem)
				}
			}
		})
	}
}

func TestToolsToTools(t *testing.T) {
	tools := []llms.Tool{
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        "get_weather",
				Description: "Get the weather for a location",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{
							"type":        "string",
							"description": "The location to get weather for",
						},
					},
					"required": []string{"location"},
				},
			},
		},
	}

	result := toolsToTools(tools)

	if len(result) != 1 {
		t.Fatalf("toolsToTools() returned %d tools, want 1", len(result))
	}
	if result[0].Name != "get_weather" {
		t.Errorf("toolsToTools() tool name = %q, want %q", result[0].Name, "get_weather")
	}
	if result[0].Description != "Get the weather for a location" {
		t.Errorf("toolsToTools() tool description = %q, want %q", result[0].Description, "Get the weather for a location")
	}
}

func TestOptions(t *testing.T) {
	t.Run("WithModel", func(t *testing.T) {
		opts := &options{}
		WithModel("claude-3-opus")(opts)
		if opts.model != "claude-3-opus" {
			t.Errorf("WithModel() got %s, want claude-3-opus", opts.model)
		}
	})

	t.Run("WithToken", func(t *testing.T) {
		opts := &options{}
		WithToken("test-token")(opts)
		if opts.token != "test-token" {
			t.Errorf("WithToken() got %s, want test-token", opts.token)
		}
	})

	t.Run("WithBaseURL", func(t *testing.T) {
		opts := &options{}
		WithBaseURL("https://test.com")(opts)
		if opts.baseURL != "https://test.com" {
			t.Errorf("WithBaseURL() got %s, want https://test.com", opts.baseURL)
		}
	})

	t.Run("WithAnthropicBetaHeader", func(t *testing.T) {
		opts := &options{}
		WithAnthropicBetaHeader("test-beta")(opts)
		if opts.anthropicBetaHeader != "test-beta" {
			t.Errorf("WithAnthropicBetaHeader() got %s, want test-beta", opts.anthropicBetaHeader)
		}
	})

	t.Run("WithLegacyTextCompletionsAPI", func(t *testing.T) {
		opts := &options{}
		WithLegacyTextCompletionsAPI()(opts)
		if !opts.useLegacyTextCompletionsAPI {
			t.Error("WithLegacyTextCompletionsAPI() did not set flag")
		}
	})
}

func TestCall(t *testing.T) {
	// Test that Call delegates to GenerateContent
	t.Skip("Call() requires integration testing with mock client")
}

func TestGenerateMessagesContent_EmptyContent(t *testing.T) {
	// This test demonstrates the need for checking len(result.Content) == 0
	// Without the fix, accessing result.Content[0] would panic when Anthropic
	// returns a response with nil or empty content (addresses issue #993)
	t.Skip("Requires mock client - would demonstrate panic without len(result.Content) == 0 check")
}

func TestProcessAnthropicResponse_ToolCallsAndThinking(t *testing.T) {
	payload := &anthropicclient.MessageResponsePayload{
		Content: []anthropicclient.Content{
			&anthropicclient.TextContent{Type: "text", Text: "Hello"},
			&anthropicclient.ToolUseContent{
				Type:  "tool_use",
				ID:    "call_1",
				Name:  "get_weather",
				Input: map[string]any{"location": "San Francisco"},
			},
			&anthropicclient.TextContent{Type: "text", Text: " world"},
			&anthropicclient.TextContent{Type: "text", Text: "<thinking>calc</thinking>Answer"},
		},
		StopReason: "tool_use",
	}
	payload.Usage.InputTokens = 12
	payload.Usage.OutputTokens = 4

	resp, err := processAnthropicResponse(payload)
	if err != nil {
		t.Fatalf("processAnthropicResponse() error = %v", err)
	}

	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}

	choice := resp.Choices[0]
	expectedContent := "Hello world<thinking>calc</thinking>Answer"
	if choice.Content != expectedContent {
		t.Errorf("choice.Content = %q, want %q", choice.Content, expectedContent)
	}

	if choice.ReasoningContent != "calc" {
		t.Errorf("choice.ReasoningContent = %q, want %q", choice.ReasoningContent, "calc")
	}

	if len(choice.ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(choice.ToolCalls))
	}

	tc := choice.ToolCalls[0]
	if tc.ID != "call_1" {
		t.Errorf("tool call id = %q, want call_1", tc.ID)
	}
	if tc.FunctionCall == nil || tc.FunctionCall.Name != "get_weather" {
		t.Fatalf("unexpected tool function: %+v", tc.FunctionCall)
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(tc.FunctionCall.Arguments), &args); err != nil {
		t.Fatalf("failed to unmarshal tool call arguments: %v", err)
	}
	if args["location"] != "San Francisco" {
		t.Errorf("tool call arguments location = %v, want San Francisco", args["location"])
	}

	outputContent, ok := choice.GenerationInfo["OutputContent"].(string)
	if !ok {
		t.Fatal("OutputContent missing from generation info")
	}
	if outputContent != "Hello worldAnswer" {
		t.Errorf("OutputContent = %q, want %q", outputContent, "Hello worldAnswer")
	}

	thinkingContent, ok := choice.GenerationInfo["ThinkingContent"].(string)
	if !ok {
		t.Fatal("ThinkingContent missing from generation info")
	}
	if thinkingContent != "calc" {
		t.Errorf("ThinkingContent = %q, want %q", thinkingContent, "calc")
	}

	if choice.FuncCall == nil || choice.FuncCall.Name != "get_weather" {
		t.Fatalf("FuncCall not set correctly: %+v", choice.FuncCall)
	}
}

func TestProcessAnthropicResponse_MetadataWithoutThinking(t *testing.T) {
	payload := &anthropicclient.MessageResponsePayload{
		Content: []anthropicclient.Content{
			&anthropicclient.TextContent{Type: "text", Text: "Hello"},
			&anthropicclient.TextContent{Type: "text", Text: " world"},
		},
		StopReason: "end_turn",
	}

	resp, err := processAnthropicResponse(payload)
	if err != nil {
		t.Fatalf("processAnthropicResponse() error = %v", err)
	}

	choice := resp.Choices[0]
	expectedContent := "Hello world"
	if choice.Content != expectedContent {
		t.Errorf("choice.Content = %q, want %q", choice.Content, expectedContent)
	}

	outputContent, ok := choice.GenerationInfo["OutputContent"].(string)
	if !ok {
		t.Fatal("OutputContent missing from generation info")
	}
	if outputContent != expectedContent {
		t.Errorf("OutputContent = %q, want %q", outputContent, expectedContent)
	}

	thinkingContent, ok := choice.GenerationInfo["ThinkingContent"].(string)
	if !ok {
		t.Fatal("ThinkingContent missing from generation info")
	}
	if thinkingContent != "" {
		t.Errorf("ThinkingContent = %q, want empty string", thinkingContent)
	}
}
