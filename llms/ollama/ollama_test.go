package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/internal/httprr"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama/internal/ollamaclient"
)

func newTestClient(t *testing.T, opts ...Option) *LLM {
	t.Helper()

	// Set up httprr for recording/replaying HTTP interactions
	rr := httprr.OpenForTest(t, http.DefaultTransport)

	// Default model for testing
	ollamaModel := "gemma3:1b"
	if envModel := os.Getenv("OLLAMA_TEST_MODEL"); envModel != "" {
		ollamaModel = envModel
	}

	// Default to localhost
	serverURL := "http://localhost:11434"
	if envURL := os.Getenv("OLLAMA_HOST"); envURL != "" && rr.Recording() {
		serverURL = envURL
	}

	// Skip if no recording exists and we're not recording
	if !rr.Recording() {
		httprr.SkipIfNoCredentialsAndRecordingMissing(t)
	}

	// Always add server URL and HTTP client
	opts = append([]Option{
		WithServerURL(serverURL),
		WithHTTPClient(rr.Client()),
		WithModel(ollamaModel),
	}, opts...)

	c, err := New(opts...)
	require.NoError(t, err)
	return c
}

// newEmbeddingTestClient creates a test client configured for embedding operations
func newEmbeddingTestClient(t *testing.T, opts ...Option) *LLM {
	t.Helper()

	// Default embedding model
	embeddingModel := "nomic-embed-text"
	if envModel := os.Getenv("OLLAMA_EMBEDDING_MODEL"); envModel != "" {
		embeddingModel = envModel
	}

	// Use the embedding model by default
	opts = append([]Option{WithModel(embeddingModel)}, opts...)

	return newTestClient(t, opts...)
}

func TestGenerateContent(t *testing.T) {
	ctx := context.Background()

	llm := newTestClient(t)

	parts := []llms.ContentPart{
		llms.TextContent{Text: "How many feet are in a nautical mile?"},
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	rsp, err := llm.GenerateContent(ctx, content)
	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]
	assert.Regexp(t, "feet", strings.ToLower(c1.Content))
}

func TestWithFormat(t *testing.T) {
	ctx := context.Background()

	llm := newTestClient(t, WithFormat("json"))

	parts := []llms.ContentPart{
		llms.TextContent{Text: "How many feet are in a nautical mile? Respond with JSON containing the answer."},
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	rsp, err := llm.GenerateContent(ctx, content)
	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]

	// check whether we got *any* kind of JSON object.
	var result map[string]any
	err = json.Unmarshal([]byte(c1.Content), &result)
	require.NoError(t, err)
	// The JSON should contain some information about feet or the answer
	assert.NotEmpty(t, result)
}

func TestWithStreaming(t *testing.T) {
	ctx := context.Background()

	llm := newTestClient(t)

	parts := []llms.ContentPart{
		llms.TextContent{Text: "How many feet are in a nautical mile?"},
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	var sb strings.Builder
	rsp, err := llm.GenerateContent(ctx, content,
		llms.WithStreamingFunc(func(_ context.Context, chunk []byte) error {
			sb.Write(chunk)
			return nil
		}))
	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]
	assert.Regexp(t, "feet", strings.ToLower(c1.Content))
	assert.Regexp(t, "feet", strings.ToLower(sb.String()))
}

func TestWithKeepAlive(t *testing.T) {
	ctx := context.Background()

	llm := newTestClient(t, WithKeepAlive("1m"))

	parts := []llms.ContentPart{
		llms.TextContent{Text: "How many feet are in a nautical mile?"},
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	resp, err := llm.GenerateContent(ctx, content)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	c1 := resp.Choices[0]
	assert.Regexp(t, "feet", strings.ToLower(c1.Content))

	// Note: gemma3:1b doesn't support embeddings
	// Use TestCreateEmbedding for embedding tests
}

func TestWithThink(t *testing.T) {
	ctx := context.Background()

	// Test that WithThink option correctly sets the think parameter
	llm := newTestClient(t, WithThink(true))

	parts := []llms.ContentPart{
		llms.TextContent{Text: "What is 2+2? Explain your reasoning step by step."},
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	// The request should include think:true in options
	resp, err := llm.GenerateContent(ctx, content)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	c1 := resp.Choices[0]
	// The response should contain the answer
	assert.Contains(t, strings.ToLower(c1.Content), "4")
}

func TestWithPullModel(t *testing.T) {
	ctx := context.Background()

	// This test verifies the WithPullModel option works correctly.
	// It uses a model that's likely already available locally (gemma3:1b)
	// to avoid expensive downloads during regular test runs.

	// Use newTestClient to get httprr support
	llm := newTestClient(t, WithPullModel())

	parts := []llms.ContentPart{
		llms.TextContent{Text: "Say hello"},
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	// The model should be pulled automatically before generating content
	resp, err := llm.GenerateContent(ctx, content)
	require.NoError(t, err)

	assert.NotEmpty(t, resp.Choices)
	c1 := resp.Choices[0]
	assert.NotEmpty(t, c1.Content)
}

func TestCreateEmbedding(t *testing.T) {
	ctx := context.Background()

	// Use the embedding-specific test client
	llm := newEmbeddingTestClient(t)

	// Test single embedding
	embeddings, err := llm.CreateEmbedding(ctx, []string{"Hello, world!"})

	// Skip if the model is not found
	if err != nil && strings.Contains(err.Error(), "model") && strings.Contains(err.Error(), "not found") {
		t.Skipf("Embedding model not found: %v. Try running 'ollama pull nomic-embed-text' first", err)
	}

	require.NoError(t, err)
	assert.Len(t, embeddings, 1)
	assert.NotEmpty(t, embeddings[0])

	// Test multiple embeddings
	texts := []string{
		"The quick brown fox jumps over the lazy dog",
		"Machine learning is a subset of artificial intelligence",
		"Ollama makes it easy to run large language models locally",
	}
	embeddings, err = llm.CreateEmbedding(ctx, texts)
	require.NoError(t, err)
	assert.Len(t, embeddings, len(texts))
	for i, emb := range embeddings {
		assert.NotEmpty(t, emb, "Embedding %d should not be empty", i)
	}
}

func TestWithPullTimeout(t *testing.T) {
	ctx := context.Background()

	if testing.Short() {
		t.Skip("Skipping pull timeout test in short mode")
	}

	// Check if we're recording - timeout tests don't work with replay
	rr := httprr.OpenForTest(t, http.DefaultTransport)
	defer rr.Close()
	if rr.Replaying() {
		t.Skip("Skipping pull timeout test when not recording (timeout behavior cannot be replayed)")
	}

	// Use a very short timeout that should fail for any real model pull
	llm := newTestClient(t,
		WithModel("llama2:70b"), // Large model that would take time to download
		WithPullModel(),
		WithPullTimeout(50*time.Millisecond), // Extremely short timeout
	)

	parts := []llms.ContentPart{
		llms.TextContent{Text: "Say hello"},
	}
	content := []llms.MessageContent{
		{
			Role:  llms.ChatMessageTypeHuman,
			Parts: parts,
		},
	}

	// This should fail with a timeout error
	_, err := llm.GenerateContent(ctx, content)

	if err == nil {
		t.Fatal("Expected error due to pull timeout, but got none")
	}
	if !strings.Contains(err.Error(), "deadline exceeded") {
		t.Fatalf("Expected timeout error, got: %v", err)
	}
}

func TestToolCalling(t *testing.T) {
	ctx := context.Background()

	llm := newTestClient(t, WithModel("llama3.2:1b"))

	weatherTool := llms.Tool{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "get_current_weather",
			Description: "Get the current weather for a location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The location to get the weather for, e.g. San Francisco, CA",
					},
					"format": map[string]any{
						"type":        "string",
						"description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
						"enum":        []string{"celsius", "fahrenheit"},
					},
				},
				"required": []string{"location", "format"},
			},
		},
	}

	content := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextContent{Text: "What is the weather today in Toronto?"},
			},
		},
	}

	rsp, err := llm.GenerateContent(ctx, content, llms.WithTools([]llms.Tool{weatherTool}))
	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]

	// Check if tool calls are present
	if len(c1.ToolCalls) > 0 {
		t.Logf("Tool calls detected: %+v", c1.ToolCalls)
		toolCall := c1.ToolCalls[0]
		assert.Equal(t, "function", toolCall.Type)
		assert.NotNil(t, toolCall.FunctionCall)
		assert.Equal(t, "get_current_weather", toolCall.FunctionCall.Name)

		// Parse arguments to verify structure
		var args map[string]any
		err := json.Unmarshal([]byte(toolCall.FunctionCall.Arguments), &args)
		require.NoError(t, err)
		assert.Contains(t, args, "location")
		t.Logf("Tool call arguments: %+v", args)
	} else {
		t.Log("No tool calls returned (this may be expected if model doesn't support tools)")
	}
}

func TestToolCallingStreaming(t *testing.T) {
	ctx := context.Background()

	// Use a model that supports tool calling
	llm := newTestClient(t, WithModel("llama3.2:1b"))

	// Define a weather tool
	weatherTool := llms.Tool{
		Type: "function",
		Function: &llms.FunctionDefinition{
			Name:        "get_current_weather",
			Description: "Get the current weather for a location",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "The location to get the weather for",
					},
				},
				"required": []string{"location"},
			},
		},
	}

	content := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.TextContent{Text: "What is the weather in Paris?"},
			},
		},
	}

	var streamedContent strings.Builder
	streamFunc := func(ctx context.Context, chunk []byte) error {
		streamedContent.Write(chunk)
		return nil
	}

	rsp, err := llm.GenerateContent(ctx, content,
		llms.WithTools([]llms.Tool{weatherTool}),
		llms.WithStreamingFunc(streamFunc),
	)
	require.NoError(t, err)

	assert.NotEmpty(t, rsp.Choices)
	c1 := rsp.Choices[0]

	// Check if tool calls are present
	if len(c1.ToolCalls) > 0 {
		t.Logf("Tool calls detected in streaming: %+v", c1.ToolCalls)
		toolCall := c1.ToolCalls[0]
		assert.Equal(t, "function", toolCall.Type)
		assert.NotNil(t, toolCall.FunctionCall)
		assert.Equal(t, "get_current_weather", toolCall.FunctionCall.Name)
	} else {
		t.Log("No tool calls returned in streaming (this may be expected)")
	}

	t.Logf("Streamed content: %s", streamedContent.String())
}

// TestToolCallTypes verifies the structure of tool call types in ollamaclient
func TestToolCallTypes(t *testing.T) {
	// Test ToolCall structure
	toolCall := ollamaclient.ToolCall{
		Function: ollamaclient.ToolFunction{
			Name: "test_function",
			Arguments: map[string]any{
				"arg1": "value1",
				"arg2": 42,
			},
		},
	}

	assert.Equal(t, "test_function", toolCall.Function.Name)
	assert.Equal(t, "value1", toolCall.Function.Arguments["arg1"])
	assert.Equal(t, 42, toolCall.Function.Arguments["arg2"])

	// Test Tool structure
	tool := ollamaclient.Tool{
		Type: "function",
		Function: ollamaclient.FunctionTool{
			Name:        "my_function",
			Description: "A test function",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"param1": map[string]any{
						"type": "string",
					},
				},
			},
		},
	}

	assert.Equal(t, "function", tool.Type)
	assert.Equal(t, "my_function", tool.Function.Name)
	assert.Equal(t, "A test function", tool.Function.Description)
	assert.NotNil(t, tool.Function.Parameters)
}
