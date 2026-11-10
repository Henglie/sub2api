package handler

import (
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
)

func validateOpenAIRequestBody(t *testing.T, body []byte) (bool, *httptest.ResponseRecorder) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	h := &OpenAIGatewayHandler{}
	return h.validateFunctionCallOutputRequest(c, body, zap.NewNop()), w
}

func TestNormalizeCodexNamespacedCallOutputCurlShape(t *testing.T) {
	body := []byte(`{"model":"gpt-5","input":[{"type":"function_call_output","id":"fco_xxx","name":"codex_apps_open_page","namespace":"agent","output":"x"}]}`)

	valid, w := validateOpenAIRequestBody(t, body)
	require.False(t, valid)
	require.Equal(t, 400, w.Code)
	require.Contains(t, w.Body.String(), "function_call_output requires call_id")

	got := normalizeOpenAIResponsesToolContext(body, zap.NewNop())
	require.NotEqual(t, body, got)
	require.Equal(t, "message", gjson.GetBytes(got, "input.0.type").String())
	require.Equal(t, "user", gjson.GetBytes(got, "input.0.role").String())
	require.Equal(t, "x", gjson.GetBytes(got, "input.0.content.0.text").String())
	require.False(t, gjson.GetBytes(got, "input.0.id").Exists())
	require.False(t, gjson.GetBytes(got, "input.0.call_id").Exists())

	valid, w = validateOpenAIRequestBody(t, got)
	require.True(t, valid)
	require.Zero(t, w.Body.Len())

	again := normalizeOpenAIResponsesToolContext(got, zap.NewNop())
	require.Equal(t, got, again)
}

func TestNormalizeCodexNamespacedCallOutputAcceptedShapes(t *testing.T) {
	tests := []struct {
		name string
		item string
	}{
		{name: "missing call id", item: `{"type":"function_call_output","id":"fco_1","name":"codex_apps_open_page","namespace":"agent","output":"x"}`},
		{name: "empty call id", item: `{"type":"function_call_output","call_id":"","name":"codex_apps_open_page","namespace":"agent","output":"x"}`},
		{name: "whitespace call id", item: `{"type":"function_call_output","call_id":"  ","name":"codex_apps_open_page","namespace":"agent","output":"x"}`},
		{name: "null call id", item: `{"type":"function_call_output","call_id":null,"name":"codex_apps_open_page","namespace":"agent","output":"x"}`},
		{name: "agent namespace other tool", item: `{"type":"function_call_output","name":"browser_click","namespace":"agent","output":"x"}`},
		{name: "codex_apps prefix other namespace", item: `{"type":"function_call_output","name":"codex_apps_selected_page_part_1_of_1","namespace":"custom","output":"x"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"model":"gpt-5","input":[` + tt.item + `]}`)
			got, changed := normalizeCodexNamespacedCallOutput(body)
			require.True(t, changed)
			require.Equal(t, "message", gjson.GetBytes(got, "input.0.type").String())
			require.Equal(t, "user", gjson.GetBytes(got, "input.0.role").String())
			require.Equal(t, "x", gjson.GetBytes(got, "input.0.content.0.text").String())
			valid, _ := validateOpenAIRequestBody(t, got)
			require.True(t, valid)
		})
	}
}

func TestNormalizeCodexNamespacedCallOutputRejectsUnrelatedShapes(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "ordinary tool output missing call id", body: `{"model":"gpt-5","input":[{"type":"function_call_output","name":"shell","output":"x"}]}`},
		{name: "ordinary tool output call id", body: `{"model":"gpt-5","input":[{"type":"function_call_output","call_id":"call_1","name":"codex_apps_open_page","namespace":"agent","output":"x"}]}`},
		{name: "non string call id", body: `{"model":"gpt-5","input":[{"type":"function_call_output","call_id":42,"name":"codex_apps_open_page","namespace":"agent","output":"x"}]}`},
		{name: "missing namespace", body: `{"model":"gpt-5","input":[{"type":"function_call_output","name":"codex_apps_open_page","output":"x"}]}`},
		{name: "empty name", body: `{"model":"gpt-5","input":[{"type":"function_call_output","name":"","namespace":"agent","output":"x"}]}`},
		{name: "non string output", body: `{"model":"gpt-5","input":[{"type":"function_call_output","name":"codex_apps_open_page","namespace":"agent","output":{"x":1}}]}`},
		{name: "bootstrap namespace remains", body: `{"model":"gpt-5","input":[{"type":"function_call_output","name":"automation_update","namespace":"codex_app","output":"x"}]}`},
		{name: "padded type", body: `{"model":"gpt-5","input":[{"type":" function_call_output ","name":"codex_apps_open_page","namespace":"agent","output":"x"}]}`},
		{name: "duplicate members", body: `{"model":"gpt-5","input":[{"type":"function_call_output","name":"codex_apps_open_page","namespace":"agent","namespace":"agent","output":"x"}]}`},
		{name: "trailing content", body: `{"model":"gpt-5","input":[{"type":"function_call_output","name":"codex_apps_open_page","namespace":"agent","output":"x"}]} trailing`},
		{name: "string input", body: `{"model":"gpt-5","input":"hello"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(tt.body)
			got, changed := normalizeCodexNamespacedCallOutput(body)
			require.False(t, changed)
			require.Equal(t, body, got)
		})
	}
}

func TestNormalizeCodexNamespacedCallOutputPreservesOrderAndContext(t *testing.T) {
	body := []byte(`{"model":"gpt-5","previous_response_id":"resp_1","metadata":{"integer":9007199254740993},"input":[{"type":"message","role":"user","content":"before"},{"type":"function_call_output","id":"fco_1","name":"codex_apps_open_page","namespace":"agent","output":"snapshot"},{"type":"message","role":"user","content":"after"}]}`)
	got, changed := normalizeCodexNamespacedCallOutput(body)
	require.True(t, changed)
	require.Equal(t, "resp_1", gjson.GetBytes(got, "previous_response_id").String())
	require.Equal(t, "9007199254740993", gjson.GetBytes(got, "metadata.integer").Raw)
	require.Equal(t, "before", gjson.GetBytes(got, "input.0.content").String())
	require.Equal(t, "message", gjson.GetBytes(got, "input.1.type").String())
	require.Equal(t, "snapshot", gjson.GetBytes(got, "input.1.content.0.text").String())
	require.Equal(t, "after", gjson.GetBytes(got, "input.2.content").String())
}

func TestNormalizeCodexNamespacedCallOutputStillRejectsOrdinaryMissingCallID(t *testing.T) {
	body := []byte(`{"model":"gpt-5","input":[{"type":"function_call_output","name":"codex_apps_open_page","namespace":"agent","output":"snapshot"},{"type":"function_call_output","name":"shell","output":"x"}]}`)
	got, changed := normalizeCodexNamespacedCallOutput(body)
	require.True(t, changed)
	valid, w := validateOpenAIRequestBody(t, got)
	require.False(t, valid)
	require.Equal(t, 400, w.Code)
	require.Contains(t, w.Body.String(), "function_call_output requires call_id")
}

func TestNormalizeCodexNamespacedCallOutputCoexistsWithAutomationBootstrap(t *testing.T) {
	output := codexAutomationBootstrap("wiki", "never", automationBootstrapPrompt)
	body := []byte(`{"model":"gpt-5","input":[{"type":"function_call_output","name":"codex_apps_open_page","namespace":"agent","output":"snapshot"},{"type":"function_call_output","name":"automation_update","namespace":"codex_app","output":` + mustJSON(t, output) + `}]}`)
	got, changed := normalizeCodexNamespacedCallOutput(body)
	require.True(t, changed)
	got, changed = normalizeCodexAutomationBootstrap(got)
	require.True(t, changed)
	valid, w := validateOpenAIRequestBody(t, got)
	require.True(t, valid)
	require.Zero(t, w.Body.Len())
}
