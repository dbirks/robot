# Azure OpenAI Realtime API Status

## ✅ Current Status: **Working** (With Correct URL Format)

The openai-agents SDK (v0.4.0) works with Azure OpenAI's Realtime API when using the **GA Protocol** URL format.

## Solution

### ✅ Use GA Protocol URL Format

**Correct (GA Protocol):**
```
wss://{resource}.openai.azure.com/openai/v1/realtime?model={deployment}
```

**Incorrect (Beta Protocol - causes validation error):**
```
wss://{resource}.cognitiveservices.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment={deployment}
```

### Problem with Beta Protocol

When using the Beta Protocol URL format, you'll see this error:
```
ValidationError: 2 validation errors for tagged-union[...]
`session.created`.session.RealtimeSessionCreateRequest.type
  Field required [type=missing, ...]
```

**Root Cause:** The Beta Protocol sends session objects without a `type` field that the SDK requires. The GA Protocol includes this field.

## Configuration

Set these in your `.env` file:

```bash
# Use your Azure OpenAI resource endpoint
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# Your deployment name
AZURE_OPENAI_DEPLOYMENT=gpt-realtime-mini

# Your API key
AZURE_OPENAI_API_KEY=your-azure-key-here
```

**Note:** `AZURE_OPENAI_API_VERSION` is NOT needed for GA Protocol.

## Related Issues

- [openai-agents-python #1748](https://github.com/openai/openai-agents-python/issues/1748) - Runtime error with Azure Realtime API (**SOLVED**)
- [openai-agents-python #96](https://github.com/openai/openai-agents-python/issues/96) - Azure OpenAI support (closed as not planned)
- [livekit/agents #3489](https://github.com/livekit/agents/issues/3489) - Similar issue with LiveKit agents

## Testing

To verify the issue yourself:
1. Set Azure OpenAI environment variables in `.env`
2. Run `uv run python -m app.agent`
3. Observe the validation error when session.created event is received

---

**Last Updated:** 2025-01-22
**SDK Version:** openai-agents 0.4.0
**Status:** ✅ Working with GA Protocol URL format
**Subscribe to:** [Issue #1748](https://github.com/openai/openai-agents-python/issues/1748)
