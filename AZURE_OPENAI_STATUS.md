# Azure OpenAI Realtime API Status

## ⚠️ Current Status: **Incompatible**

The openai-agents SDK (v0.4.0) has a **schema incompatibility** with Azure OpenAI's Realtime API.

## Problem Details

### Error
```
ValidationError: 2 validation errors for tagged-union[...]
`session.created`.session.RealtimeSessionCreateRequest.type
  Field required [type=missing, ...]
```

### Root Cause

1. **Azure OpenAI** sends a `session.created` event without a `type` field in the session object
2. **openai-agents SDK** expects a `type` field to distinguish between session types
3. When SDK tries to send `session.type` back to Azure, Azure rejects it as an unknown parameter

### Schema Difference

**OpenAI Agents SDK expects:**
```json
{
  "type": "session.created",
  "session": {
    "type": "session",  // ❌ Required by SDK
    "object": "realtime.session",
    "id": "...",
    ...
  }
}
```

**Azure OpenAI sends:**
```json
{
  "type": "session.created",
  "session": {
    "object": "realtime.session",  // ✅ No type field
    "id": "...",
    ...
  }
}
```

## Recommendation

**Use OpenAI API directly** until the SDK adds proper Azure support.

## Related Issues

- [openai-agents-python #96](https://github.com/openai/openai-agents-python/issues/96) - Azure OpenAI support (closed as not planned)
- [livekit/agents #3489](https://github.com/livekit/agents/issues/3489) - Similar issue with LiveKit agents

## Workaround Options

### Option 1: Use OpenAI API Directly (Recommended)
Set `OPENAI_API_KEY` in your `.env` file.

### Option 2: Wait for SDK Update
Monitor [openai-agents-python releases](https://github.com/openai/openai-agents-python/releases) for Azure support.

### Option 3: Use Low-Level WebSocket Client
Implement Azure Realtime API using raw WebSocket connections:
- See [Azure Realtime API Reference](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/realtime-audio-reference)
- See [Azure Realtime Audio SDK](https://github.com/Azure-Samples/aoai-realtime-audio-sdk)

## Testing

To verify the issue yourself:
1. Set Azure OpenAI environment variables in `.env`
2. Run `uv run python -m app.agent`
3. Observe the validation error when session.created event is received

---

**Last Updated:** 2025-01-22  
**SDK Version:** openai-agents 0.4.0  
**Status:** Incompatible - use OpenAI API directly
