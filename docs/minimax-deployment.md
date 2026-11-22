# MiniMax, tools, search and forecasts

1. Install the reviewed Home-LLM integration from its official upstream. Review
   its current HA compatibility and pin a source revision for your installation.
2. Create an OpenAI-compatible provider using https://api.minimax.io/v1 and enter
   YOUR API key privately in HA. The example selects MiniMax-M3, streaming,
   1024 output tokens, 90-second timeout and three tool iterations. Check current
   provider model availability/terms; do not assume another model accepts the
   same thinking parameters. This workflow uses {"thinking":{"type":"disabled"}}.
3. Enable Device control and expose only intended entities. Broad administrative
   access is unnecessary. Use unique player/device names and aliases.
4. Adapt llm/samantha-instructions.jinja and the example JSON prompt. Preserve
   tool-before-confirmation, no invented success, concise speech, no web search
   for private household data, and untrusted-search-result handling.
5. Optional web search: install llm/install-search.sh, review search_server.py,
   and run its isolated service. Use Start-EchoMuse-LLM.ps1 -SearchOnly for the
   private MCP loopback tunnel, then configure HA's MCP client. No Ollama model
   is needed for this MiniMax path. Search can fail or return no results.
6. Merge llm/minimax/home-weather-forecast.yaml into your HA scripts.yaml,
   replacing weather.home with YOUR entity. Reload Scripts and expose only
   Get home weather forecast. The tool returns local dates/units/forecasts,
   rather than confusing current weather state with tomorrow's forecast.
7. The included home-llm-json-selector.patch fixes nested JSON values being
   rendered as [object Object] in a text selector. Inspect matching source before
   applying it, back up config_flow.py and restart HA once. Repair an already
   damaged thinking value in the UI to JSON text {"type":"disabled"}; confirm
   a subsequent unchanged reopen/save retains the nested dictionary.
8. For an installation with no rooms, guard _generate_icl_examples with
   if not self.in_context_examples or not all_areas: return [] and keep ICL off.
   Do not apply a patch if your upstream version already fixes it.
9. Select this conversation agent in your preferred Assist pipeline, Whisper
   STT and Orpheus TTS. Test one real exposed device, a forecast and a search.

Home-LLM can cache rendered prompts per conversation. After exposing new tools,
start a fresh conversation or change/save the template to refresh instructions.
API errors should be diagnosed in logs; do not speak raw stack traces as a
normal answer. Public examples have no real HA entry, pipeline or device IDs.
