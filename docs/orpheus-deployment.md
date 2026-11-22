# Orpheus to Home Assistant

First complete tts-training.md: working /health and /stream, trained/exported
weights, your selected-model.json, isolated vLLM runtime and manual activation.
Do not train or download a model merely by starting the Wyoming adapter.

1. Review scripts/install-orpheus-adapter.sh. It creates the lightweight adapter
   environment using voice/orpheus-adapter-requirements.lock.txt and installs
   voice/wyoming-orpheus.service. Update user paths and selection file first.
2. Run the adapter CPU tests and scripts/probe-orpheus.py. A protocol handshake
   is distinct from a complete audible synthesis test.
3. Start the desktop SSH forward with Start-EchoMuse-Whisper.ps1 -IncludeOrpheus.
   Pi loopback 10200 forwards to the desktop adapter. Keep 8812 and 10200 private.
4. Add Wyoming at 127.0.0.1:10200 in HA, select Orpheus TTS/English and the
   advertised voice ID in the preferred assistant. Keep Whisper as STT.
5. Explicitly load Orpheus with its reviewed CLI/desktop start control. Test a
   short and long spoken response, cancellation and end-of-utterance flushing.
   Confirm the Dot actually speaks all final words, not merely that HA shows done.
6. Stop the GPU model when idle if desired. The adapter and CPU Whisper may stay
   up. Stop GPU inference before any training reservation.

The public distribution preserves the committed adapter snapshot. Local
uncommitted experiments from another installation were deliberately not copied.
