# Spotify on an EchoMuse Dot

1. Run music/compose.yaml on the same local network as HA and the Dot. It uses
   the official Music Assistant 2.10.13 image, host networking and persistent
   ./data. No privileged container or router forwarding is required.
2. Open http://YOUR_CONTROLLER_IP:8095 and create your own admin account. Audio
   streams normally use port 8097. Keep its database/token state private.
3. Add Spotify under Music Sources using YOUR Premium account. Existing HA
   Spotify integration is separate; it does not by itself stream to a modified Dot.
4. Choose Librespot if compatible with your account. If every track skips with
   audio_key error 0 1 / No Ogg capture pattern, Spotify is refusing audio keys;
   switch to Spotify Soloist. This can affect older accounts too.
5. Soloist requires its own API key and one-time Spotify-app pairing. Complete
   its terms/account steps yourself. Use the same Premium account and Wi-Fi;
   select Music Assistant Pairing when asked. The optional developer Client ID
   checkbox is DIFFERENT from Soloist's required API key and can stay off.
   If using a personal Client ID, add https://music-assistant.io/callback to
   the developer app's redirects while retaining any existing HA redirect.
6. In MA add the Home Assistant plugin and authorize it, then add Player Provider
   Home Assistant Media Players and select the underlying ESPHome Dot player.
   On compatible firmware Sendspin with pairing is another option; do not assume
   current upstream controls exist on older installed firmware.
7. Add the Music Assistant integration in HA and expose its Dot media player.
   Test one manual playlist, verify audible sound and then pause it.
8. Distinguish the MA entity (example media_player.dot_music) from the raw ESPHome
   speaker (media_player.dot_speaker) and Spotify account-control entity. Give
   the MA entity a unique name/alias. HA can prefix a child entity name with its
   device name: verify the ACTUAL exposed name in live context.
9. In Samantha's prompt default omitted music outputs to that exact MA name.
   Spotify names the service, not the output speaker. For search_media, always
   pass player_name; play the returned media_content_id/type on the SAME player.
   Do not use Unpause to select a named song/playlist or retry the same entity
   under another alias. Honor explicitly named alternative speakers.
10. Try "Samantha, play Magnetic on Spotify". Inspect search and play tool results,
    MA stream startup and actual speaker audio. Typed input does not test Whisper;
    an unusual artist name may be transcribed incorrectly. Search title alone
    and use catalogue metadata when safe; ask if ambiguous.

This is an LLM instruction default, not a global Home Assistant speaker setting.
No rooms are required when a unique explicit player is supplied by the agent.
Use your own playlist and speaker identifiers; no account/library data is included.

https://www.music-assistant.io/music-providers/spotify/
https://www.music-assistant.io/integration/voice/
https://www.music-assistant.io/player-support/home-assistant/
