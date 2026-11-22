"""Owner-controlled deployment: legacy finalization cannot activate a provider."""
raise RuntimeError(
    'Legacy finalization is disabled for the continuous-corpus rebuild. '
    'Use the selected continuous-corpus checkpoints and evaluation pipeline; '
    'provider activation remains owner-controlled.'
)
