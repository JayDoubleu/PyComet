import warnings

# Suppress specific Pydantic V2 warning from litellm before any imports
warnings.filterwarnings(
    "ignore",
    message="Valid config keys have changed in V2:*",
    category=UserWarning,
    module="pydantic._internal._config",
)
