"""Register global litellm success callback for eval cost tracking."""
import litellm

from .helpers import _token_counter_callback

litellm.success_callback = [_token_counter_callback]
