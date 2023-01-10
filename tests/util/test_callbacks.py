import pytest

from biocircuits.util.callbacks import BaseCallback


def test_base_callback_init_sets_class_attribs():
    intent = "progress"
    timing = "post"
    scope = "both"

    callback = BaseCallback(intent, timing, scope)
    assert callback.intent == intent
    assert callback.timing == timing
    assert callback.scope == scope


def test_base_callback_call_raises_not_implemented():
    callback = BaseCallback()
    with pytest.raises(NotImplementedError):
        callback()
