import pytest

from biocircuits.util.callbacks import BaseCallback, LambdaCallback


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


def test_lambda_callback():
    callback = LambdaCallback(lambda model, trainer: model * 2 + trainer)
    assert callback(2, 3) == 7


def test_lambda_callback_progress_intent():
    callback = LambdaCallback(lambda d: d["foo"], intent="progress")
    assert callback({"foo": 5}) == 5
