from minimlx_runtime.adapters.qwen35 import Qwen35Adapter


class FakeArray:
    def __init__(self, values):
        self.values = list(values)

    def astype(self, _dtype):
        return self

    def __getitem__(self, key):
        # The production path adds a singleton sequence axis, then selects
        # the final sequence position from model logits.
        if key == (slice(None), None):
            return self
        return self

    def tolist(self):
        return list(self.values)


class FakeModel:
    def __init__(self):
        self.inputs = []

    def __call__(self, tokens, *, cache):
        self.inputs.append(tokens)
        return FakeArray([token + 1 for token in tokens.values])


class FakeMx:
    uint32 = object()

    @staticmethod
    def argmax(logits, *, axis):
        assert axis == -1
        return logits

    @staticmethod
    def async_eval(_value):
        pass

    @staticmethod
    def concatenate(values, *, axis):
        assert axis == 0
        return FakeArray(token for value in values for token in value.values)

    @staticmethod
    def eval(_value):
        pass


def test_greedy_decode_tail_keeps_token_arrays_and_collects_once():
    adapter = Qwen35Adapter()
    model = FakeModel()
    adapter.model = model
    sampled = FakeArray([4])

    tail = adapter._greedy_decode_device_tail(FakeMx, sampled, cache=object(), token_count=3)

    assert tail == [5, 6, 7]
    assert len(model.inputs) == 3
    assert all(isinstance(tokens, FakeArray) for tokens in model.inputs)
    assert sampled.values == [4]


def test_greedy_decode_device_tail_with_zero_tokens_does_no_work():
    adapter = Qwen35Adapter()
    model = FakeModel()
    adapter.model = model

    assert adapter._greedy_decode_device_tail(FakeMx, FakeArray([4]), object(), 0) == []
    assert model.inputs == []
