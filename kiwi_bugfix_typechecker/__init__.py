def test_assert() -> None:
    try:
        assert False
        # noinspection PyUnreachableCode
        raise RuntimeError("assert keyword doesn't work")
    except AssertionError:
        pass
