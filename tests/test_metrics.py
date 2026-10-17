from peek_future.evaluation.metrics import compute_wer, compute_bwer_uwer


def test_compute_wer():
    refs = ["the brown fox", "hello world"]
    hyps = ["the brown fox", "hello there world"]
    wer = compute_wer(hyps, refs)
    assert wer > 0.0


def test_compute_bwer_uwer():
    refs = ["apple released new iphone in california"]
    hyps = ["apple released new phone in california"]
    # "apple", "california" are named entities
    entities = [["apple", "california"]]

    res = compute_bwer_uwer(hyps, refs, entities)
    assert res["total_words"] == 6
    assert res["b_words"] == 2  # apple, california
    assert res["u_words"] == 4  # released, new, iphone, in
    # iphone -> phone is a substitution error in unbiased word!
    assert res["b_wer"] == 0.0
    assert res["u_wer"] == 25.0


if __name__ == "__main__":
    print("Running metrics tests...")
    test_compute_wer()
    print("WER test passed.")
    test_compute_bwer_uwer()
    print("B-WER / U-WER test passed.")
    print("All metrics tests passed successfully!")
