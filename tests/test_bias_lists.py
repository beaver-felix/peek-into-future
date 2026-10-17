from peek_future.data_prep.build_bias_lists import (
    create_training_bias_list,
    create_eval_bias_list,
    collect_unique_entities,
)


def test_bias_lists():
    sample_entities = ["London", "Paris"]
    global_pool = ["London", "Paris", "Berlin", "Tokyo", "Rome", "Madrid"]

    # Test training bias list
    train_bias = create_training_bias_list(sample_entities, global_pool, kappa=2)
    assert train_bias[0] == "<null>"
    # Positive entities + kappa * |pos| negatives
    assert len(train_bias) >= 3

    # Test eval bias list
    eval_bias = create_eval_bias_list(sample_entities, global_pool, target_size=5)
    assert eval_bias[0] == "<null>"
    assert len(eval_bias) == 6  # null + 5 entities
    assert "London" in eval_bias
    assert "Paris" in eval_bias

    # Test collect unique entities
    records = [
        {"entities": [{"text": "London"}, {"text": "Paris"}]},
        {"entities": [{"text": "London"}, {"text": "Tokyo"}]},
    ]
    unique = collect_unique_entities(records)
    assert unique == ["London", "Paris", "Tokyo"]


if __name__ == "__main__":
    print("Running bias list tests...")
    test_bias_lists()
    print("All bias list tests passed!")
