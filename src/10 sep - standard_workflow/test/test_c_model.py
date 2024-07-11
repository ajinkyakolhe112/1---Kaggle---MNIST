def test_single_example():
    tmp_img = torch.randn(1,28,28,1)

    model = Baseline_NN()

    predicted_probs = model(tmp_img)
    logger.debug(f'{predicted_probs}')