from agn_real.reporting.model_names import display_model_name


def test_model_name_mapping_has_one_manuscript_agn():
    assert display_model_name("rc_agn") == "AGN"
    assert display_model_name("agn") == "AGN"
    assert display_model_name("standard_agn") == "Unconditioned + cosine"
    assert display_model_name("unconditioned_cosine") == "Unconditioned + cosine"

