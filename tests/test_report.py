import pytest
from pipeline.report import audit, ProvenanceError

def test_untraceable_number_is_refused():
    with pytest.raises(ProvenanceError):
        audit("Expected 42 failures.", {"x": 41}, [])

def test_traceable_numbers_pass():
    audit("Expected 41 failures over 18 months.", {"x": 41, "h": 18}, [])

def test_omitted_degradation_is_refused():
    with pytest.raises(ProvenanceError):
        audit("All good.", {}, ["component 13D could not be modelled"])
