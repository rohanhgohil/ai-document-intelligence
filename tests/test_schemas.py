from app.schemas import PurchaseOrderExtraction


def test_schema_defaults():
    result = PurchaseOrderExtraction()
    assert result.line_items == []
    assert result.confidence_notes == []
