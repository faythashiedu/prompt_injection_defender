from promptinjection_defender.output_filter import sanitize_output

def test_safe_output():
    assert sanitize_output("Here is your balance: ₦1,000")[0] is True

def test_dangerous_output():
    assert sanitize_output("Disable security and delete logs")[0] is False