
import copy


def test_tc03_reject_string_in_numeric_input(
    client,
    valid_payload,
    auth_headers
):
    """TC03.1 - PM2.5 chứa chuỗi thay vì số."""

    payload = copy.deepcopy(valid_payload)
    payload["values"][0][0] = "invalid_pm25"

    response = client.post(
        "/forecast",
        json=payload,
        headers=auth_headers
    )

    assert response.status_code == 422
    assert "detail" in response.json()

    print("\nTC03.1 - String input")
    print("HTTP Status:", response.status_code)
    print("Invalid PM2.5 string rejected: Yes")


def test_tc03_reject_missing_values_field(
    client,
    auth_headers
):
    """TC03.2 - Request thiếu trường values."""

    response = client.post(
        "/forecast",
        json={},
        headers=auth_headers
    )

    assert response.status_code == 422
    assert "detail" in response.json()

    print("\nTC03.2 - Missing values field")
    print("HTTP Status:", response.status_code)
    print("Missing field rejected: Yes")


def test_tc03_reject_invalid_row_format(
    client,
    valid_payload,
    auth_headers
):
    """TC03.3 - Một hàng có 6 biến thay vì 7 biến."""

    payload = copy.deepcopy(valid_payload)
    payload["values"][0] = payload["values"][0][:6]

    response = client.post(
        "/forecast",
        json=payload,
        headers=auth_headers
    )

    assert response.status_code == 422
    assert "detail" in response.json()

    print("\nTC03.3 - Invalid feature count")
    print("HTTP Status:", response.status_code)
    print("Expected features: 7")
    print("Received features: 6")
