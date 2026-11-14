
def test_tc02_reject_input_with_only_80_hours(
    client,
    valid_payload,
    auth_headers
):
    """
    TC02 - Kiểm tra API từ chối dữ liệu thiếu giờ.

    Input:
        80 giờ thay vì 96 giờ.

    Expected:
        HTTP 422
        Không tạo dự báo.
    """

    payload = {
        "values": valid_payload["values"][:80]
    }

    response = client.post(
        "/forecast",
        json=payload,
        headers=auth_headers
    )

    assert response.status_code == 422, (
        f"TC02 thất bại. HTTP {response.status_code}: "
        f"{response.text}"
    )

    result = response.json()

    assert "detail" in result
    assert "forecast_pm25" not in result

    print("\n========== TC02 RESULT ==========")
    print("HTTP Status:", response.status_code)
    print("Input hours:", len(payload["values"]))
    print("Expected hours: 96")
    print("Forecast generated: No")
    print("================================")
