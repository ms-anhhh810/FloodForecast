# Flood Route MCP Server

MCP server cung cấp công cụ kiểm tra tuyến đường và tình trạng ngập lụt, sử dụng dữ liệu thời tiết từ weatherapi.com và độ cao từ open-meteo API.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Chạy server

```bash
uvicorn flood_route_server_python.main:app --port 8000
```

Hoặc:

```bash
python -m flood_route_server_python.main
```

## Biến môi trường

- `MAPBOX_ACCESS_TOKEN`: Token để sử dụng Mapbox Directions API. Mặc định sử dụng token demo.
- `SEARCHAPI_KEY`: API key để sử dụng Google Maps API qua searchapi.io. Mặc định sử dụng key demo.
- `WEATHERAPI_KEY`: API key để sử dụng weatherapi.com API cho dữ liệu thời tiết chính xác. Mặc định sử dụng key demo. Lấy key tại: https://www.weatherapi.com/
- `OPENAI_API_KEY`: API key để sử dụng OpenAI API cho phân loại nguy cơ ngập lụt (tùy chọn). Nếu không có, sẽ dùng rule-based fallback.

## Sử dụng

Server cung cấp tool `check-flood-route` với các tham số:
- `destination` (bắt buộc): Điểm đến (tên địa điểm, ví dụ: "Dinh Độc Lập")
- `startLocation` (tùy chọn): Điểm xuất phát. Nếu không có, server sẽ hỏi người dùng.

Tool sẽ:
1. Geocode địa điểm để lấy tọa độ GPS
2. Lấy dữ liệu thời tiết từ weatherapi.com API (lượng mưa chính xác hơn)
3. Lấy dữ liệu độ cao từ open-meteo API
4. Tính toán nguy cơ ngập lụt dựa trên:
   - **Lượng mưa (precipitation)**: Trường quan trọng nhất từ `current.precip_mm` và `forecast.forecastday[].hour[].precip_mm`
   - **Độ ẩm (humidity)**: Từ `current.humidity` và `forecast.forecastday[].hour[].humidity`
   - **Độ cao (elevation)**: Từ open-meteo elevation API
   - **Xác suất mưa**: Từ `forecast.forecastday[].hour[].chance_of_rain` và `will_it_rain`
5. Tính toán tuyến đường tránh các khu vực ngập lụt
6. Trả về bản đồ với tuyến đường và các khu vực ngập lụt

## Các trường dữ liệu quan trọng để quyết định ngập lụt

Từ API weatherapi.com, các trường sau được sử dụng để đánh giá nguy cơ ngập lụt:

1. **`current.precip_mm`** (QUAN TRỌNG NHẤT): Lượng mưa hiện tại tính bằng mm
2. **`current.humidity`**: Độ ẩm hiện tại (%)
3. **`forecast.forecastday[].day.totalprecip_mm`**: Tổng lượng mưa trong ngày (mm)
4. **`forecast.forecastday[].hour[].precip_mm`**: Lượng mưa theo giờ (mm) - dùng để dự báo
5. **`forecast.forecastday[].hour[].humidity`**: Độ ẩm theo giờ (%)
6. **`forecast.forecastday[].hour[].will_it_rain`**: Có mưa hay không (0/1)
7. **`forecast.forecastday[].hour[].chance_of_rain`**: Xác suất mưa (%)
8. **`forecast.forecastday[].hour[].condition.text`**: Điều kiện thời tiết (text mô tả)

Kết hợp với độ cao địa hình, hệ thống sẽ phân loại nguy cơ ngập lụt thành 5 mức: High, Medium High, Medium, Medium Low, Low.

