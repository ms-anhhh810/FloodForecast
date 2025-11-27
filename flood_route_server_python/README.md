# Flood Route MCP Server

MCP server cung cấp công cụ kiểm tra tuyến đường và tình trạng ngập lụt, sử dụng dữ liệu thời tiết và độ cao từ open-meteo API.

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
- `OPENAI_API_KEY`: API key để sử dụng OpenAI API cho phân loại nguy cơ ngập lụt (tùy chọn). Nếu không có, sẽ dùng rule-based fallback.

## Sử dụng

Server cung cấp tool `check-flood-route` với các tham số:
- `destination` (bắt buộc): Điểm đến (tên địa điểm, ví dụ: "Dinh Độc Lập")
- `startLocation` (tùy chọn): Điểm xuất phát. Nếu không có, server sẽ hỏi người dùng.

Tool sẽ:
1. Geocode địa điểm để lấy tọa độ GPS
2. Lấy dữ liệu thời tiết từ open-meteo API
3. Lấy dữ liệu độ cao từ open-meteo API
4. Tính toán nguy cơ ngập lụt dựa trên độ ẩm, lượng mưa và độ cao
5. Tính toán tuyến đường tránh các khu vực ngập lụt
6. Trả về bản đồ với tuyến đường và các khu vực ngập lụt

