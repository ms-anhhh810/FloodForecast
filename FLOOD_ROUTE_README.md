# Ứng dụng Flood Route - Tránh Ngập Lụt

Ứng dụng này giúp kiểm tra tuyến đường và tình trạng ngập lụt, hiển thị bản đồ với tuyến đường tránh các khu vực ngập lụt.

## Tính năng

- ✅ Geocoding địa điểm (chuyển đổi tên địa điểm thành tọa độ GPS)
- ✅ Lấy dữ liệu thời tiết từ open-meteo API (độ ẩm, lượng mưa, tốc độ gió)
- ✅ Lấy dữ liệu độ cao từ open-meteo API
- ✅ Tính toán nguy cơ ngập lụt dựa trên:
  - Độ ẩm cao (>80%)
  - Lượng mưa (>5mm)
  - Độ cao thấp (<10m)
- ✅ Tính toán tuyến đường từ điểm xuất phát đến điểm đến
- ✅ Hiển thị bản đồ với:
  - Tuyến đường (màu xanh)
  - Khu vực ngập lụt (màu đỏ, trong suốt)
  - Điểm xuất phát (màu xanh lá)
  - Điểm đến (màu đỏ)

## Cài đặt

### 1. Cài đặt dependencies

```bash
pnpm install
```

### 2. Build components

```bash
pnpm run build
```

### 3. Serve static assets

```bash
pnpm run serve
```

Assets sẽ được serve tại `http://localhost:4444`

### 4. Cài đặt và chạy Python MCP server

```bash
cd flood_route_server_python
python -m venv .venv
source .venv/bin/activate  # hoặc .venv\Scripts\activate trên Windows
pip install -r requirements.txt
uvicorn main:app --port 8000
```

## Sử dụng trong ChatGPT

### 1. Thêm connector

1. Bật [developer mode](https://platform.openai.com/docs/guides/developer-mode) trong ChatGPT
2. Vào Settings > Connectors
3. Thêm connector với URL: `http://localhost:8000/mcp` 
   - **QUAN TRỌNG**: Phải có `/mcp` ở cuối URL, không phải root `/`
   - Nếu dùng ngrok để expose public URL: `https://your-ngrok-url.ngrok-free.app/mcp`
   - **Lưu ý**: Nếu dùng Visual Studio Dev Tunnels và gặp lỗi 406 GitHub OAuth, hãy dùng ngrok thay thế

### 2. Sử dụng tool

Bạn có thể hỏi ChatGPT:

- "Đi tới Dinh Độc Lập có bị ngập không?"
- "Kiểm tra tuyến đường từ [địa điểm A] đến Dinh Độc Lập"
- "Có ngập lụt không nếu đi từ 10.7823,106.6896 đến Dinh Độc Lập?"

### 3. Tool sẽ:

1. Geocode địa điểm để lấy tọa độ GPS
2. Nếu không có điểm xuất phát, sẽ hỏi bạn
3. Lấy dữ liệu thời tiết và độ cao
4. Tính toán nguy cơ ngập lụt
5. Tính toán tuyến đường
6. Hiển thị bản đồ với tuyến đường và khu vực ngập lụt

## Cấu trúc dự án

```
├── src/flood-route/          # React component cho bản đồ
│   └── index.jsx
├── flood_route_server_python/ # Python MCP server
│   ├── main.py               # Server logic
│   ├── requirements.txt      # Python dependencies
│   └── README.md
└── assets/                   # Built assets (sau khi chạy pnpm run build)
    └── flood-route.html
```

## API sử dụng

- **Google Maps API (via searchapi.io)**: Chuyển đổi tên địa điểm thành tọa độ GPS (tự động thêm "Thành phố Hồ Chí Minh" nếu thiếu)
- **Mapbox Directions API**: Tính toán tuyến đường
- **Open-Meteo Forecast API**: Lấy dữ liệu thời tiết
- **Open-Meteo Elevation API**: Lấy dữ liệu độ cao

## Biến môi trường

- `MAPBOX_ACCESS_TOKEN`: Token để sử dụng Mapbox Directions API (mặc định sử dụng token demo)
- `SEARCHAPI_KEY`: API key để sử dụng Google Maps API qua searchapi.io (mặc định sử dụng key demo)
  - Lưu ý: Có thể lấy API key miễn phí từ https://www.searchapi.io/

## Ví dụ sử dụng

### Ví dụ 1: Hỏi về điểm đến

**Người dùng:** "Đi tới Dinh Độc Lập có bị ngập không?"

**ChatGPT sẽ:**
1. Gọi tool `check-flood-route` với `destination: "Dinh Độc Lập"`
2. Vì không có `startLocation`, tool sẽ trả về câu hỏi: "Bạn muốn xuất phát từ đâu?"
3. Người dùng cung cấp điểm xuất phát
4. Tool tính toán và hiển thị bản đồ

### Ví dụ 2: Cung cấp đầy đủ thông tin

**Người dùng:** "Kiểm tra tuyến đường từ Bến Thành đến Dinh Độc Lập"

**ChatGPT sẽ:**
1. Gọi tool với `destination: "Dinh Độc Lập"` và `startLocation: "Bến Thành"`
2. Tool geocode cả hai địa điểm
3. Lấy dữ liệu thời tiết và độ cao
4. Tính toán nguy cơ ngập lụt
5. Tính toán tuyến đường
6. Hiển thị bản đồ với kết quả

## Lưu ý

- Token Mapbox mặc định là token demo, có giới hạn. Để sử dụng production, cần đăng ký tài khoản Mapbox và thêm token vào biến môi trường.
- Thuật toán tính toán ngập lụt hiện tại là đơn giản. Có thể cải thiện bằng cách:
  - Kiểm tra nhiều điểm dọc tuyến đường
  - Sử dụng dữ liệu lịch sử ngập lụt
  - Tích hợp với các API ngập lụt chuyên dụng
- Tuyến đường hiện tại chưa thực sự tránh các khu vực ngập lụt trong tính toán (chỉ đánh dấu). Có thể cải thiện bằng cách sử dụng Mapbox Directions API với avoid polygons.

## Troubleshooting

### Lỗi: "Widget HTML not found"
- Chạy `pnpm run build` để build components trước khi chạy server

### Lỗi: "Không tìm thấy địa điểm"
- Thử với tên địa điểm cụ thể hơn
- Hoặc cung cấp tọa độ GPS trực tiếp (format: lng,lat)

### Lỗi: "Connection refused"
- Đảm bảo static assets server đang chạy (`pnpm run serve`)
- Đảm bảo MCP server đang chạy (`uvicorn main:app --port 8000`)

### Lỗi: "406 GitHub OAuth" (Visual Studio Dev Tunnels)
- **Nguyên nhân**: Dev Tunnels yêu cầu GitHub authentication nhưng bị từ chối
- **Giải pháp**: 
  - **Cách 1 (Khuyên dùng)**: Dùng ngrok thay thế:
    ```bash
    # Cài đặt ngrok (nếu chưa có)
    # macOS: brew install ngrok
    # Hoặc download từ https://ngrok.com/download
    
    # Expose port 8000
    ngrok http 8000
    ```
    Sau đó dùng URL từ ngrok: `https://your-url.ngrok-free.app/mcp`
  
  - **Cách 2**: Cấu hình lại Dev Tunnels với authentication đúng
  - **Cách 3**: Dùng localhost nếu ChatGPT và server cùng máy (chỉ test local)

### Lỗi: "Not Acceptable: Client must accept text/event-stream"

**⚠️ QUAN TRỌNG: Lỗi này khi truy cập từ trình duyệt là BÌNH THƯỜNG!**

- **Khi truy cập từ trình duyệt**: Lỗi này là expected behavior vì trình duyệt không gửi header `Accept: text/event-stream` cần thiết cho SSE endpoint
- **Khi ChatGPT connector gặp lỗi này**:
  - Đảm bảo URL trong ChatGPT connector là: `http://localhost:8000/mcp` (có `/mcp` ở cuối)
  - Không dùng `http://localhost:8000/` (thiếu `/mcp`)
  - Nếu dùng ngrok/dev tunnel: `https://your-url.ngrok-free.app/mcp` (có `/mcp`)
  - Kiểm tra server đang chạy: truy cập `http://localhost:8000/` (không có `/mcp`) để xem health check
  - Restart MCP server sau khi thay đổi URL
  - Đảm bảo dev tunnel/ngrok đang hoạt động và expose đúng port 8000

