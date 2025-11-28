# Flow Debug: "Đi từ Dinh Độc Lập tới Lata Camping có ngập không"

## Flow Chi Tiết

### 1. **Nhận Request từ ChatGPT**
```
User: "Đi từ Dinh Độc Lập tới Lata Camping có ngập không"
  ↓
ChatGPT → MCP Server: Call tool "check-flood-route"
  ↓
Arguments: {
  "destination": "Lata Camping",
  "start_location": "Dinh Độc Lập"
}
```

### 2. **Geocoding (Lấy tọa độ)**
```
2.1. Geocode "Lata Camping"
  → API: https://www.searchapi.io/api/v1/search?engine=google_maps
  → Kết quả: (lng, lat) của Lata Camping
  
2.2. Geocode "Dinh Độc Lập"
  → API: https://www.searchapi.io/api/v1/search?engine=google_maps
  → Kết quả: (lng, lat) của Dinh Độc Lập
```

### 3. **Lấy Route từ Mapbox**
```
3.1. Gọi Mapbox Directions API
  → Input: start_coords, destination_coords
  → Output: route_coords (list of [lng, lat] points)
           route_info (distance, duration)
```

### 4. **Tính Flood Risk cho từng Segment** ⚠️ **QUAN TRỌNG**

```
4.1. Chia route thành 8-10 segments
  → step = len(route_coords) // 8
  → sample_indices = [0, step, 2*step, ..., len-1]

4.2. Với MỖI segment:
  
  a) Tính midpoint của segment:
     mid_lat = (start_point[1] + end_point[1]) / 2
     mid_lng = (start_point[0] + end_point[0]) / 2
  
  b) Lấy Weather Data tại midpoint:
     → API: https://api.open-meteo.com/v1/forecast
     → Params: latitude=mid_lat, longitude=mid_lng
              minutely_15=precipitation,...
              hourly=precipitation,...
     → Response: { minutely_15: { precipitation: [...] }, hourly: {...} }
  
  c) Lấy Elevation tại midpoint:
     → API: https://api.open-meteo.com/v1/elevation
     → Params: latitude=mid_lat, longitude=mid_lng
     → Response: { elevation: [value] }
  
  d) Tính Precipitation:
     → Ưu tiên: minutely_15.precipitation[:96] (24h, mỗi 15 phút)
     → Fallback: hourly.precipitation[:24]
     → precipitation = max(precip_values)  ⚠️ LẤY MAX TRONG 24H
  
  e) Classify Flood Risk:
     → Gọi LLM (gpt-4o-mini):
       Input: precipitation (mm), elevation (m)
       Output: "High" | "Medium High" | "Medium" | "Medium Low" | "Low"
     → Fallback nếu LLM fail:
       - precipitation > 20mm AND elevation < 5m → "High"
       - precipitation > 15mm AND elevation < 10m → "Medium High"
       - precipitation > 10mm AND elevation < 15m → "Medium"
       - precipitation > 5mm AND elevation < 20m → "Medium Low"
       - else → "Low"
  
  f) Gán màu sắc:
     "High" → #dc2626 (đỏ đậm)
     "Medium High" → #f97316 (cam)
     "Medium" → #eab308 (vàng)
     "Medium Low" → #3b82f6 (xanh dương)
     "Low" → #60a5fa (xanh nhạt)
```

### 5. **Lấy Weather Summary (cho Info Panel)**
```
5.1. Tính midpoint của toàn route:
     mid_lat = (start_lat + dest_lat) / 2
     mid_lng = (start_lng + dest_lng) / 2

5.2. Lấy weather data tại midpoint:
     → API: https://api.open-meteo.com/v1/forecast
     → Lấy current_precipitation, max_precipitation

5.3. Tính Rain Stop ETA:
     → Tìm thời điểm đầu tiên precipitation = 0
     → Format: "X phút nữa" hoặc "X giờ Y phút nữa"
```

### 6. **Tổng hợp và Trả về**
```
6.1. Kiểm tra hasHighRisk:
     → hasHighRisk = any(segment.riskLevel in ["High", "Medium High"] 
                         for segment in route_segments)

6.2. Tạo response_data:
     {
       "start": {...},
       "destination": {...},
       "route": {...},
       "routeSegments": [
         {
           "coordinates": [[lng, lat], ...],
           "riskLevel": "Low",
           "color": "#60a5fa",
           "elevation": 10.5,
           "precipitation": 0.0
         },
         ...
       ],
       "weatherInfo": {
         "currentPrecipitation": 0.0,
         "maxPrecipitation": 0.0,
         "rainStopETA": "Không mưa"
       },
       "hasHighRisk": false,
       "routeInfo": {...}
     }

6.3. Trả về structuredContent cho widget
```

## ⚠️ VẤN ĐỀ CÓ THỂ XẢY RA

### 1. **Precipitation đang lấy MAX trong 24h, không phải CURRENT**
   - Code hiện tại: `precipitation = max(precip_values)`
   - Nếu hiện tại không mưa nhưng có mưa trong 24h → vẫn lấy max
   - **FIX**: Nên lấy `precipitation = precip_values[0]` (current) hoặc average trong 1-2h đầu

### 2. **Fallback Logic quá strict**
   - Cần >20mm mưa + <5m elevation mới là "High"
   - Thực tế: 10-15mm mưa + <10m elevation đã có thể ngập
   - **FIX**: Giảm threshold hoặc cải thiện LLM prompt

### 3. **LLM có thể đang quá conservative**
   - LLM có thể luôn trả về "Low" nếu không có context về địa phương
   - **FIX**: Thêm context về Hồ Chí Minh (vùng trũng, dễ ngập)

### 4. **Elevation data có thể không chính xác**
   - API elevation có thể không phản ánh đúng độ cao thực tế
   - **FIX**: Thêm logging để kiểm tra elevation values

### 5. **Không kiểm tra historical flood data**
   - Chỉ dựa vào weather forecast, không có data về vùng thường xuyên ngập
   - **FIX**: Thêm database hoặc API về flood-prone areas

## 🔧 ĐỀ XUẤT FIX

1. **Thêm logging chi tiết** để debug từng bước
2. **Sửa logic precipitation**: Lấy current + average 2h đầu thay vì max 24h
3. **Cải thiện LLM prompt**: Thêm context về HCM, flood-prone areas
4. **Giảm threshold trong fallback**: Phản ánh thực tế hơn
5. **Thêm validation**: Kiểm tra nếu tất cả segments đều "Low" → cảnh báo


