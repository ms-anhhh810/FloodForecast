# Kiến trúc LLM Recommendation cho Flood Route

## 🎯 Mục tiêu
Cho phép LLM (ChatGPT) phân tích dữ liệu current + forecast 6h để đưa ra lời khuyên cụ thể cho người dùng.

## 🏗️ Kiến trúc

### 1. **Data Flow**

```
User Query: "Đi từ A đến B có ngập không?"
    ↓
MCP Tool: check-flood-route
    ↓
1. Geocode locations
2. Get route from Mapbox
3. Calculate flood risks for each segment
4. Get weather data (current + 6h forecast)
    ↓
Response Structure:
{
  text: "Phân tích chi tiết..." (cho LLM đọc),
  structuredContent: {
    forecast6h: { ... },  // Data cho LLM phân tích
    routeSegments: [ ... ],
    weatherInfo: { ... },
    riskDistribution: { ... }
  }
}
    ↓
LLM (ChatGPT) tự động:
- Đọc text response
- Phân tích structuredContent
- Đưa ra lời khuyên cụ thể
```

### 2. **Response Structure**

#### Text Response (cho LLM đọc)
- Tóm tắt tình trạng hiện tại
- Dự báo 6 giờ tới
- Phân tích nguy cơ ngập
- Gợi ý LLM sử dụng structuredContent

#### Structured Content (cho LLM phân tích chi tiết)
```json
{
  "forecast6h": {
    "current": {
      "temperature": 28.5,
      "humidity": 85,
      "precipitation": 0.5,
      "windSpeed": 12.3
    },
    "next6h": [
      {
        "time": "2024-01-15T14:00",
        "precipitation": 0.5,
        "temperature": 28.5,
        "humidity": 85
      },
      // ... 23 more intervals (15-minutely)
    ],
    "summary": {
      "totalPrecipitation": 15.2,
      "maxPrecipitation": 3.5,
      "avgPrecipitation": 0.63,
      "precipitationTrend": "increasing",
      "rainyIntervals": 12,
      "totalIntervals": 24
    }
  },
  "routeSegments": [
    {
      "coordinates": [[lng, lat], ...],
      "riskLevel": "High",
      "color": "#dc2626",
      "elevation": 3.2,
      "precipitation": 2.5
    },
    // ... more segments
  ],
  "riskDistribution": {
    "High": 2,
    "MediumHigh": 3,
    "Medium": 1,
    "MediumLow": 2,
    "Low": 0
  }
}
```

## 💡 Cách LLM sử dụng

### LLM sẽ tự động:
1. **Đọc text response** → Hiểu tình hình tổng quan
2. **Phân tích structuredContent.forecast6h** → Hiểu chi tiết 6h tới
3. **Phân tích routeSegments** → Biết đoạn nào nguy hiểm
4. **Đưa ra lời khuyên** dựa trên:
   - Xu hướng mưa (increasing/decreasing/stable)
   - Tổng lượng mưa dự kiến
   - Số đoạn có nguy cơ cao
   - Thời điểm mưa tạnh

### Ví dụ lời khuyên LLM có thể đưa ra:

**Trường hợp 1: Có nguy cơ cao**
```
"Dựa trên dữ liệu phân tích:
- Hiện tại đang mưa 2.5mm và dự kiến tăng lên 15.2mm trong 6h tới
- Có 2 đoạn đường với nguy cơ cao, 3 đoạn nguy cơ khá cao
- Xu hướng mưa đang tăng dần

💡 Lời khuyên:
- Nên tránh đi tuyến này trong 2-3 giờ tới
- Nếu bắt buộc phải đi, hãy đi chậm, tránh các đoạn có màu đỏ/cam trên bản đồ
- Chuẩn bị phương tiện dự phòng hoặc chọn tuyến thay thế"
```

**Trường hợp 2: An toàn**
```
"Dựa trên dữ liệu phân tích:
- Lượng mưa hiện tại và dự kiến đều thấp (<5mm)
- Tất cả các đoạn đều có nguy cơ thấp
- Mưa dự kiến tạnh trong 1 giờ

💡 Lời khuyên:
- Tuyến đường an toàn, bạn có thể đi ngay
- Lưu ý đi chậm ở các đoạn có màu vàng (nguy cơ trung bình)
- Kiểm tra lại trước khi đi nếu thời tiết thay đổi"
```

## 🔧 Implementation Details

### 1. Hàm `extract_6h_forecast()`
- Lấy data từ `minutely_15` (ưu tiên) hoặc `hourly`
- Tính toán summary statistics
- Format data cho LLM dễ phân tích

### 2. Text Response Format
- Structured format với emoji để dễ đọc
- Tóm tắt key metrics
- Gợi ý LLM sử dụng structuredContent

### 3. Structured Content
- `forecast6h`: Chi tiết 6h forecast
- `routeSegments`: Từng đoạn với risk level
- `riskDistribution`: Phân bố risk levels
- `weatherInfo`: Thông tin tổng quan

## ✅ Lợi ích

1. **LLM có đủ context** để đưa ra lời khuyên chính xác
2. **Data structure rõ ràng** → LLM dễ parse và phân tích
3. **Text + Structured** → LLM có cả overview và detail
4. **Tự động** → Không cần thêm tool call, LLM tự phân tích

## 🚀 Next Steps

1. Test với ChatGPT để xem LLM có đưa ra lời khuyên tốt không
2. Fine-tune text response nếu cần
3. Có thể thêm historical data nếu cần
4. Có thể thêm alternative routes nếu có nguy cơ cao


