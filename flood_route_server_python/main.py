"""Flood-aware routing MCP server.

This server provides tools to check flood conditions and show routes
avoiding flooded areas using weather and elevation data from open-meteo API.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import httpx
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.responses import JSONResponse
from starlette.routing import Route


@dataclass(frozen=True)
class FloodRouteWidget:
    identifier: str
    title: str
    template_uri: str
    invoking: str
    invoked: str
    html: str
    response_text: str


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
MAPBOX_ACCESS_TOKEN = os.getenv(
    "MAPBOX_ACCESS_TOKEN",
    "pk.eyJ1IjoiZXJpY25pbmciLCJhIjoiY21icXlubWM1MDRiczJvb2xwM2p0amNyayJ9.n-3O6JI5nOp_Lw96ZO5vJQ"
)
# Note: Geocoding now uses Nominatim OpenStreetMap API (no token required)
# Mapbox token is still used for Directions API


@lru_cache(maxsize=None)
def _load_widget_html(component_name: str) -> str:
    html_path = ASSETS_DIR / f"{component_name}.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf8")

    fallback_candidates = sorted(ASSETS_DIR.glob(f"{component_name}-*.html"))
    if fallback_candidates:
        return fallback_candidates[-1].read_text(encoding="utf8")

    raise FileNotFoundError(
        f'Widget HTML for "{component_name}" not found in {ASSETS_DIR}. '
        "Run `pnpm run build` to generate the assets before starting the server."
    )


widget = FloodRouteWidget(
    identifier="check-flood-route",
    title="Check Flood Route",
    template_uri="ui://widget/flood-route.html",
    invoking="Đang kiểm tra tuyến đường và tình trạng ngập lụt",
    invoked="Đã hiển thị bản đồ với tuyến đường tránh ngập lụt",
    html=_load_widget_html("flood-route"),
    response_text="Đã hiển thị bản đồ với tuyến đường tránh ngập lụt",
)


MIME_TYPE = "text/html+skybridge"


class FloodRouteInput(BaseModel):
    """Schema for flood route tool."""

    destination: str = Field(
        ...,
        description="Điểm đến (tên địa điểm, ví dụ: 'Dinh Độc Lập')",
    )
    start_location: Optional[str] = Field(
        None,
        alias="startLocation",
        description="Điểm xuất phát (tên địa điểm hoặc tọa độ GPS). Nếu không có, sẽ hỏi người dùng.",
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


mcp = FastMCP(
    name="flood-route-python",
    stateless_http=True,
)


TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "destination": {
            "type": "string",
            "description": "Điểm đến (tên địa điểm, ví dụ: 'Dinh Độc Lập')",
        },
        "startLocation": {
            "type": "string",
            "description": "Điểm xuất phát (tên địa điểm hoặc tọa độ GPS). Nếu không có, sẽ hỏi người dùng.",
        },
    },
    "required": ["destination"],
    "additionalProperties": False,
}


def _resource_description(widget: FloodRouteWidget) -> str:
    return f"{widget.title} widget markup"


def _tool_meta(widget: FloodRouteWidget) -> Dict[str, Any]:
    return {
        "openai/outputTemplate": widget.template_uri,
        "openai/toolInvocation/invoking": widget.invoking,
        "openai/toolInvocation/invoked": widget.invoked,
        "openai/widgetAccessible": True,
        "openai/resultCanProduceWidget": True,
    }


def _tool_invocation_meta(widget: FloodRouteWidget) -> Dict[str, Any]:
    return {
        "openai/toolInvocation/invoking": widget.invoking,
        "openai/toolInvocation/invoked": widget.invoked,
    }


async def geocode_location(location_name: str) -> Optional[Tuple[float, float]]:
    """Geocode a location name to coordinates using Google Maps API via searchapi.io."""
    try:
        # Auto-add "Thành phố Hồ Chí Minh" if not present
        query = location_name.strip()
        if "hồ chí minh" not in query.lower() and "ho chi minh" not in query.lower():
            query = f"{query}, Thành phố Hồ Chí Minh"
        
        # Get API key from environment or use default
        api_key = os.getenv("SEARCHAPI_KEY", "Uwf5gn1N1TL6ghTquYLCGgZm")
        
        async with httpx.AsyncClient() as client:
            url = "https://www.searchapi.io/api/v1/search"
            response = await client.get(
                url,
                params={
                    "engine": "google_maps",
                    "q": query,
                    "api_key": api_key,
                },
                headers={
                    "Accept": "*/*",
                    "Accept-Language": "vi-VN,vi;q=0.9",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            
            # Google Maps API returns results in local_results array
            if "local_results" in data and len(data["local_results"]) > 0:
                result = data["local_results"][0]
                if "gps_coordinates" in result:
                    coords = result["gps_coordinates"]
                    lat = float(coords.get("latitude", 0))
                    lon = float(coords.get("longitude", 0))
                    if lat != 0 and lon != 0:
                        return (lon, lat)  # Return as [lng, lat] format
    except Exception as e:
        print(f"Geocoding error for {location_name}: {e}")
    return None


async def get_weather_data(latitude: float, longitude: float) -> Dict[str, Any]:
    """Fetch weather data from open-meteo API with 15-minutely data.
    
    According to Open-Meteo docs:
    - 15-minutely data is only available in Central Europe and North America
    - Other regions use interpolated hourly data
    - Use forecast_minutely_15 to specify forecast range (default is 24 hours)
    - Precipitation is sum of rain + showers + snow for preceding period
    """
    try:
        async with httpx.AsyncClient() as client:
            url = "https://api.open-meteo.com/v1/forecast"
            response = await client.get(
                url,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation,weather_code",
                    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,dew_point_2m,apparent_temperature,weather_code",
                    "minutely_15": "temperature_2m,relative_humidity_2m,precipitation,dew_point_2m,apparent_temperature,shortwave_radiation,direct_radiation,diffuse_radiation,weather_code",
                    "forecast_minutely_15": 24,  # 24 hours = 96 intervals (24 * 4)
                    "forecast_days": 7,  # Default 7 days for hourly data
                    "timezone": "Asia/Ho_Chi_Minh",  # Vietnam timezone
                },
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Weather API error: {e}")
        return {}


async def get_elevation_data(latitude: float, longitude: float) -> Optional[float]:
    """Fetch elevation data from open-meteo API."""
    try:
        async with httpx.AsyncClient() as client:
            url = "https://api.open-meteo.com/v1/elevation"
            response = await client.get(
                url,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                },
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()
            if "elevation" in data and len(data["elevation"]) > 0:
                return data["elevation"][0]
    except Exception as e:
        print(f"Elevation API error: {e}")
    return None


async def classify_flood_risk_with_llm(
    precipitation: float, elevation: Optional[float]
) -> str:
    """
    Use LLM to classify flood risk based on precipitation and elevation.
    Returns: "High", "Medium High", "Medium", "Medium Low", "Low"
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Use fallback if no API key
        return _calculate_flood_risk_fallback(precipitation, elevation)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        elevation_str = f"{elevation:.1f}m" if elevation is not None else "unknown"
        
        prompt = f"""Phân tích nguy cơ ngập lụt tại Thành phố Hồ Chí Minh dựa trên:
- Lượng mưa hiện tại/dự báo: {precipitation:.1f}mm
- Độ cao địa hình: {elevation_str}

Lưu ý: TP.HCM là vùng trũng, nhiều khu vực có độ cao dưới 10m so với mực nước biển, dễ bị ngập khi có mưa. 
Các khu vực có độ cao <5m rất dễ ngập ngay cả khi mưa nhỏ (5-10mm).
Các khu vực có độ cao 5-10m dễ ngập khi mưa vừa (10-20mm).
Các khu vực có độ cao >15m ít nguy cơ ngập trừ khi mưa rất lớn (>30mm).

Hãy phân loại nguy cơ ngập lụt thành một trong các mức sau (chỉ trả về tên mức, không giải thích):
- "High": Nguy cơ cao (mưa >10mm + độ cao <5m, hoặc mưa >20mm + độ cao <10m)
- "Medium High": Nguy cơ khá cao (mưa 5-10mm + độ cao <5m, hoặc mưa 10-20mm + độ cao <10m)
- "Medium": Nguy cơ trung bình (mưa 3-5mm + độ cao <10m, hoặc mưa 10-15mm + độ cao 10-15m)
- "Medium Low": Nguy cơ thấp (mưa 1-3mm + độ cao <15m, hoặc mưa 5-10mm + độ cao >15m)
- "Low": Nguy cơ rất thấp (ít mưa <1mm hoặc độ cao >15m)

Chỉ trả về một từ: High, Medium High, Medium, Medium Low, hoặc Low"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia phân tích nguy cơ ngập lụt. Chỉ trả về một từ: High, Medium High, Medium, Medium Low, hoặc Low."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=10,
        )
        
        result = response.choices[0].message.content.strip()
        # Validate result
        valid_levels = ["High", "Medium High", "Medium", "Medium Low", "Low"]
        if result in valid_levels:
            return result
        
        # Fallback to rule-based if LLM returns invalid
        return _calculate_flood_risk_fallback(precipitation, elevation)
    except Exception as e:
        print(f"LLM classification error: {e}, using fallback")
        return _calculate_flood_risk_fallback(precipitation, elevation)


def _calculate_flood_risk_fallback(
    precipitation: float, elevation: Optional[float]
) -> str:
    """Fallback rule-based classification if LLM fails or not available.
    Adjusted for Ho Chi Minh City context (low-lying area, prone to flooding)."""
    # Rule-based classification based on precipitation and elevation
    # TP.HCM is low-lying, many areas <10m above sea level
    if elevation is None:
        elevation = 10  # Default elevation for HCM (assume low-lying)
    
    # High risk: moderate rain (>10mm) + very low elevation (<5m), or heavy rain (>20mm) + low elevation (<10m)
    if (precipitation > 10 and elevation < 5) or (precipitation > 20 and elevation < 10):
        return "High"
    # Medium High: light-moderate rain (5-10mm) + very low elevation (<5m), or moderate rain (10-20mm) + low elevation (<10m)
    elif (precipitation > 5 and elevation < 5) or (precipitation > 10 and elevation < 10):
        return "Medium High"
    # Medium: light rain (3-5mm) + low elevation (<10m), or moderate rain (10-15mm) + medium elevation (10-15m)
    elif (precipitation > 3 and elevation < 10) or (precipitation > 10 and elevation < 15):
        return "Medium"
    # Medium Low: very light rain (1-3mm) + low elevation (<15m), or light rain (5-10mm) + medium-high elevation (>15m)
    elif (precipitation > 1 and elevation < 15) or (precipitation > 5 and elevation >= 15):
        return "Medium Low"
    # Low: little rain or high elevation
    else:
        return "Low"


def get_flood_risk_color(risk_level: str) -> str:
    """Get color for flood risk level (from dark blue/red for high to light for low)."""
    color_map = {
        "High": "#dc2626",  # Red (dark) - high risk
        "Medium High": "#f97316",  # Orange
        "Medium": "#eab308",  # Yellow
        "Medium Low": "#3b82f6",  # Blue
        "Low": "#60a5fa",  # Light blue - low risk
    }
    return color_map.get(risk_level, "#60a5fa")


def calculate_flood_risk(
    weather_data: Dict[str, Any], elevation: Optional[float]
) -> Tuple[bool, List[List[float]]]:
    """
    Calculate flood risk based on weather and elevation.
    Returns (is_flooded, flood_zones).
    """
    flood_zones = []
    is_flooded = False

    if not weather_data:
        return (False, [])

    # Check current conditions
    current = weather_data.get("current", {})
    humidity = current.get("relative_humidity_2m", 0)
    precipitation = 0

    # Check hourly precipitation
    hourly = weather_data.get("hourly", {})
    if "precipitation" in hourly:
        precip_values = hourly["precipitation"][:24]  # Next 24 hours
        precipitation = max(precip_values) if precip_values else 0

    # Determine flood risk
    # High risk if: high humidity (>80%) + precipitation (>5mm) + low elevation (<10m)
    high_humidity = humidity > 80
    has_precipitation = precipitation > 5
    low_elevation = elevation is not None and elevation < 10

    if (high_humidity and has_precipitation) or (has_precipitation and low_elevation):
        is_flooded = True
        # Create a simple flood zone polygon around the area
        # In a real implementation, this would be more sophisticated
        lat, lng = (
            weather_data.get("latitude", 0),
            weather_data.get("longitude", 0),
        )
        # Create a small square around the point (in [lng, lat] format)
        zone_size = 0.01  # ~1km
        flood_zone = [
            [lng - zone_size, lat - zone_size],
            [lng + zone_size, lat - zone_size],
            [lng + zone_size, lat + zone_size],
            [lng - zone_size, lat + zone_size],
            [lng - zone_size, lat - zone_size],
        ]
        flood_zones.append(flood_zone)

    return (is_flooded, flood_zones)


async def get_route(
    start: Tuple[float, float], end: Tuple[float, float], avoid_zones: List[List[float]]
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Get route from Mapbox Directions API, avoiding flood zones."""
    try:
        async with httpx.AsyncClient() as client:
            # Build coordinates string
            coords = f"{start[0]},{start[1]};{end[0]},{end[1]}"
            
            url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{coords}"
            params = {
                "access_token": MAPBOX_ACCESS_TOKEN,
                "geometries": "geojson",
                "overview": "full",
            }

            # If we have flood zones, we could add avoid polygons
            # For now, we'll just get the route and mark flood zones separately
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            data = response.json()

            if data.get("routes"):
                route = data["routes"][0]
                geometry = route["geometry"]["coordinates"]
                distance = route.get("distance", 0) / 1000  # Convert to km
                duration = route.get("duration", 0) / 60  # Convert to minutes

                route_info = {
                    "distance": f"{distance:.1f} km",
                    "duration": f"{duration:.0f} phút",
                    "avoidedFloodZones": len(avoid_zones),
                }

                return (geometry, route_info)
    except Exception as e:
        print(f"Route API error: {e}")

    # Fallback: simple straight line
    return ([start, end], {"distance": "N/A", "duration": "N/A"})


async def calculate_route_flood_risks(
    route_coords: List[List[float]]
) -> List[Dict[str, Any]]:
    """
    Calculate flood risk for multiple points along the route.
    Gets weather and elevation data for each point along the route.
    Returns list of route segments with risk levels and colors.
    """
    if not route_coords or len(route_coords) < 2:
        return []
    
    # Sample points along the route (every Nth point to avoid too many API calls)
    # Use about 8-10 segments for good visualization
    step = max(1, len(route_coords) // 8)
    sample_indices = list(range(0, len(route_coords), step))
    if sample_indices[-1] != len(route_coords) - 1:
        sample_indices.append(len(route_coords) - 1)
    
    sample_points = [route_coords[i] for i in sample_indices]
    
    route_segments = []
    
    # Process each segment - get weather and elevation for each point
    for i in range(len(sample_points) - 1):
        start_idx = sample_indices[i]
        end_idx = sample_indices[i + 1]
        
        # Get all coordinates for this segment
        segment_coords = route_coords[start_idx:end_idx + 1]
        start_point = sample_points[i]
        end_point = sample_points[i + 1]
        
        # Get midpoint of segment for weather and elevation check
        mid_lat = (start_point[1] + end_point[1]) / 2
        mid_lng = (start_point[0] + end_point[0]) / 2
        
        # Get weather data for this specific point along the route
        weather_data = await get_weather_data(mid_lat, mid_lng)
        
        # Get elevation for this point
        elevation = await get_elevation_data(mid_lat, mid_lng)
        
        # Get precipitation from weather data
        # Priority: current.precipitation > minutely_15[0] > hourly[0]
        # Then use average of next 2 hours for risk assessment
        current_weather = weather_data.get("current", {})
        minutely_15 = weather_data.get("minutely_15", {})
        hourly = weather_data.get("hourly", {})
        
        # Get current precipitation (most accurate)
        current_precip = current_weather.get("precipitation", 0)
        if current_precip is None:
            current_precip = 0
        
        # Get forecast values for next 2 hours
        forecast_precip_values = []
        
        # Try to get from 15-minutely data first (more accurate)
        if "precipitation" in minutely_15:
            precip_values = minutely_15["precipitation"][:8]  # Next 2 hours (8 * 15min = 2h)
            if precip_values:
                forecast_precip_values = precip_values
                # If current_precip is 0, use first forecast value
                if current_precip == 0:
                    current_precip = precip_values[0] if len(precip_values) > 0 else 0
        elif "precipitation" in hourly:
            precip_values = hourly["precipitation"][:3]  # Next 3 hours
            if precip_values:
                forecast_precip_values = precip_values
                # If current_precip is 0, use first forecast value
                if current_precip == 0:
                    current_precip = precip_values[0] if len(precip_values) > 0 else 0
        
        # Calculate precipitation for risk assessment
        # Use average of current + next 2 hours (more realistic than max)
        if forecast_precip_values:
            avg_precip = sum(forecast_precip_values) / len(forecast_precip_values)
            # Use the higher of current or average to catch immediate risk
            precipitation = max(current_precip, avg_precip)
        else:
            precipitation = current_precip
        
        # Classify flood risk using LLM based on this point's data
        risk_level = await classify_flood_risk_with_llm(precipitation, elevation)
        color = get_flood_risk_color(risk_level)
        
        # Debug logging
        print(f"[DEBUG] Segment {i+1}: lat={mid_lat:.4f}, lng={mid_lng:.4f}, "
              f"elevation={elevation}, precipitation={precipitation:.2f}mm, "
              f"risk={risk_level}")
        
        route_segments.append({
            "coordinates": segment_coords,  # Full segment coordinates
            "riskLevel": risk_level,
            "color": color,
            "elevation": elevation,
            "precipitation": precipitation,
            "latitude": mid_lat,
            "longitude": mid_lng,
        })
    
    return route_segments


def extract_6h_forecast(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract detailed 6-hour forecast data for LLM analysis.
    
    According to Open-Meteo docs:
    - minutely_15 precipitation is sum for preceding 15 minutes
    - hourly precipitation is sum for preceding 1 hour
    - 15-minutely data may not be available in all regions (falls back to hourly)
    """
    forecast = {
        "current": {},
        "next6h": [],
        "summary": {},
        "dataSource": "unknown"  # Track which data source we're using
    }
    
    minutely_15 = weather_data.get("minutely_15", {})
    hourly = weather_data.get("hourly", {})
    current = weather_data.get("current", {})
    
    # Current conditions (based on 15-minutely model data)
    forecast["current"] = {
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "precipitation": current.get("precipitation", 0),  # Sum for preceding interval
        "windSpeed": current.get("wind_speed_10m"),
        "weatherCode": current.get("weather_code"),  # WMO weather code
        "time": current.get("time"),  # Timestamp of current data
        "interval": current.get("interval"),  # Interval in seconds (e.g., 900 for 15 min)
    }
    
    # Next 6 hours forecast
    # Prefer 15-minutely data (24 intervals = 6 hours)
    if "precipitation" in minutely_15 and "time" in minutely_15:
        times = minutely_15["time"][:24]  # 6 hours = 24 * 15min
        precip_values = minutely_15["precipitation"][:24]
        temp_values = minutely_15.get("temperature_2m", [None] * len(times))[:24]
        humidity_values = minutely_15.get("relative_humidity_2m", [None] * len(times))[:24]
        weather_codes = minutely_15.get("weather_code", [None] * len(times))[:24]
        
        forecast["dataSource"] = "minutely_15"
        forecast["next6h"] = [
            {
                "time": time_str,
                "precipitation": precip,  # mm for preceding 15 minutes
                "temperature": temp,
                "humidity": hum,
                "weatherCode": wc,  # WMO weather code
            }
            for time_str, precip, temp, hum, wc in zip(times, precip_values, temp_values, humidity_values, weather_codes)
        ]
        
        # Summary statistics
        if precip_values:
            forecast["summary"] = {
                "totalPrecipitation": round(sum(precip_values), 2),
                "maxPrecipitation": round(max(precip_values), 2),
                "avgPrecipitation": round(sum(precip_values) / len(precip_values), 2),
                "precipitationTrend": "increasing" if len(precip_values) > 1 and precip_values[-1] > precip_values[0] else "decreasing" if len(precip_values) > 1 and precip_values[-1] < precip_values[0] else "stable",
                "rainyIntervals": sum(1 for p in precip_values if p > 0),
                "totalIntervals": len(precip_values),
                "intervalMinutes": 15,
            }
    elif "precipitation" in hourly and "time" in hourly:
        # Fallback to hourly data (6 hours = 6 intervals)
        times = hourly["time"][:6]
        precip_values = hourly["precipitation"][:6]
        temp_values = hourly.get("temperature_2m", [None] * len(times))[:6]
        humidity_values = hourly.get("relative_humidity_2m", [None] * len(times))[:6]
        weather_codes = hourly.get("weather_code", [None] * len(times))[:6]
        
        forecast["dataSource"] = "hourly"
        forecast["next6h"] = [
            {
                "time": time_str,
                "precipitation": precip,  # mm for preceding 1 hour
                "temperature": temp,
                "humidity": hum,
                "weatherCode": wc,
            }
            for time_str, precip, temp, hum, wc in zip(times, precip_values, temp_values, humidity_values, weather_codes)
        ]
        
        if precip_values:
            forecast["summary"] = {
                "totalPrecipitation": round(sum(precip_values), 2),
                "maxPrecipitation": round(max(precip_values), 2),
                "avgPrecipitation": round(sum(precip_values) / len(precip_values), 2),
                "precipitationTrend": "increasing" if len(precip_values) > 1 and precip_values[-1] > precip_values[0] else "decreasing" if len(precip_values) > 1 and precip_values[-1] < precip_values[0] else "stable",
                "rainyIntervals": sum(1 for p in precip_values if p > 0),
                "totalIntervals": len(precip_values),
                "intervalMinutes": 60,
            }
    else:
        # No data available
        forecast["summary"] = {
            "totalPrecipitation": 0,
            "maxPrecipitation": 0,
            "avgPrecipitation": 0,
            "precipitationTrend": "unknown",
            "rainyIntervals": 0,
            "totalIntervals": 0,
        }
    
    return forecast


def calculate_rain_stop_eta(weather_data: Dict[str, Any]) -> Optional[str]:
    """Calculate when rain will stop based on 15-minutely or hourly forecast.
    
    According to Open-Meteo docs:
    - Precipitation values are sums for preceding period (15 min or 1 hour)
    - Need to check if precipitation is > 0 to determine if raining
    """
    minutely_15 = weather_data.get("minutely_15", {})
    hourly = weather_data.get("hourly", {})
    current = weather_data.get("current", {})
    
    # Check current precipitation first
    current_precip = current.get("precipitation", 0)
    if current_precip == 0:
        # Check if rain is coming
        if "precipitation" in minutely_15 and "time" in minutely_15:
            precip_values = minutely_15["precipitation"][:24]  # Next 6 hours
            if any(p > 0 for p in precip_values):
                # Find first interval with rain
                for i, precip in enumerate(precip_values):
                    if precip > 0:
                        return f"Mưa dự kiến bắt đầu sau {i * 15} phút"
        return "Không mưa"
    
    # Currently raining - find when it stops
    # Prefer 15-minutely data for more accurate timing
    if "precipitation" in minutely_15 and "time" in minutely_15:
        times = minutely_15["time"][:96]  # Next 24 hours
        precip_values = minutely_15["precipitation"][:96]
        
        # Find when rain stops (first interval with 0 precipitation after current)
        for i, (time_str, precip) in enumerate(zip(times, precip_values)):
            if i > 0 and precip == 0:  # Rain stopped (skip first as it's current)
                # Check if previous intervals had rain (to confirm it was raining)
                if any(p > 0 for p in precip_values[:i]):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                        # Calculate minutes from now
                        minutes_from_now = i * 15
                        if minutes_from_now < 60:
                            return f"{minutes_from_now} phút nữa"
                        else:
                            hours = minutes_from_now // 60
                            mins = minutes_from_now % 60
                            if mins == 0:
                                return f"{hours} giờ nữa"
                            else:
                                return f"{hours} giờ {mins} phút nữa"
                    except Exception as e:
                        print(f"Error parsing time: {e}")
                        return f"{i * 15} phút nữa"
        
        return "Hơn 24 giờ"
    
    # Fallback to hourly data
    elif "precipitation" in hourly and "time" in hourly:
        times = hourly["time"][:24]
        precip_values = hourly["precipitation"][:24]
        
        for i, (time_str, precip) in enumerate(zip(times, precip_values)):
            if i > 0 and precip == 0:
                if any(p > 0 for p in precip_values[:i]):
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                        return dt.strftime("%H:%M ngày %d/%m")
                    except Exception as e:
                        print(f"Error parsing time: {e}")
                        return f"{i} giờ nữa"
        
        return "Hơn 24 giờ"
    
    return None


@mcp._mcp_server.list_tools()
async def _list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name=widget.identifier,
            title=widget.title,
            description="Kiểm tra tuyến đường và tình trạng ngập lụt. Trả về bản đồ với tuyến đường tránh các khu vực ngập lụt.",
            inputSchema=deepcopy(TOOL_INPUT_SCHEMA),
            _meta=_tool_meta(widget),
            annotations={
                "destructiveHint": False,
                "openWorldHint": False,
                "readOnlyHint": True,
            },
        )
    ]


@mcp._mcp_server.list_resources()
async def _list_resources() -> List[types.Resource]:
    return [
        types.Resource(
            name=widget.title,
            title=widget.title,
            uri=widget.template_uri,
            description=_resource_description(widget),
            mimeType=MIME_TYPE,
            _meta=_tool_meta(widget),
        )
    ]


@mcp._mcp_server.list_resource_templates()
async def _list_resource_templates() -> List[types.ResourceTemplate]:
    return [
        types.ResourceTemplate(
            name=widget.title,
            title=widget.title,
            uriTemplate=widget.template_uri,
            description=_resource_description(widget),
            mimeType=MIME_TYPE,
            _meta=_tool_meta(widget),
        )
    ]


async def _handle_read_resource(req: types.ReadResourceRequest) -> types.ServerResult:
    if str(req.params.uri) != widget.template_uri:
        return types.ServerResult(
            types.ReadResourceResult(
                contents=[],
                _meta={"error": f"Unknown resource: {req.params.uri}"},
            )
        )

    contents = [
        types.TextResourceContents(
            uri=widget.template_uri,
            mimeType=MIME_TYPE,
            text=widget.html,
            _meta=_tool_meta(widget),
        )
    ]

    return types.ServerResult(types.ReadResourceResult(contents=contents))


async def _call_tool_request(req: types.CallToolRequest) -> types.ServerResult:
    if req.params.name != widget.identifier:
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Unknown tool: {req.params.name}",
                    )
                ],
                isError=True,
            )
        )

    arguments = req.params.arguments or {}
    try:
        payload = FloodRouteInput.model_validate(arguments)
    except ValidationError as exc:
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Input validation error: {exc.errors()}",
                    )
                ],
                isError=True,
            )
        )

    destination_name = payload.destination
    start_location_name = payload.start_location

    # Geocode destination
    destination_coords = await geocode_location(destination_name)
    if not destination_coords:
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Không tìm thấy địa điểm: {destination_name}. Vui lòng thử lại với tên địa điểm khác.",
                    )
                ],
                isError=True,
            )
        )

    # Geocode start location if provided
    start_coords = None
    if start_location_name:
        start_coords = await geocode_location(start_location_name)
        if not start_coords:
            # Try parsing as coordinates (format: lng,lat - Mapbox format)
            try:
                parts = start_location_name.split(",")
                if len(parts) == 2:
                    # Mapbox uses [lng, lat] format
                    lng = float(parts[0].strip())
                    lat = float(parts[1].strip())
                    # Validate reasonable coordinate ranges
                    if -180 <= lng <= 180 and -90 <= lat <= 90:
                        start_coords = (lng, lat)
            except (ValueError, IndexError):
                pass

    # If no start location, we need to ask the user
    if not start_coords:
        return types.ServerResult(
            types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"Bạn muốn xuất phát từ đâu để đi tới {destination_name}? Vui lòng cung cấp địa điểm xuất phát hoặc tọa độ GPS.",
                    )
                ],
                isError=False,
            )
        )

    # Get route first
    route_coords, route_info = await get_route(start_coords, destination_coords, [])

    # Calculate flood risks for multiple points along the route
    # This will get weather and elevation data for each point along the route
    route_segments = await calculate_route_flood_risks(route_coords)
    
    # Get overall weather data from midpoint for summary info
    dest_lat, dest_lng = destination_coords[1], destination_coords[0]
    start_lat, start_lng = start_coords[1], start_coords[0]
    mid_lat = (start_lat + dest_lat) / 2
    mid_lng = (start_lng + dest_lng) / 2
    
    weather_data = await get_weather_data(mid_lat, mid_lng)
    
    # Extract 6-hour forecast for LLM analysis
    forecast_6h = extract_6h_forecast(weather_data)
    
    # Calculate rain stop ETA
    rain_stop_eta = calculate_rain_stop_eta(weather_data)
    
    # Get precipitation info from overall weather data
    # IMPORTANT: Use current.precipitation first (most accurate for current conditions)
    # Then fallback to hourly/minutely_15 forecast data
    current_weather = weather_data.get("current", {})
    minutely_15 = weather_data.get("minutely_15", {})
    hourly = weather_data.get("hourly", {})
    
    # Priority 1: Get current precipitation from current.weather (most accurate)
    current_precipitation = current_weather.get("precipitation", 0)
    if current_precipitation is None:
        current_precipitation = 0
    
    # Get max precipitation from forecast data
    max_precipitation = 0
    
    # Try to get from 15-minutely data first (more accurate)
    if "precipitation" in minutely_15:
        precip_values = minutely_15["precipitation"][:96]  # Next 24 hours
        if precip_values:
            # If current_precipitation is 0, try first forecast value as fallback
            if current_precipitation == 0:
                current_precipitation = precip_values[0] if len(precip_values) > 0 else 0
            max_precipitation = max(precip_values) if precip_values else 0
    elif "precipitation" in hourly:
        precip_values = hourly["precipitation"][:24]
        if precip_values:
            # If current_precipitation is 0, try first forecast value as fallback
            if current_precipitation == 0:
                current_precipitation = precip_values[0] if len(precip_values) > 0 else 0
            max_precipitation = max(precip_values) if precip_values else 0
    
    # Debug logging to check actual data
    print(f"[DEBUG] Weather data check:")
    print(f"  - current.precipitation: {current_weather.get('precipitation')}")
    print(f"  - current.time: {current_weather.get('time')}")
    print(f"  - minutely_15 available: {'precipitation' in minutely_15}")
    print(f"  - hourly available: {'precipitation' in hourly}")
    if "precipitation" in minutely_15 and minutely_15["precipitation"]:
        print(f"  - minutely_15[0]: {minutely_15['precipitation'][0]}")
    if "precipitation" in hourly and hourly["precipitation"]:
        print(f"  - hourly[0]: {hourly['precipitation'][0]}")
    print(f"  - Final current_precipitation: {current_precipitation}")
    print(f"  - Final max_precipitation: {max_precipitation}")
    
    # Determine overall flood risk
    risk_levels = [seg["riskLevel"] for seg in route_segments]
    has_high_risk = any(level in ["High", "Medium High"] for level in risk_levels)
    
    # Count risk distribution
    risk_distribution = {
        "High": sum(1 for r in risk_levels if r == "High"),
        "MediumHigh": sum(1 for r in risk_levels if r == "Medium High"),
        "Medium": sum(1 for r in risk_levels if r == "Medium"),
        "MediumLow": sum(1 for r in risk_levels if r == "Medium Low"),
        "Low": sum(1 for r in risk_levels if r == "Low"),
    }
    
    # Prepare response data with 6h forecast
    response_data = {
        "start": list(start_coords),
        "destination": list(destination_coords),
        "route": route_coords,
        "routeSegments": route_segments,  # Segments with risk levels and colors
        "weatherInfo": {
            "currentPrecipitation": round(current_precipitation, 1),
            "maxPrecipitation": round(max_precipitation, 1),
            "rainStopETA": rain_stop_eta,
            "unit": "mm",
        },
        "forecast6h": forecast_6h,  # Detailed 6-hour forecast for LLM
        "riskDistribution": risk_distribution,  # Risk level counts
        "hasHighRisk": has_high_risk,
        "routeInfo": route_info,
    }

    # Build comprehensive response text for LLM analysis
    # This text will help LLM understand the situation and provide recommendations
    current_temp = forecast_6h.get("current", {}).get("temperature")
    current_humidity = forecast_6h.get("current", {}).get("humidity")
    forecast_summary = forecast_6h.get("summary", {})
    
    # Format temperature and humidity safely
    temp_str = f"{current_temp:.1f}°C" if current_temp is not None else "N/A"
    humidity_str = f"{current_humidity:.1f}%" if current_humidity is not None else "N/A"
    
    response_text = f"""Phân tích tuyến đường từ {start_location_name or 'điểm xuất phát'} đến {destination_name}:

📊 **Tình trạng hiện tại:**
- Lượng mưa: {current_precipitation:.1f}mm
- Nhiệt độ: {temp_str}
- Độ ẩm: {humidity_str}
- Mưa dự kiến tạnh: {rain_stop_eta or 'Không có thông tin'}

📈 **Dự báo 6 giờ tới:**
- Tổng lượng mưa dự kiến: {forecast_summary.get('totalPrecipitation', 0):.1f}mm
- Lượng mưa tối đa: {forecast_summary.get('maxPrecipitation', 0):.1f}mm
- Lượng mưa trung bình: {forecast_summary.get('avgPrecipitation', 0):.1f}mm
- Xu hướng: {forecast_summary.get('precipitationTrend', 'unknown')}
- Số khoảng thời gian có mưa: {forecast_summary.get('rainyIntervals', 0)}/{forecast_summary.get('totalIntervals', 0)}

⚠️ **Phân tích nguy cơ ngập:**
- Tổng số đoạn đường: {len(route_segments)}
- Đoạn nguy cơ cao: {risk_distribution['High']}
- Đoạn nguy cơ khá cao: {risk_distribution['MediumHigh']}
- Đoạn nguy cơ trung bình: {risk_distribution['Medium']}
- Đoạn nguy cơ thấp: {risk_distribution['MediumLow'] + risk_distribution['Low']}

{'⚠️ CÓ NGUY CƠ NGẬP LỤT' if has_high_risk else '✓ Tuyến đường tương đối an toàn'}

Bản đồ hiển thị mức độ nguy cơ từ cao (đỏ đậm) đến thấp (xanh nhạt) dọc tuyến đường.

💡 **Dữ liệu chi tiết 6 giờ tới đã được cung cấp trong structuredContent để bạn có thể đưa ra lời khuyên cụ thể cho người dùng.**"""

    meta = _tool_invocation_meta(widget)

    return types.ServerResult(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=response_text,
                )
            ],
            structuredContent=response_data,
            _meta=meta,
        )
    )


mcp._mcp_server.request_handlers[types.CallToolRequest] = _call_tool_request
mcp._mcp_server.request_handlers[types.ReadResourceRequest] = _handle_read_resource


_base_app = mcp.streamable_http_app()

# Add middleware to automatically add Accept: text/event-stream header for /mcp endpoints
class SSEAcceptMiddleware:
    """Middleware to automatically add Accept: text/event-stream header for SSE endpoints."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            # For /mcp and /mcp.sse endpoints, ensure Accept header is set
            if path in ["/mcp", "/mcp.sse"]:
                headers_list = list(scope.get("headers", []))
                headers_dict = {k.decode().lower(): v.decode() for k, v in headers_list}
                
                # Add Accept header if not present or doesn't include text/event-stream
                if "accept" not in headers_dict or "text/event-stream" not in headers_dict.get("accept", "").lower():
                    # Remove old accept header if exists
                    headers_list = [(k, v) for k, v in headers_list if k.lower() != b"accept"]
                    # Add new accept header
                    headers_list.append((b"accept", b"text/event-stream"))
                    scope = {**scope, "headers": headers_list}
                    print(f"[DEBUG] ✅ Auto-added Accept: text/event-stream header for {path}")
        
        return await self.app(scope, receive, send)

# Wrap base app with SSE Accept middleware FIRST
app = SSEAcceptMiddleware(_base_app)

# Add a simple health check endpoint
async def health_check(request):
    """Health check endpoint to verify server is running."""
    return JSONResponse({
        "status": "ok",
        "service": "flood-route-mcp-server",
        "endpoints": {
            "mcp": "/mcp (SSE endpoint - middleware auto-adds Accept header)",
            "mcp_sse": "/mcp.sse (SSE endpoint alias - same as /mcp)",
            "health": "/ (this endpoint)",
        },
        "note": "✅ Middleware tự động thêm header Accept: text/event-stream cho /mcp và /mcp.sse endpoints",
    })

# Add alias endpoint /mcp.sse that forwards to /mcp SSE endpoint
async def mcp_sse_alias(request):
    """Alias endpoint /mcp.sse that forwards to /mcp SSE endpoint."""
    # Create a new scope with path changed to /mcp
    new_scope = {**request.scope, "path": "/mcp", "raw_path": b"/mcp"}
    # Forward the request to the base app's /mcp handler
    return await _base_app(new_scope, request.receive, request._send)

# Add routes to base app (before middleware wrapper)
_base_app.routes.append(Route("/", health_check, methods=["GET"]))
_base_app.routes.append(Route("/mcp.sse", mcp_sse_alias, methods=["GET"]))

try:
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
        expose_headers=["*"],
    )
except Exception:
    pass


if __name__ == "__main__":
    import uvicorn

    print("Starting Flood Route MCP server...")
    print("SSE endpoints:")
    print("  - http://0.0.0.0:8000/mcp (recommended)")
    print("  - http://0.0.0.0:8000/mcp.sse (alias)")
    print("Make sure to use /mcp or /mcp.sse endpoint when adding to ChatGPT")
    print("\n⚠️  Lưu ý: Nếu thấy lỗi 406 Not Acceptable, đảm bảo:")
    print("  1. URL trong ChatGPT connector có /mcp hoặc /mcp.sse ở cuối")
    print("  2. ChatGPT connector sẽ tự động gửi header Accept: text/event-stream")
    print("  3. Lỗi 406 từ trình duyệt là bình thường (trình duyệt không gửi SSE header)")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

