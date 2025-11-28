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
from datetime import datetime, timedelta
import json
import re
import math

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
WEATHERAPI_KEY = os.getenv(
    "WEATHERAPI_KEY",
    "d68d7e5b61434f2787913715252811"  # Default demo key
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
    time_start: Optional[str] = Field(
        None,
        alias="timeStart",
        description="Thời gian bắt đầu đi (ví dụ: '30p nữa', '1 giờ nữa', '14:00', '2025-11-28 14:00'). Nếu không có, mặc định forecast 6 giờ.",
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
        "timeStart": {
            "type": "string",
            "description": "Thời gian bắt đầu đi (ví dụ: '30p nữa', '1 giờ nữa', '14:00', '2025-11-28 14:00'). Nếu không có, mặc định forecast 6 giờ.",
        }
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


def parse_time_start(time_start_str: Optional[str]) -> Optional[datetime]:
    """
    Parse time start string to datetime.
    
    Supports formats:
    - "30p nữa", "30 phút nữa" -> 30 minutes from now
    - "1 giờ nữa", "1h nữa" -> 1 hour from now
    - "14:00" -> today at 14:00
    - "2025-11-28 14:00" -> specific date and time
    
    Returns None if cannot parse or None input (defaults to 6 hours forecast).
    """
    if not time_start_str:
        return None
    
    time_start_str = time_start_str.strip().lower()
    now = datetime.now()

    minutes_match = re.search(r'(\d+)\s*(?:p|phút)\s*nữa', time_start_str)
    if minutes_match:
        minutes = int(minutes_match.group(1))
        return now + timedelta(minutes=minutes)
    
    hours_match = re.search(r'(\d+)\s*(?:giờ|h)\s*nữa', time_start_str)
    if hours_match:
        hours = int(hours_match.group(1))
        return now + timedelta(hours=hours)
    
    time_match = re.search(r'(\d{1,2}):(\d{2})', time_start_str)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target < now:
            target += timedelta(days=1)
        return target
    
    try:
        target = datetime.strptime(time_start_str, "%Y-%m-%d %H:%M")
        return target
    except ValueError:
        pass
    
    try:
        target = datetime.strptime(time_start_str, "%Y-%m-%d %H:%M:%S")
        return target
    except ValueError:
        pass
    
    return None


def _calculate_arrival_time(start_time: Optional[datetime], duration_str: str) -> str:
    """
    Calculate arrival time based on start time and duration.
    
    Args:
        start_time: Start time (if None, uses current time)
        duration_str: Duration string like "279 phút" or "4.5 giờ"
    
    Returns:
        Formatted arrival time string
    """
    if not duration_str or duration_str == "N/A":
        return "N/A"
    
    # Parse duration from string like "279 phút" or "4.5 giờ"
    duration_minutes = 0
    minutes_match = re.search(r'(\d+(?:\.\d+)?)\s*phút', duration_str)
    if minutes_match:
        duration_minutes = int(float(minutes_match.group(1)))
    else:
        hours_match = re.search(r'(\d+(?:\.\d+)?)\s*giờ', duration_str)
        if hours_match:
            duration_minutes = int(float(hours_match.group(1)) * 60)
    
    if duration_minutes == 0:
        return "N/A"
    
    # Calculate arrival time
    if start_time:
        arrival_time = start_time + timedelta(minutes=duration_minutes)
    else:
        arrival_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    return arrival_time.strftime("%H:%M ngày %d/%m")


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great circle distance between two points on the earth (in km).
    
    Args:
        coord1: (longitude, latitude) of first point
        coord2: (longitude, latitude) of second point
    
    Returns:
        Distance in kilometers
    """
    # Radius of earth in kilometers
    R = 6371.0
    
    lat1 = math.radians(coord1[1])
    lon1 = math.radians(coord1[0])
    lat2 = math.radians(coord2[1])
    lon2 = math.radians(coord2[0])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance


async def geocode_location(location_name: str) -> Optional[Tuple[float, float]]:
    """Geocode a location name to coordinates using Google Maps API via searchapi.io."""
    try:     
        # Get API key from environment or use default
        api_key = os.getenv("SEARCHAPI_KEY", "")
        
        # Use location_name directly as query
        query = location_name.strip()
        
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


async def get_weather_data_weatherapi(latitude: float, longitude: float) -> Dict[str, Any]:
    """Fetch weather data from weatherapi.com API.
    
    Returns data with structure:
    - current: {precip_mm, humidity, temp_c, condition, ...}
    - forecast.forecastday[].day: {totalprecip_mm, avghumidity, ...}
    - forecast.forecastday[].hour[]: {precip_mm, humidity, will_it_rain, chance_of_rain, ...}
    
    Key fields for flood risk assessment:
    - current.precip_mm: Current precipitation (mm) - MOST IMPORTANT
    - current.humidity: Current humidity (%)
    - forecast.forecastday[].hour[].precip_mm: Hourly precipitation forecast (mm)
    - forecast.forecastday[].hour[].humidity: Hourly humidity forecast (%)
    - forecast.forecastday[].hour[].will_it_rain: Will it rain (0/1)
    - forecast.forecastday[].hour[].chance_of_rain: Chance of rain (%)
    """
    try:
        async with httpx.AsyncClient() as client:
            url = "https://api.weatherapi.com/v1/forecast.json"
            response = await client.get(
                url,
                params={
                    "key": WEATHERAPI_KEY,
                    "q": f"{latitude},{longitude}",  # lat,lon format
                    "days": 3,  # 3 days forecast
                    "aqi": "no",
                    "alerts": "yes",
                },
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"WeatherAPI error: {e}")
        return {}


async def get_weather_data(latitude: float, longitude: float) -> Dict[str, Any]:
    """Fetch weather data from open-meteo API with 15-minutely data.
    
    DEPRECATED: Use get_weather_data_weatherapi() instead for more accurate precipitation data.
    This function is kept for backward compatibility.
    
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


def extract_precipitation_from_weatherapi(
    weather_data: Dict[str, Any], 
    hours_ahead: int = 2,
    target_time: Optional[datetime] = None
) -> Tuple[float, List[float], Dict[str, Any]]:
    """
    Extract precipitation and related weather data from weatherapi.com format.
    
    Args:
        weather_data: Weather data from weatherapi.com
        hours_ahead: Number of hours to look ahead (default 2)
        target_time: Target datetime to forecast for. If provided, extracts data at that time.
                    If None, uses current time + hours_ahead.
    
    Returns:
        (current_precipitation, forecast_precipitation_list, weather_indicators)
        - current_precipitation: Precipitation at target time (or current) in mm
        - forecast_precipitation_list: List of hourly precipitation for next N hours from target time
        - weather_indicators: Dict with chance_of_rain, will_it_rain, humidity, etc.
    
    Key fields used:
    - current.precip_mm: Current precipitation (mm) - MOST IMPORTANT
    - current.humidity: Current humidity (%)
    - forecast.forecastday[].hour[].precip_mm: Hourly precipitation forecast (mm)
    - forecast.forecastday[].hour[].chance_of_rain: Chance of rain (%)
    - forecast.forecastday[].hour[].will_it_rain: Will it rain (0/1)
    - forecast.forecastday[].hour[].humidity: Hourly humidity (%)
    """
    current_precip = 0.0
    forecast_precip = []
    weather_indicators = {
        "current_humidity": None,
        "max_chance_of_rain": 0,
        "will_rain_soon": False,
        "avg_chance_of_rain": 0,
        "max_humidity": 0,
        "avg_humidity": 0,
    }
    
    if not weather_data:
        return (current_precip, forecast_precip, weather_indicators)
    
    # Get current precipitation and humidity
    current = weather_data.get("current", {})
    current_precip = current.get("precip_mm", 0.0)
    if current_precip is None:
        current_precip = 0.0
    
    current_humidity = current.get("humidity")
    weather_indicators["current_humidity"] = current_humidity
    
    # Get hourly forecast precipitation and related data
    forecast = weather_data.get("forecast", {})
    forecastday_list = forecast.get("forecastday", [])
    
    chance_of_rain_values = []
    will_it_rain_values = []
    humidity_values = []
    
    if forecastday_list:
        # Get hours from today and next days
        all_hours = []
        for day_data in forecastday_list[:2]:  # Today and tomorrow (enough for 24+ hours)
            hours = day_data.get("hour", [])
            all_hours.extend(hours)
        
        # Find target hour if target_time is provided
        target_hour_index = None
        if target_time:
            target_hour_str = target_time.strftime("%Y-%m-%d %H:00")
            for i, hour_data in enumerate(all_hours):
                hour_time = hour_data.get("time", "")
                if hour_time.startswith(target_hour_str[:13]):  # Match date and hour
                    target_hour_index = i
                    break
        
        # If target_time specified, start from that hour, otherwise start from current
        start_index = target_hour_index if target_hour_index is not None else 0
        end_index = min(start_index + hours_ahead, len(all_hours))
        
        # Extract data for target time and next N hours
        for i in range(start_index, end_index):
            if i >= len(all_hours):
                break
            hour_data = all_hours[i]
            
            precip = hour_data.get("precip_mm", 0.0)
            if precip is None:
                precip = 0.0
            forecast_precip.append(precip)
            
            # If this is the target hour, use its precipitation as current
            if target_time and i == start_index:
                current_precip = precip
            
            # Extract chance of rain and will_it_rain
            chance = hour_data.get("chance_of_rain", 0)
            if chance is None:
                chance = 0
            chance_of_rain_values.append(chance)
            
            will_rain = hour_data.get("will_it_rain", 0)
            if will_rain is None:
                will_rain = 0
            will_it_rain_values.append(will_rain)
            
            # Extract humidity
            humidity = hour_data.get("humidity")
            if humidity is not None:
                humidity_values.append(humidity)
                # If this is the target hour, use its humidity as current
                if target_time and i == start_index:
                    weather_indicators["current_humidity"] = humidity
        
        # Calculate indicators
        if chance_of_rain_values:
            weather_indicators["max_chance_of_rain"] = max(chance_of_rain_values)
            weather_indicators["avg_chance_of_rain"] = sum(chance_of_rain_values) / len(chance_of_rain_values)
        
        if will_it_rain_values:
            weather_indicators["will_rain_soon"] = any(w == 1 for w in will_it_rain_values)
        
        if humidity_values:
            weather_indicators["max_humidity"] = max(humidity_values)
            weather_indicators["avg_humidity"] = sum(humidity_values) / len(humidity_values)
        elif current_humidity is not None:
            weather_indicators["max_humidity"] = current_humidity
            weather_indicators["avg_humidity"] = current_humidity
    
    return (current_precip, forecast_precip, weather_indicators)


async def classify_flood_risk_with_llm(
    precipitation: float, 
    elevation: Optional[float],
    weather_indicators: Optional[Dict[str, Any]] = None
) -> str:
    """
    Use LLM to classify flood risk based on precipitation, elevation, and weather indicators.
    Returns: "High", "Medium High", "Medium", "Medium Low", "Low"
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Use fallback if no API key
        return _calculate_flood_risk_fallback(precipitation, elevation, weather_indicators)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        elevation_str = f"{elevation:.1f}m" if elevation is not None else "unknown"
        
        # Build weather indicators string
        indicators_str = ""
        if weather_indicators:
            current_humidity = weather_indicators.get("current_humidity")
            max_chance = weather_indicators.get("max_chance_of_rain", 0)
            avg_chance = weather_indicators.get("avg_chance_of_rain", 0)
            will_rain = weather_indicators.get("will_rain_soon", False)
            max_humidity = weather_indicators.get("max_humidity", 0)
            
            indicators_str = f"""
- Độ ẩm hiện tại: {current_humidity}% (nếu có)
- Xác suất mưa tối đa (2h tới): {max_chance}%
- Xác suất mưa trung bình (2h tới): {avg_chance:.1f}%
- Sẽ mưa sớm: {'Có' if will_rain else 'Không'}
- Độ ẩm tối đa (2h tới): {max_humidity}%"""
        
        prompt = f"""Phân tích nguy cơ ngập lụt tại vị trí đó dựa trên:
- Lượng mưa hiện tại/dự báo: {precipitation:.1f}mm
- Độ cao địa hình: {elevation_str}{indicators_str}

Lưu ý quan trọng (đánh giá dựa trên dữ liệu thực tế):
- Khu vực có độ cao <5m dễ ngập khi mưa vừa (>8mm) hoặc mưa lớn (>10mm)
- Khu vực có độ cao 5-10m dễ ngập khi mưa lớn (>10mm) hoặc mưa vừa (>8mm) + xác suất mưa cao (>60%) + độ ẩm cao (>80%)
- Khu vực có độ cao >15m ít ngập trừ khi mưa lớn (>20mm)
- Nâng mức nguy cơ khi có mưa thực tế (>5mm) hoặc xác suất mưa cao (>60%) + sẽ mưa sớm + độ ẩm cao (>80%)
- Độ ẩm cao (>80%) tăng nguy cơ khi kết hợp với mưa thực tế (>5mm)

Hãy phân loại nguy cơ ngập lụt thành một trong các mức sau (chỉ trả về tên mức, không giải thích):
- "High": Nguy cơ cao (mưa >10mm + độ cao <5m, hoặc mưa >15mm + độ cao <10m, hoặc mưa >8mm + xác suất mưa >60% + độ cao <5m + độ ẩm >80%)
- "Medium High": Nguy cơ khá cao (mưa 8-10mm + độ cao <5m, hoặc mưa 10-15mm + độ cao <10m, hoặc mưa >6mm + xác suất mưa >55% + độ cao <10m)
- "Medium": Nguy cơ trung bình (mưa 5-8mm + độ cao <10m, hoặc mưa 8-10mm + độ cao 10-15m, hoặc mưa >4mm + xác suất mưa >50% + độ cao <15m)
- "Medium Low": Nguy cơ thấp (mưa 2-5mm + độ cao <15m, hoặc mưa 4-8mm + độ cao >15m, hoặc mưa >1mm + xác suất mưa >40%)
- "Low": Nguy cơ rất thấp (ít mưa <2mm và xác suất mưa <40% hoặc độ cao >15m)

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
        return _calculate_flood_risk_fallback(precipitation, elevation, weather_indicators)
    except Exception as e:
        print(f"LLM classification error: {e}, using fallback")
        return _calculate_flood_risk_fallback(precipitation, elevation, weather_indicators)


def _calculate_flood_risk_fallback(
    precipitation: float, 
    elevation: Optional[float],
    weather_indicators: Optional[Dict[str, Any]] = None
) -> str:
    """Fallback rule-based classification if LLM fails or not available.
    Now includes chance_of_rain, will_it_rain, and humidity for better accuracy."""
    # Rule-based classification based on precipitation, elevation, and weather indicators
    if elevation is None:
        elevation = 10  # Default elevation (assume low-lying)
    
    # Extract weather indicators
    max_chance_of_rain = 0
    avg_chance_of_rain = 0
    will_rain_soon = False
    current_humidity = None
    max_humidity = 0
    
    if weather_indicators:
        max_chance_of_rain = weather_indicators.get("max_chance_of_rain", 0)
        avg_chance_of_rain = weather_indicators.get("avg_chance_of_rain", 0)
        will_rain_soon = weather_indicators.get("will_rain_soon", False)
        current_humidity = weather_indicators.get("current_humidity")
        max_humidity = weather_indicators.get("max_humidity", 0)
    
    # Adjust precipitation based on chance of rain if current precipitation is low
    # Adjust if chance of rain is moderate-high (>60%) and will rain soon
    effective_precipitation = precipitation
    if precipitation < 5 and max_chance_of_rain > 60 and will_rain_soon:
        # High chance of rain and will rain soon - adjust risk upward
        effective_precipitation = max(precipitation, 5.0)  # Treat as light rain
    elif precipitation < 5 and max_chance_of_rain > 55 and will_rain_soon:
        effective_precipitation = max(precipitation, 3.0)  # Treat as very light rain
    
    # High humidity (>80%) increases flood risk when combined with actual rain (>5mm)
    high_humidity_risk = ((current_humidity is not None and current_humidity > 80) or max_humidity > 80) and precipitation > 5
    
    # High risk: 
    # - heavy rain (>10mm) + very low elevation (<5m)
    # - very heavy rain (>15mm) + low elevation (<10m)
    # - moderate rain (>8mm) + high chance (>60%) + will rain soon + very low elevation (<5m) + high humidity (>80%)
    if (effective_precipitation > 10 and elevation < 5) or \
       (effective_precipitation > 15 and elevation < 10) or \
       (effective_precipitation > 8 and max_chance_of_rain > 60 and will_rain_soon and elevation < 5 and high_humidity_risk):
        return "High"
    
    # Medium High: 
    # - moderate rain (8-10mm) + very low elevation (<5m)
    # - heavy rain (10-15mm) + low elevation (<10m)
    # - light-moderate rain (>6mm) + high chance (>55%) + will rain soon + low elevation (<10m)
    elif (effective_precipitation > 8 and elevation < 5) or \
         (effective_precipitation > 10 and elevation < 10) or \
         (effective_precipitation > 6 and max_chance_of_rain > 55 and will_rain_soon and elevation < 10):
        return "Medium High"
    
    # Medium: 
    # - light-moderate rain (5-8mm) + low elevation (<10m)
    # - moderate rain (8-10mm) + medium elevation (10-15m)
    # - light rain (>4mm) + moderate-high chance (>50%) + will rain soon + medium elevation (<15m)
    elif (effective_precipitation > 5 and elevation < 10) or \
         (effective_precipitation > 8 and elevation < 15) or \
         (effective_precipitation > 4 and max_chance_of_rain > 50 and will_rain_soon and elevation < 15):
        return "Medium"
    
    # Medium Low: 
    # - very light rain (2-5mm) + low elevation (<15m)
    # - light rain (4-8mm) + medium-high elevation (>15m)
    # - any rain (>1mm) + moderate chance (>40%)
    elif (effective_precipitation > 2 and elevation < 15) or \
         (effective_precipitation > 4 and elevation >= 15) or \
         (precipitation > 1 and max_chance_of_rain > 40):
        return "Medium Low"
    
    # Low: very little rain (<2mm) or low chance of rain (<40%) or high elevation
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
    route_coords: List[List[float]],
    target_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Calculate flood risk for multiple points along the route.
    Divides route into 1km segments and checks flood risk for each segment.
    Gets weather and elevation data for each point along the route.
    
    Args:
        route_coords: List of route coordinates
        target_time: Target datetime to forecast for. If provided, forecasts at that time.
                    If None, uses current time (default 6 hours forecast).
    
    Returns:
        List of route segments with risk levels and colors.
    """
    if not route_coords or len(route_coords) < 2:
        return []
    
    route_segments = []
    current_distance = 0.0  # Track cumulative distance in km
    segment_start_idx = 0
    segment_coords_accumulator = [route_coords[0]]  # Accumulate coordinates for current 1km segment
    
    # Process route point by point, creating 1km segments
    for i in range(1, len(route_coords)):
        prev_point = route_coords[i - 1]
        curr_point = route_coords[i]
        
        # Calculate distance from previous point
        segment_distance = haversine_distance(prev_point, curr_point)
        current_distance += segment_distance
        
        # Add current point to accumulator
        segment_coords_accumulator.append(curr_point)
        
        # When we've accumulated 1km, process this segment
        if current_distance >= 1.0:
            # Get midpoint of this 1km segment for weather and elevation check
            mid_idx = len(segment_coords_accumulator) // 2
            mid_point = segment_coords_accumulator[mid_idx]
            mid_lat = mid_point[1]
            mid_lng = mid_point[0]
            
            # Get weather data for this specific point along the route using weatherapi.com
            weather_data = await get_weather_data_weatherapi(mid_lat, mid_lng)
            
            # Get elevation for this point
            elevation = await get_elevation_data(mid_lat, mid_lng)
            
            # Extract precipitation and weather indicators from weatherapi.com format
            # If target_time is provided, forecast at that time, otherwise use current time + 2 hours
            current_precip, forecast_precip_values, weather_indicators = extract_precipitation_from_weatherapi(
                weather_data, 
                hours_ahead=2,
                target_time=target_time
            )
            
            # Calculate precipitation for risk assessment
            # Use average of current + next 2 hours (more realistic than max)
            if forecast_precip_values:
                avg_precip = sum(forecast_precip_values) / len(forecast_precip_values)
                # Use the higher of current or average to catch immediate risk
                precipitation = max(current_precip, avg_precip)
            else:
                precipitation = current_precip
            
            # Classify flood risk using LLM based on this point's data (now with weather indicators)
            risk_level = await classify_flood_risk_with_llm(precipitation, elevation, weather_indicators)
            color = get_flood_risk_color(risk_level)
            
            # Debug logging with weather indicators
            chance_of_rain = weather_indicators.get("max_chance_of_rain", 0) if weather_indicators else 0
            will_rain = weather_indicators.get("will_rain_soon", False) if weather_indicators else False
            humidity = weather_indicators.get("current_humidity") if weather_indicators else None
            print(f"[DEBUG] Segment {len(route_segments)+1} (~{len(route_segments)+1}km): lat={mid_lat:.4f}, lng={mid_lng:.4f}, "
                  f"elevation={elevation}, precipitation={precipitation:.2f}mm, "
                  f"chance_of_rain={chance_of_rain}%, will_rain={will_rain}, humidity={humidity}%, "
                  f"risk={risk_level}")
            
            route_segments.append({
                "coordinates": segment_coords_accumulator.copy(),  # Full segment coordinates
                "riskLevel": risk_level,
                "color": color,
                "elevation": elevation,
                "precipitation": precipitation,
                "latitude": mid_lat,
                "longitude": mid_lng,
            })
            
            # Reset for next 1km segment
            current_distance = 0.0
            segment_start_idx = i
            segment_coords_accumulator = [curr_point]  # Start new segment from current point
    
    # Process remaining segment if any (less than 1km at the end)
    if len(segment_coords_accumulator) > 1 and current_distance > 0:
        # Get midpoint of remaining segment
        mid_idx = len(segment_coords_accumulator) // 2
        mid_point = segment_coords_accumulator[mid_idx]
        mid_lat = mid_point[1]
        mid_lng = mid_point[0]
        
        # Get weather data for this specific point along the route using weatherapi.com
        weather_data = await get_weather_data_weatherapi(mid_lat, mid_lng)
        
        # Get elevation for this point
        elevation = await get_elevation_data(mid_lat, mid_lng)
        
        # Extract precipitation and weather indicators from weatherapi.com format
        current_precip, forecast_precip_values, weather_indicators = extract_precipitation_from_weatherapi(
            weather_data, 
            hours_ahead=2,
            target_time=target_time
        )
        
        # Calculate precipitation for risk assessment
        if forecast_precip_values:
            avg_precip = sum(forecast_precip_values) / len(forecast_precip_values)
            precipitation = max(current_precip, avg_precip)
        else:
            precipitation = current_precip
        
        # Classify flood risk
        risk_level = await classify_flood_risk_with_llm(precipitation, elevation, weather_indicators)
        color = get_flood_risk_color(risk_level)
        
        # Debug logging
        chance_of_rain = weather_indicators.get("max_chance_of_rain", 0) if weather_indicators else 0
        will_rain = weather_indicators.get("will_rain_soon", False) if weather_indicators else False
        humidity = weather_indicators.get("current_humidity") if weather_indicators else None
        print(f"[DEBUG] Final segment (~{len(route_segments)+1}km, {current_distance:.2f}km): lat={mid_lat:.4f}, lng={mid_lng:.4f}, "
              f"elevation={elevation}, precipitation={precipitation:.2f}mm, "
              f"chance_of_rain={chance_of_rain}%, will_rain={will_rain}, humidity={humidity}%, "
              f"risk={risk_level}")
        
        route_segments.append({
            "coordinates": segment_coords_accumulator.copy(),
            "riskLevel": risk_level,
            "color": color,
            "elevation": elevation,
            "precipitation": precipitation,
            "latitude": mid_lat,
            "longitude": mid_lng,
        })
    
    return route_segments


def extract_6h_forecast(weather_data: Dict[str, Any], target_time: Optional[datetime] = None) -> Dict[str, Any]:
    """Extract detailed 6-hour forecast data for LLM analysis.
    
    Supports both weatherapi.com and open-meteo formats.
    Prefers weatherapi.com format if available (more accurate precipitation).
    
    Args:
        weather_data: Weather data from API
        target_time: Target datetime to forecast from. If provided, forecasts from that time.
                    If None, uses current time (default 6 hours from now).
    """
    forecast = {
        "current": {},
        "next6h": [],
        "summary": {},
        "dataSource": "unknown"  # Track which data source we're using
    }
    
    # Check if this is weatherapi.com format
    if "current" in weather_data and "precip_mm" in weather_data.get("current", {}):
        # weatherapi.com format
        current = weather_data.get("current", {})
        forecast_data = weather_data.get("forecast", {})
        forecastday_list = forecast_data.get("forecastday", [])
        
        # Get all hours from forecast
        all_hours = []
        for day_data in forecastday_list[:2]:  # Today and tomorrow
            hours = day_data.get("hour", [])
            all_hours.extend(hours)
        
        # Find target hour if target_time is provided
        start_index = 0
        if target_time:
            target_hour_str = target_time.strftime("%Y-%m-%d %H:00")
            for i, hour_data in enumerate(all_hours):
                hour_time = hour_data.get("time", "")
                if hour_time.startswith(target_hour_str[:13]):  # Match date and hour
                    start_index = i
                    # Use target hour's data as "current"
                    forecast["current"] = {
                        "temperature": hour_data.get("temp_c"),
                        "humidity": hour_data.get("humidity"),
                        "precipitation": hour_data.get("precip_mm", 0.0),  # mm
                        "windSpeed": hour_data.get("wind_kph"),  # km/h
                        "condition": hour_data.get("condition", {}).get("text"),
                        "time": hour_data.get("time"),
                    }
                    break
            else:
                # If target hour not found, use current
                forecast["current"] = {
                    "temperature": current.get("temp_c"),
                    "humidity": current.get("humidity"),
                    "precipitation": current.get("precip_mm", 0.0),  # mm
                    "windSpeed": current.get("wind_kph"),  # km/h
                    "condition": current.get("condition", {}).get("text"),
                    "time": current.get("last_updated"),
                }
        else:
            # Current conditions (no target_time)
            forecast["current"] = {
                "temperature": current.get("temp_c"),
                "humidity": current.get("humidity"),
                "precipitation": current.get("precip_mm", 0.0),  # mm
                "windSpeed": current.get("wind_kph"),  # km/h
                "condition": current.get("condition", {}).get("text"),
                "time": current.get("last_updated"),
            }
        
        # Get next 6 hours from start_index
        next_6h = all_hours[start_index:start_index + 6]
        forecast["dataSource"] = "weatherapi_hourly"
        forecast["next6h"] = [
            {
                "time": hour_data.get("time"),
                "precipitation": hour_data.get("precip_mm", 0.0),  # mm
                "temperature": hour_data.get("temp_c"),
                "humidity": hour_data.get("humidity"),
                "willItRain": hour_data.get("will_it_rain", 0),
                "chanceOfRain": hour_data.get("chance_of_rain", 0),
                "condition": hour_data.get("condition", {}).get("text"),
            }
            for hour_data in next_6h
        ]
        
        # Summary statistics
        precip_values = [h.get("precip_mm", 0.0) for h in next_6h]
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
            forecast["summary"] = {
                "totalPrecipitation": 0,
                "maxPrecipitation": 0,
                "avgPrecipitation": 0,
                "precipitationTrend": "unknown",
                "rainyIntervals": 0,
                "totalIntervals": 0,
            }
        
        return forecast
    
    # Fallback to open-meteo format (legacy support)
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
    """Calculate when rain will stop based on forecast.
    
    Supports both weatherapi.com and open-meteo formats.
    """
    # Check if this is weatherapi.com format
    if "current" in weather_data and "precip_mm" in weather_data.get("current", {}):
        # weatherapi.com format
        current = weather_data.get("current", {})
        current_precip = current.get("precip_mm", 0.0)
        
        if current_precip == 0:
            # Check if rain is coming
            forecast = weather_data.get("forecast", {})
            forecastday_list = forecast.get("forecastday", [])
            
            all_hours = []
            for day_data in forecastday_list[:2]:  # Today and tomorrow
                hours = day_data.get("hour", [])
                all_hours.extend(hours)
            
            # Find first hour with rain
            for i, hour_data in enumerate(all_hours[:24]):  # Check next 24 hours
                precip = hour_data.get("precip_mm", 0.0)
                will_it_rain = hour_data.get("will_it_rain", 0)
                if precip > 0 or will_it_rain == 1:
                    if i == 0:
                        return "Đang mưa"
                    return f"Mưa dự kiến bắt đầu sau {i} giờ"
            
            return "Không mưa"
        
        # Currently raining - find when it stops
        forecast = weather_data.get("forecast", {})
        forecastday_list = forecast.get("forecastday", [])
        
        all_hours = []
        for day_data in forecastday_list[:2]:  # Today and tomorrow
            hours = day_data.get("hour", [])
            all_hours.extend(hours)
        
        # Find when rain stops (first hour with no rain after current)
        for i, hour_data in enumerate(all_hours[:24]):  # Check next 24 hours
            precip = hour_data.get("precip_mm", 0.0)
            will_it_rain = hour_data.get("will_it_rain", 0)
            
            if i > 0 and precip == 0 and will_it_rain == 0:
                # Check if previous hours had rain
                prev_had_rain = any(
                    h.get("precip_mm", 0.0) > 0 or h.get("will_it_rain", 0) == 1
                    for h in all_hours[:i]
                )
                if prev_had_rain:
                    if i < 24:
                        return f"{i} giờ nữa"
        
        return "Hơn 24 giờ"
    
    # Fallback to open-meteo format (legacy support)
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
    time_start_str = payload.time_start

    # Parse time_start to datetime
    target_time = parse_time_start(time_start_str)
    if target_time:
        print(f"[DEBUG] Target time parsed: {target_time} (from '{time_start_str}')")
    else:
        print(f"[DEBUG] No target time specified, using default 6-hour forecast")

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
    # If target_time is provided, forecast at that time, otherwise use current time (6-hour forecast)
    route_segments = await calculate_route_flood_risks(route_coords, target_time=target_time)
    
    # Get overall weather data from midpoint for summary info using weatherapi.com
    dest_lat, dest_lng = destination_coords[1], destination_coords[0]
    start_lat, start_lng = start_coords[1], start_coords[0]
    mid_lat = (start_lat + dest_lat) / 2
    mid_lng = (start_lng + dest_lng) / 2
    
    weather_data = await get_weather_data_weatherapi(mid_lat, mid_lng)
    
    # Extract precipitation at target time if provided, otherwise use current time
    # If target_time is provided, get data at that time, otherwise use current + 6 hours forecast
    if target_time:
        # Get precipitation at target time
        current_precip, forecast_precip, _ = extract_precipitation_from_weatherapi(
            weather_data, 
            hours_ahead=2,  # 2 hours from target time
            target_time=target_time
        )
        current_precipitation = current_precip
        # Get max from forecast starting at target time
        max_precipitation = max(forecast_precip) if forecast_precip else current_precipitation
    else:
        # Default: use current time with 6-hour forecast
        current_precip, forecast_precip, _ = extract_precipitation_from_weatherapi(
            weather_data, 
            hours_ahead=6  # 6 hours from now
        )
        current_precipitation = current_precip
        max_precipitation = max(forecast_precip) if forecast_precip else current_precipitation
    
    # Extract 6-hour forecast for LLM analysis (from target_time if provided, otherwise from now)
    forecast_6h = extract_6h_forecast(weather_data, target_time=target_time)
    
    # Calculate rain stop ETA
    rain_stop_eta = calculate_rain_stop_eta(weather_data)
    
    # Debug logging to check actual data
    current_weather = weather_data.get("current", {})
    forecast = weather_data.get("forecast", {})
    forecastday_list = forecast.get("forecastday", [])
    print(f"[DEBUG] WeatherAPI data check:")
    print(f"  - Target time: {target_time}")
    print(f"  - current.precip_mm: {current_weather.get('precip_mm')}")
    print(f"  - current.humidity: {current_weather.get('humidity')}")
    print(f"  - current.last_updated: {current_weather.get('last_updated')}")
    print(f"  - forecast days: {len(forecastday_list)}")
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
    # Note: weatherInfo removed from structuredContent - only shown in response text
    response_data = {
        "start": list(start_coords),
        "destination": list(destination_coords),
        "route": route_coords,
        "routeSegments": route_segments,  # Segments with risk levels and colors
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
    
    # Format target time info
    time_info = ""
    if target_time:
        time_diff = target_time - datetime.now()
        if time_diff.total_seconds() > 0:
            hours = int(time_diff.total_seconds() // 3600)
            minutes = int((time_diff.total_seconds() % 3600) // 60)
            if hours > 0:
                time_info = f"\n⏰ **Thời gian dự kiến đi:** {target_time.strftime('%H:%M ngày %d/%m')} ({hours} giờ {minutes} phút nữa)"
            else:
                time_info = f"\n⏰ **Thời gian dự kiến đi:** {target_time.strftime('%H:%M ngày %d/%m')} ({minutes} phút nữa)"
    
    response_text = f"""Phân tích tuyến đường từ {start_location_name or 'điểm xuất phát'} đến {destination_name}:{time_info}

📊 **Tình trạng{' tại thời điểm đi' if target_time else ' hiện tại'}:**
- Lượng mưa: {current_precipitation:.1f}mm
- Nhiệt độ: {temp_str}
- Độ ẩm: {humidity_str}
- Mưa dự kiến tạnh: {rain_stop_eta or 'Không có thông tin'}

📈 **Dự báo{' từ thời điểm đi' if target_time else ' 6 giờ tới'}:**
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

💡 **Dữ liệu chi tiết 6 giờ tới đã được cung cấp trong structuredContent để bạn có thể đưa ra lời khuyên cụ thể cho người dùng.**

---

## 📋 **Kết luận:**

Khi bạn đi từ **{start_location_name or 'điểm xuất phát'}** đến **{destination_name}**{' lúc ' + target_time.strftime('%H:%M ngày %d/%m') if target_time else ''}, quãng đường khoảng **{route_info.get('distance', 'N/A')}** và thời gian di chuyển dự kiến là **{route_info.get('duration', 'N/A')}**, bạn sẽ đến nơi vào khoảng **{_calculate_arrival_time(target_time, route_info.get('duration', ''))}**.

{f'⚠️ **Cảnh báo:** Trên lộ trình hiện tại, có đoạn có nguy cơ ngập cao (màu đỏ, High). Bạn nên cân nhắc đi chậm hoặc chọn tuyến đường khác nếu có thể.' if risk_distribution['High'] > 0 else '✅ Tuyến đường này không có đoạn nào có nguy cơ ngập cao.'}"""

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

