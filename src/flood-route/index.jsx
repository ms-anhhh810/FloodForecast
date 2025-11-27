import React, { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import { createRoot } from "react-dom/client";
import { useOpenAiGlobal } from "../use-openai-global";
import { useMaxHeight } from "../use-max-height";
import { Maximize2, Navigation, AlertTriangle } from "lucide-react";

mapboxgl.accessToken =
  "pk.eyJ1IjoiZXJpY25pbmciLCJhIjoiY21icXlubWM1MDRiczJvb2xwM2p0amNyayJ9.n-3O6JI5nOp_Lw96ZO5vJQ";

export default function App() {
  const mapRef = useRef(null);
  const mapObj = useRef(null);
  const routeLayerRef = useRef(null);
  const floodZonesRef = useRef([]);
  const markersRef = useRef([]);
  const toolInput = useOpenAiGlobal("toolInput");
  const toolOutput = useOpenAiGlobal("toolOutput");
  
  const [routeData, setRouteData] = useState(null);
  const [routeSegments, setRouteSegments] = useState([]);
  const [floodZones, setFloodZones] = useState([]);
  const [startPoint, setStartPoint] = useState(null);
  const [endPoint, setEndPoint] = useState(null);
  const [isFlooded, setIsFlooded] = useState(false);
  const [routeInfo, setRouteInfo] = useState(null);
  const [weatherInfo, setWeatherInfo] = useState(null);

  const displayMode = useOpenAiGlobal("displayMode");
  const maxHeight = useMaxHeight() ?? undefined;

  // Parse data from tool output (structuredContent)
  useEffect(() => {
    if (toolOutput) {
      // toolOutput contains the structuredContent from the tool response
      const data = toolOutput;
      
      if (data.start) {
        setStartPoint(data.start);
      }
      if (data.destination) {
        setEndPoint(data.destination);
      }
      if (data.route) {
        setRouteData(data.route);
      }
      if (data.routeSegments) {
        setRouteSegments(data.routeSegments);
      }
      if (data.floodZones) {
        setFloodZones(data.floodZones);
      }
      if (data.hasHighRisk !== undefined) {
        setIsFlooded(data.hasHighRisk);
      }
      if (data.routeInfo) {
        setRouteInfo(data.routeInfo);
      }
      if (data.weatherInfo) {
        setWeatherInfo(data.weatherInfo);
      }
    }
  }, [toolOutput]);

  // Initialize map
  useEffect(() => {
    if (mapObj.current) return;
    
    const center = startPoint || endPoint || [106.6896, 10.7823]; // Default to Ho Chi Minh City
    
    mapObj.current = new mapboxgl.Map({
      container: mapRef.current,
      style: "mapbox://styles/mapbox/streets-v12",
      center: center,
      zoom: 12,
      attributionControl: false,
    });

    mapObj.current.on("load", () => {
      // Route segments will be added dynamically with different colors

      // Add flood zones source and layer
      mapObj.current.addSource("flood-zones", {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: [],
        },
      });

      mapObj.current.addLayer({
        id: "flood-zones",
        type: "fill",
        source: "flood-zones",
        paint: {
          "fill-color": "#ef4444",
          "fill-opacity": 0.3,
        },
      });

      mapObj.current.addLayer({
        id: "flood-zones-outline",
        type: "line",
        source: "flood-zones",
        paint: {
          "line-color": "#dc2626",
          "line-width": 2,
          "line-opacity": 0.6,
        },
      });

      updateMap();
    });

    requestAnimationFrame(() => mapObj.current?.resize());
    window.addEventListener("resize", () => mapObj.current?.resize());

    return () => {
      window.removeEventListener("resize", () => mapObj.current?.resize());
      mapObj.current?.remove();
    };
  }, []);

  // Update map when data changes
  useEffect(() => {
    if (mapObj.current?.loaded()) {
      updateMap();
    }
  }, [routeData, routeSegments, floodZones, startPoint, endPoint]);

  function updateMap() {
    if (!mapObj.current || !mapObj.current.loaded()) return;

    // Clear existing markers
    markersRef.current.forEach((m) => m.remove());
    markersRef.current = [];

    // Add start marker
    if (startPoint) {
      const startMarker = new mapboxgl.Marker({ color: "#10b981" })
        .setLngLat(startPoint)
        .addTo(mapObj.current);
      markersRef.current.push(startMarker);
    }

    // Add end marker
    if (endPoint) {
      const endMarker = new mapboxgl.Marker({ color: "#ef4444" })
        .setLngLat(endPoint)
        .addTo(mapObj.current);
      markersRef.current.push(endMarker);
    }

    // Remove old route layers
    for (let i = 0; i < 100; i++) {
      const layerId = `route-segment-${i}`;
      const sourceId = `route-segment-source-${i}`;
      if (mapObj.current.getLayer(layerId)) {
        mapObj.current.removeLayer(layerId);
      }
      if (mapObj.current.getSource(sourceId)) {
        mapObj.current.removeSource(sourceId);
      }
    }

    // Add route segments with different colors
    if (routeSegments && routeSegments.length > 0) {
      routeSegments.forEach((segment, index) => {
        const sourceId = `route-segment-source-${index}`;
        const layerId = `route-segment-${index}`;
        
        if (!mapObj.current.getSource(sourceId)) {
          mapObj.current.addSource(sourceId, {
            type: "geojson",
            data: {
              type: "Feature",
              geometry: {
                type: "LineString",
                coordinates: segment.coordinates,
              },
              properties: {
                riskLevel: segment.riskLevel,
              },
            },
          });

          mapObj.current.addLayer({
            id: layerId,
            type: "line",
            source: sourceId,
            layout: {
              "line-join": "round",
              "line-cap": "round",
            },
            paint: {
              "line-color": segment.color || "#3b82f6",
              "line-width": 5,
              "line-opacity": 0.9,
            },
          });
        } else {
          mapObj.current.getSource(sourceId).setData({
            type: "Feature",
            geometry: {
              type: "LineString",
              coordinates: segment.coordinates,
            },
            properties: {
              riskLevel: segment.riskLevel,
            },
          });
        }
      });
    } else if (routeData) {
      // Fallback to single route if no segments
      const sourceId = "route-fallback";
      if (!mapObj.current.getSource(sourceId)) {
        mapObj.current.addSource(sourceId, {
          type: "geojson",
          data: {
            type: "Feature",
            geometry: {
              type: "LineString",
              coordinates: routeData,
            },
          },
        });

        mapObj.current.addLayer({
          id: "route-fallback",
          type: "line",
          source: sourceId,
          layout: {
            "line-join": "round",
            "line-cap": "round",
          },
          paint: {
            "line-color": "#3b82f6",
            "line-width": 5,
            "line-opacity": 0.9,
          },
        });
      } else {
        mapObj.current.getSource(sourceId).setData({
          type: "Feature",
          geometry: {
            type: "LineString",
            coordinates: routeData,
          },
        });
      }
    }

    // Update flood zones
    if (floodZones.length > 0 && mapObj.current.getSource("flood-zones")) {
      const floodGeoJson = {
        type: "FeatureCollection",
        features: floodZones.map((zone) => ({
          type: "Feature",
          geometry: {
            type: "Polygon",
            coordinates: [zone],
          },
          properties: {
            risk: zone.risk || "high",
          },
        })),
      };
      mapObj.current.getSource("flood-zones").setData(floodGeoJson);
    }

    // Fit bounds
    if (startPoint && endPoint) {
      const bounds = new mapboxgl.LngLatBounds(startPoint, startPoint);
      bounds.extend(endPoint);
      mapObj.current.fitBounds(bounds, { padding: 60, animate: true });
    } else if (endPoint) {
      mapObj.current.flyTo({ center: endPoint, zoom: 14 });
    }
  }

  useEffect(() => {
    if (!mapObj.current) return;
    mapObj.current.resize();
  }, [maxHeight, displayMode]);

  return (
    <>
      <div
        style={{
          maxHeight,
          height: displayMode === "fullscreen" ? maxHeight - 40 : 480,
        }}
        className={
          "relative antialiased w-full min-h-[480px] overflow-hidden " +
          (displayMode === "fullscreen"
            ? "rounded-none border-0"
            : "border border-black/10 dark:border-white/10 rounded-2xl sm:rounded-3xl")
        }
      >
        {displayMode !== "fullscreen" && (
          <button
            aria-label="Enter fullscreen"
            className="absolute top-4 right-4 z-30 rounded-full bg-white text-black shadow-lg ring ring-black/5 p-2.5 pointer-events-auto"
            onClick={() => {
              if (window?.webplus?.requestDisplayMode) {
                window.webplus.requestDisplayMode({ mode: "fullscreen" });
              }
            }}
          >
            <Maximize2
              strokeWidth={1.5}
              className="h-4.5 w-4.5"
              aria-hidden="true"
            />
          </button>
        )}

        {/* Info Panel */}
        {(isFlooded || routeInfo || weatherInfo) && (
          <div className="absolute top-4 left-4 z-30 bg-white rounded-lg shadow-lg p-4 max-w-sm pointer-events-auto">
            {isFlooded && (
              <div className="flex items-center gap-2 text-red-600 mb-3">
                <AlertTriangle className="h-5 w-5" />
                <span className="font-semibold">Có nguy cơ ngập lụt!</span>
              </div>
            )}
            {weatherInfo && (
              <div className="space-y-2 text-sm mb-3 border-b pb-3">
                <div className="font-medium text-gray-800">Thông tin thời tiết:</div>
                <div className="text-gray-600">
                  Lượng mưa hiện tại: <span className="font-semibold">{weatherInfo.currentPrecipitation}mm</span>
                </div>
                <div className="text-gray-600">
                  Lượng mưa tối đa: <span className="font-semibold">{weatherInfo.maxPrecipitation}mm</span>
                </div>
                {weatherInfo.rainStopETA && (
                  <div className="text-blue-600">
                    Mưa dự kiến tạnh: <span className="font-semibold">{weatherInfo.rainStopETA}</span>
                  </div>
                )}
              </div>
            )}
            {routeInfo && (
              <div className="space-y-1 text-sm mb-3">
                <div className="flex items-center gap-2">
                  <Navigation className="h-4 w-4 text-blue-600" />
                  <span className="font-medium">Thông tin tuyến đường:</span>
                </div>
                {routeInfo.distance && (
                  <div className="text-gray-600">
                    Khoảng cách: {routeInfo.distance}
                  </div>
                )}
                {routeInfo.duration && (
                  <div className="text-gray-600">
                    Thời gian: {routeInfo.duration}
                  </div>
                )}
              </div>
            )}
            {routeSegments && routeSegments.length > 0 && (
              <div className="mt-3 pt-3 border-t text-xs">
                <div className="font-medium text-gray-800 mb-2">Mức độ nguy cơ:</div>
                <div className="flex flex-wrap gap-2">
                  {["High", "Medium High", "Medium", "Medium Low", "Low"].map((level) => {
                    const colorMap = {
                      "High": "#dc2626",
                      "Medium High": "#f97316",
                      "Medium": "#eab308",
                      "Medium Low": "#3b82f6",
                      "Low": "#60a5fa",
                    };
                    const hasLevel = routeSegments.some(s => s.riskLevel === level);
                    if (!hasLevel) return null;
                    return (
                      <div key={level} className="flex items-center gap-1">
                        <div 
                          className="w-3 h-3 rounded-full" 
                          style={{ backgroundColor: colorMap[level] }}
                        ></div>
                        <span className="text-gray-600">{level}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Map */}
        <div className="absolute inset-0 overflow-hidden">
          <div
            ref={mapRef}
            className="w-full h-full"
            style={{
              maxHeight,
              height: displayMode === "fullscreen" ? maxHeight : undefined,
            }}
          />
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 z-30 bg-white rounded-lg shadow-lg p-3 text-xs pointer-events-auto">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-500 rounded-full"></div>
              <span>Điểm xuất phát</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-500 rounded-full"></div>
              <span>Điểm đến</span>
            </div>
            <div className="mt-2 pt-2 border-t">
              <div className="font-medium mb-1">Mức độ nguy cơ:</div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-1 bg-red-600"></div>
                  <span>High</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-1 bg-orange-500"></div>
                  <span>Medium High</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-1 bg-yellow-500"></div>
                  <span>Medium</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-1 bg-blue-500"></div>
                  <span>Medium Low</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-1 bg-blue-300"></div>
                  <span>Low</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

createRoot(document.getElementById("flood-route-root")).render(<App />);

