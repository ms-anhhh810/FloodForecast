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
        {(isFlooded || routeInfo) && (
          <div className="absolute top-4 left-4 z-30 bg-white/95 backdrop-blur-sm rounded-xl shadow-xl p-5 max-w-sm pointer-events-auto border border-gray-100">
            {isFlooded && (
              <div className="flex items-center gap-2 text-red-600 mb-4 pb-4 border-b border-red-100">
                <AlertTriangle className="h-5 w-5" />
                <span className="font-semibold">Có nguy cơ ngập lụt!</span>
              </div>
            )}
            {routeInfo && (
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2 mb-2">
                  <Navigation className="h-4 w-4 text-blue-600" />
                  <span className="font-semibold text-gray-800">Thông tin tuyến đường:</span>
                </div>
                {routeInfo.distance && (
                  <div className="text-gray-700 pl-6">
                    <span className="text-gray-500">Khoảng cách:</span> <span className="font-medium">{routeInfo.distance}</span>
                  </div>
                )}
                {routeInfo.duration && (
                  <div className="text-gray-700 pl-6">
                    <span className="text-gray-500">Thời gian:</span> <span className="font-medium">{routeInfo.duration}</span>
                  </div>
                )}
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
        {(startPoint || endPoint) && (
          <div className="absolute bottom-4 left-4 z-30 bg-white/95 backdrop-blur-sm rounded-xl shadow-xl p-4 text-sm pointer-events-auto border border-gray-100">
            <div className="space-y-2.5">
              {startPoint && (
                <div className="flex items-center gap-2.5">
                  <div className="w-4 h-4 bg-green-500 rounded-full shadow-sm ring-2 ring-green-200"></div>
                  <span className="text-gray-700 font-medium">Điểm xuất phát</span>
                </div>
              )}
              {endPoint && (
                <div className="flex items-center gap-2.5">
                  <div className="w-4 h-4 bg-red-500 rounded-full shadow-sm ring-2 ring-red-200"></div>
                  <span className="text-gray-700 font-medium">Điểm đến</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
}

createRoot(document.getElementById("flood-route-root")).render(<App />);

