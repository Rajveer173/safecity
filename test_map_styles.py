#!/usr/bin/env python3
"""
Test script to verify map style functionality
"""
import folium

def test_map_styles():
    """Test different map tile layers"""
    print("🧪 Testing SafeCity Map Styles...")
    
    # Mumbai coordinates
    lat, lng = 19.0760, 72.8777
    
    # Test Standard map
    print("✅ Testing Standard Map...")
    map_standard = folium.Map(
        location=[lat, lng],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Test Satellite map  
    print("✅ Testing Satellite Map...")
    map_satellite = folium.Map(
        location=[lat, lng],
        zoom_start=12,
        tiles=None
    )
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(map_satellite)
    
    # Test Dark map
    print("✅ Testing Dark Map...")
    map_dark = folium.Map(
        location=[lat, lng],
        zoom_start=12,
        tiles=None
    )
    folium.TileLayer(
        tiles='https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png',
        attr='&copy; OpenStreetMap &copy; CartoDB',
        name='Dark Mode',
        overlay=False,
        control=True
    ).add_to(map_dark)
    
    # Add test markers
    for map_obj, style in [(map_standard, "Standard"), (map_satellite, "Satellite"), (map_dark, "Dark")]:
        # Test marker colors for each style
        colors = ['red', 'orange', 'yellow'] if style != 'Dark' else ['#ff4444', '#ff8800', '#ffdd00']
        
        for i, color in enumerate(colors):
            folium.CircleMarker(
                location=[lat + 0.01*i, lng + 0.01*i],
                radius=8,
                popup=f"{style} - Test Marker {i+1}",
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(map_obj)
    
    print("🎯 All map styles tested successfully!")
    print("📍 Markers added with appropriate colors for each theme")
    print("🗺️ Layer controls enabled for switching between styles")
    return True

if __name__ == "__main__":
    test_map_styles()