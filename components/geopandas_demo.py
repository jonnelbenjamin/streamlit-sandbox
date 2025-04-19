# components/geopandas_demo.py
import geopandas as gpd
import numpy as np
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon
from geodatasets import get_path
import streamlit as st

# Data Loading with geodatasets
@st.cache_data
def load_data():
    data = {}
    try:
        path_to_file = get_path('nybb')
        data["NYC Boroughs"] = gpd.read_file(path_to_file)
        data["NYC Boroughs"]["population"] = [1694251, 2648452, 2333054, 1446788, 487155]
    except Exception as e:
        st.warning(f"Couldn't load NYC data: {e}")
        nyc_data = {
            'BoroName': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'],
            'geometry': [
                Polygon([[-74.05, 40.70], [-73.95, 40.70], [-73.95, 40.80], [-74.05, 40.80], [-74.05, 40.70]]),
                Polygon([[-74.05, 40.60], [-73.95, 40.60], [-73.95, 40.70], [-74.05, 40.70], [-74.05, 40.60]]),
                Polygon([[-73.95, 40.70], [-73.85, 40.70], [-73.85, 40.80], [-73.95, 40.80], [-73.95, 40.70]]),
                Polygon([[-73.95, 40.80], [-73.85, 40.80], [-73.85, 40.90], [-73.95, 40.90], [-73.95, 40.80]]),
                Polygon([[-74.20, 40.50], [-74.10, 40.50], [-74.10, 40.60], [-74.20, 40.60], [-74.20, 40.50]])
            ],
            'population': [1694251, 2648452, 2333054, 1446788, 487155]
        }
        data["NYC Boroughs"] = gpd.GeoDataFrame(nyc_data, crs="EPSG:4326")
    
    try:
        cities_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_populated_places.zip"
        data["World Cities"] = gpd.read_file(cities_url)
        # Ensure we have point data
        if data["World Cities"].geom_type[0] != 'Point':
            data["World Cities"].geometry = data["World Cities"].geometry.centroid
    except Exception as e:
        st.warning(f"Couldn't load cities data: {e}")
        cities = {
            'name': ['New York', 'London', 'Tokyo'],
            'geometry': [
                Point(-74.006, 40.7128),
                Point(-0.1278, 51.5074),
                Point(139.6917, 35.6895)
            ]
        }
        data["World Cities"] = gpd.GeoDataFrame(cities, crs="EPSG:4326")
    
    return data

def main():
    st.title("🌍 GeoAnalysis Tool")
    
    # Load data
    data = load_data()
    dataset = st.selectbox("Dataset", list(data.keys()))
    gdf = data[dataset].copy()  # Create a copy to avoid modifying cached data
    
    # Data summary
    with st.expander("🔍 Data Summary", expanded=True):
        cols = st.columns(3)
        cols[0].metric("Features", len(gdf))
        cols[1].metric("CRS", gdf.crs.to_epsg())
        cols[2].metric("Type", gdf.geom_type[0])
        st.dataframe(gdf.drop('geometry', axis=1).head(3), height=120)
    
    # Visualization
    viz_type = st.radio("Mode", ["Map", "Heatmap", "3D", "Analysis"], horizontal=True)
    
    if viz_type == "Map":
        col1, col2 = st.columns([1, 3])
        with col1:
            basemap = st.selectbox("Basemap", ["OpenStreetMap", "Stamen Terrain"])
            zoom = st.slider("Zoom", 1, 18, 10)
        
        m = folium.Map(
            location=[40, -95], 
            zoom_start=zoom,
            tiles=basemap,
            attr='Map data © OpenStreetMap contributors'
        )
        
        if gdf.geom_type[0] in ['Polygon', 'MultiPolygon']:
            folium.GeoJson(gdf).add_to(m)
        else:
            for _, row in gdf.iterrows():
                folium.CircleMarker(
                    [row.geometry.y, row.geometry.x],
                    radius=5,
                    color='blue'
                ).add_to(m)
        
        folium_static(m, width=600, height=400)
    
    elif viz_type == "Heatmap":
        from folium.plugins import HeatMap
        
        # Create a copy for heatmap to avoid modifying original data
        heatmap_gdf = gdf.copy()
        
        if heatmap_gdf.geom_type[0] in ['Polygon', 'MultiPolygon']:
            with st.spinner("Converting polygons to centroids..."):
                heatmap_gdf.geometry = heatmap_gdf.geometry.centroid
            st.info("Converted polygons to centroids for heatmap visualization")
        
        m = folium.Map(location=[40, -95], zoom_start=4)
        heat_data = [[point.y, point.x] for point in heatmap_gdf.geometry]
        HeatMap(heat_data).add_to(m)
        folium_static(m, width=600, height=400)
    
    elif viz_type == "3D":
        if gdf.geom_type[0] in ['Polygon', 'MultiPolygon']:
            import pydeck as pdk
            gdf['elevation'] = np.random.randint(100, 1000, len(gdf))
            
            st.pydeck_chart(pdk.Deck(
                layers=[pdk.Layer(
                    "PolygonLayer",
                    gdf,
                    get_polygon="geometry.coordinates",
                    get_fill_color=[255, 140, 0],
                    get_elevation="elevation",
                    elevation_scale=100
                )],
                initial_view_state=pdk.ViewState(
                    latitude=40,
                    longitude=-95,
                    zoom=3,
                    pitch=45
                )
            ))
        else:
            st.warning("3D view requires polygon data")
    
    elif viz_type == "Analysis":
        if st.button("Analyze"):
            with st.spinner("Processing..."):
                if gdf.geom_type[0] in ['Polygon', 'MultiPolygon']:
                    gdf['area'] = gdf.geometry.area / 1e6
                    st.metric("Total Area", f"{gdf['area'].sum():.2f} km²")
                    st.bar_chart(gdf.set_index('BoroName' if 'BoroName' in gdf.columns else 'name')['area'])
                else:
                    st.metric("Points", len(gdf))
                    st.map(gdf)

if __name__ == "__main__":
    main()