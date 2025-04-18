# components/geopandas_demo.py

import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
import folium
from streamlit_folium import folium_static
import os
from urllib.request import urlretrieve
import requests
import json

def download_file(url, local_path):
    """Download file from URL if it doesn't exist locally"""
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            urlretrieve(url, local_path)
            return local_path
        except Exception as e:
            st.error(f"Failed to download file from {url}. Error: {str(e)}")
            return None
    return local_path

def load_sample_data():
    """Load sample datasets from reliable sources with fallbacks"""
    data = {}
    
    # NYC boroughs data - try multiple sources
    nyc_sources = [
        "https://data.cityofnewyork.us/api/geospatial/tqmj-j8zm?method=export&format=GeoJSON",
        "https://raw.githubusercontent.com/giswqs/geemap/master/examples/data/nybb.geojson",
        "https://raw.githubusercontent.com/jupyter-widgets/ipyleaflet/master/examples/nyc.geojson"
    ]
    
    nyc_path = "data/nyc_boroughs.geojson"
    for source in nyc_sources:
        try:
            if download_file(source, nyc_path):
                data["NYC Boroughs"] = gpd.read_file(nyc_path)
                # Add calculated columns
                data["NYC Boroughs"]['area'] = data["NYC Boroughs"].geometry.area
                data["NYC Boroughs"]['population'] = [1694251, 2648452, 2333054, 1446788, 487155]
                break
        except Exception as e:
            continue
    
    # US states data (from Natural Earth)
    try:
        states_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_1_states_provinces.zip"
        states_path = "data/us_states.zip"
        if download_file(states_url, states_path):
            data["US States"] = gpd.read_file(states_path)
            data["US States"]['GDP'] = np.random.randint(100, 1000, size=len(data["US States"]))
    except Exception as e:
        st.warning(f"Couldn't load US States data: {str(e)}")
    
    # World Cities (from Natural Earth)
    try:
        cities_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_populated_places.zip"
        cities_path = "data/world_cities.zip"
        if download_file(cities_url, cities_path):
            data["World Cities"] = gpd.read_file(cities_path)
            data["World Cities"]['name'] = data["World Cities"]['name'].str.replace(r'\[.*\]', '', regex=True)
    except Exception as e:
        st.warning(f"Couldn't load World Cities data: {str(e)}")
    
    # Fallback if no data loaded
    if not data:
        st.warning("Using fallback datasets")
        try:
            data["World Cities"] = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
            data["US States"] = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            data["US States"]['GDP'] = np.random.randint(100, 1000, size=len(data["US States"]))
            
            # Create simple NYC fallback
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
        except:
            st.error("Could not load any sample data. Please check your internet connection.")
    
    return data

def show_geopandas_demo():
    """Enhanced geopandas demonstration with all fixes"""
    
    st.title("🌍 Advanced Geospatial Analysis")
    st.write("Explore geospatial data analysis capabilities using GeoPandas")
    
    # Load data
    sample_data = load_sample_data()
    if not sample_data:
        st.error("No data available")
        return
    
    # Section 1: Data Selection
    st.header("1. Data Selection")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        dataset = st.selectbox("Choose dataset", list(sample_data.keys()))
        gdf = sample_data[dataset]
        
        # CRS handling
        crs_options = {
            "EPSG:4326 (WGS84)": "EPSG:4326",
            "EPSG:3857 (Web Mercator)": "EPSG:3857",
            "EPSG:2263 (NY State Plane)": "EPSG:2263"
        }
        selected_crs = st.selectbox("Transform CRS to:", list(crs_options.keys()))
        gdf = gdf.to_crs(crs_options[selected_crs])
        
        if gdf.geom_type[0] in ['Polygon', 'MultiPolygon']:
            gdf['area'] = gdf.geometry.area
    
    with col2:
        st.write(f"**CRS:** {gdf.crs}")
        st.write(f"**Features:** {len(gdf)}")
        
        tab1, tab2 = st.tabs(["Data", "Map"])
        with tab1:
            st.dataframe(gdf.drop(columns='geometry', errors='ignore'))
        with tab2:
            fig, ax = plt.subplots()
            gdf.plot(ax=ax)
            ax.set_axis_off()
            st.pyplot(fig)
    
    # Section 2: Analysis
    st.header("2. Spatial Analysis")
    
    analysis_type = st.selectbox("Analysis type", 
                               ["Point-in-Polygon", "Spatial Join", "Proximity"])
    
    if analysis_type == "Point-in-Polygon":
        num_points = st.slider("Number of points", 10, 1000, 100)
        
        # Generate points
        minx, miny, maxx, maxy = gdf.total_bounds
        x = np.random.uniform(minx, maxx, num_points)
        y = np.random.uniform(miny, maxy, num_points)
        points = gpd.GeoSeries([Point(xy) for xy in zip(x, y)], crs=gdf.crs)
        
        # Perform join
        joined = gpd.sjoin(gpd.GeoDataFrame(geometry=points), gdf, how='left')
        counts = joined.groupby('index_right').size()
        gdf['point_count'] = counts.reindex(gdf.index, fill_value=0)
        
        # Visualize
        fig, ax = plt.subplots()
        gdf.plot(ax=ax, column='point_count', legend=True)
        points.plot(ax=ax, color='red', markersize=1)
        st.pyplot(fig)
    
    # Section 3: Visualization
    st.header("3. Advanced Visualization")
    
    if st.checkbox("Show interactive map"):
        m = folium.Map(location=[40, -95], zoom_start=4)
        
        if gdf.geom_type[0] in ['Polygon', 'MultiPolygon']:
            folium.GeoJson(gdf).add_to(m)
        else:
            for _, row in gdf.iterrows():
                folium.Marker([row.geometry.y, row.geometry.x]).add_to(m)
        
        folium_static(m, width=800, height=500)

if __name__ == "__main__":
    show_geopandas_demo()