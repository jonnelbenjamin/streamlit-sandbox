# components/geopandas_demo.py
import geopandas as gpd
import numpy as np
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon, LineString
from geodatasets import get_path
import streamlit as st
import json
from folium.plugins import HeatMap, MarkerCluster, MeasureControl, MiniMap
import pydeck as pdk
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import networkx as nx

# --- Comprehensive Data Loading ---
@st.cache_data
def load_data():
    """Load datasets with all advanced metrics and proper error handling"""
    data = {}
    
    # 1. NYC Boroughs with complete attribute set
    try:
        path_to_file = get_path('nybb')
        nyc = gpd.read_file(path_to_file)
        nyc = nyc.to_crs("EPSG:4326")
        
        # Ensure all arrays match length
        n = len(nyc)
        nyc["population"] = [1694251, 2648452, 2333054, 1446788, 487155][:n]
        nyc["area_km2"] = nyc.geometry.area / 1e6
        nyc["density"] = nyc["population"] / nyc["area_km2"]
        nyc["compactness"] = (4 * np.pi * nyc.geometry.area) / (nyc.geometry.length ** 2)
        nyc["centroid"] = nyc.geometry.centroid
        nyc["centroid_coords"] = nyc.geometry.centroid.apply(lambda p: [p.y, p.x])
        data["NYC Boroughs"] = nyc
        
    except Exception as e:
        st.warning(f"NYC data loading failed: {e}")
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
        nyc = gpd.GeoDataFrame(nyc_data, crs="EPSG:4326")
        nyc["area_km2"] = nyc.geometry.area / 1e6
        nyc["density"] = nyc["population"] / nyc["area_km2"]
        nyc["compactness"] = (4 * np.pi * nyc.geometry.area) / (nyc.geometry.length ** 2)
        nyc["centroid"] = nyc.geometry.centroid
        nyc["centroid_coords"] = nyc.geometry.centroid.apply(lambda p: [p.y, p.x])
        data["NYC Boroughs"] = nyc

    # 2. World Cities with full attribute set
    try:
        cities_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_populated_places.zip"
        cities = gpd.read_file(cities_url)
        cities = cities.to_crs("EPSG:4326")
        if cities.geom_type[0] != 'Point':
            cities.geometry = cities.geometry.centroid
        n = len(cities)
        cities["size"] = np.random.randint(5, 20, n)
        cities["importance"] = np.random.uniform(0.5, 1.0, n)
        cities["population"] = np.random.randint(50000, 5000000, n)
        data["World Cities"] = cities
    except Exception as e:
        st.warning(f"Cities data loading failed: {e}")
        cities_data = {
            'name': ['New York', 'London', 'Tokyo'],
            'geometry': [
                Point(-74.006, 40.7128),
                Point(-0.1278, 51.5074),
                Point(139.6917, 35.6895)
            ],
            'size': [15, 12, 18],
            'importance': [0.9, 0.85, 0.95],
            'population': [8419000, 8982000, 13960000]
        }
        data["World Cities"] = gpd.GeoDataFrame(cities_data, crs="EPSG:4326")

    # 3. Transportation Network with complete attributes
    transport_data = {
        'type': ['Highway', 'Rail', 'Subway', 'Bus', 'Bike'],
        'geometry': [
            LineString([(-74.05, 40.70), (-73.95, 40.70), (-73.85, 40.70)]),
            LineString([(-74.00, 40.75), (-73.90, 40.75), (-73.80, 40.75)]),
            LineString([(-74.02, 40.65), (-73.92, 40.65), (-73.82, 40.65)]),
            LineString([(-74.03, 40.68), (-73.98, 40.72), (-73.93, 40.68)]),
            LineString([(-74.01, 40.73), (-73.96, 40.77), (-73.91, 40.73)])
        ],
        'capacity': [1000, 800, 600, 200, 50],
        'length_km': [12.5, 8.3, 6.7, 4.2, 3.1],
        'usage': [0.85, 0.72, 0.91, 0.65, 0.45]
    }
    data["Transport Network"] = gpd.GeoDataFrame(transport_data, crs="EPSG:4326")

    return data

# --- Advanced Spatial Analysis Functions ---
def perform_spatial_analysis(gdf):
    """Complete spatial analysis suite with all original metrics"""
    results = {}
    
    # Basic metrics
    results["Feature Count"] = len(gdf)
    results["CRS"] = str(gdf.crs)
    
    # Geometry-specific metrics
    if gdf.geom_type[0] in ['Polygon', 'MultiPolygon']:
        results["Total Area (km²)"] = gdf.geometry.area.sum() / 1e6
        results["Avg Compactness"] = ((4 * np.pi * gdf.geometry.area) / (gdf.geometry.length ** 2)).mean()
        convex_hull = gdf.unary_union.convex_hull
        results["Convex Hull Area"] = convex_hull.area / 1e6
        results["Shape Complexity"] = gdf.geometry.length.mean()
        
    elif gdf.geom_type[0] == 'Point':
        coords = np.array([[p.x, p.y] for p in gdf.geometry])
        dbscan = DBSCAN(eps=0.1, min_samples=2).fit(coords)
        results["Cluster Count"] = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        vor = Voronoi(coords)
        results["Voronoi Regions"] = len(vor.point_region)
        results["Mean Nearest Neighbor"] = np.mean([np.min(np.linalg.norm(coords - coord, axis=1)) 
                                                  for coord in coords])
        
    elif gdf.geom_type[0] == 'LineString':
        G = nx.Graph()
        for line in gdf.geometry:
            for i in range(len(line.coords)-1):
                G.add_edge(line.coords[i], line.coords[i+1])
        results["Network Nodes"] = G.number_of_nodes()
        results["Network Edges"] = G.number_of_edges()
        results["Network Density"] = nx.density(G)
    
    return results

# --- Complete Visualization Functions ---
def create_interactive_map(gdf):
    """Full-featured interactive map with all original capabilities"""
    m = folium.Map(location=[40, -95], zoom_start=4,
                   tiles='CartoDB positron',
                   control_scale=True)
    
    # Add all map controls
    MiniMap(position='bottomleft').add_to(m)
    MeasureControl(position='topright').add_to(m)
    
    # Handle all geometry types
    if gdf.geom_type[0] in ['Polygon', 'MultiPolygon']:
        geojson_data = json.loads(gdf.to_json())
        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': plt.cm.YlOrRd(feature['properties']['density']/gdf['density'].max()) 
                             if 'density' in gdf.columns else '#3186cc',
                'color': '#000000',
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[col for col in gdf.columns if col != 'geometry'],
                aliases=[col.replace('_', ' ').title() for col in gdf.columns 
                       if col != 'geometry'],
                sticky=True
            )
        ).add_to(m)
        
        if 'centroid_coords' in gdf.columns:
            for _, row in gdf.iterrows():
                folium.CircleMarker(
                    location=row['centroid_coords'],
                    radius=3,
                    color='blue',
                    fill=True
                ).add_to(m)
                
    elif gdf.geom_type[0] == 'Point':
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in gdf.iterrows():
            # Safely access the 'name' column
            name = row.get('name', 'Unknown')
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=row['size']/2 if 'size' in gdf.columns else 5,
                color='#3186cc',
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{name}</b><br>Population: {row.get('population', 'N/A')}",
                    max_width=300
                )
            ).add_to(marker_cluster)
        
        HeatMap(
            data=[[point.y, point.x] for point in gdf.geometry],
            radius=15
        ).add_to(m)
        
    elif gdf.geom_type[0] == 'LineString':
        for _, row in gdf.iterrows():
            folium.PolyLine(
                locations=[[y, x] for x, y in row.geometry.coords],
                color='blue' if row['type'] == 'Highway' else 
                     'green' if row['type'] == 'Rail' else 
                     'red' if row['type'] == 'Subway' else 'gray',
                weight=3 if row['type'] == 'Highway' else 2,
                popup=f"{row['type']} (Length: {row.get('length_km', '?')} km)"
            ).add_to(m)
    
    folium.LayerControl().add_to(m)
    return m

def create_3d_visualization(gdf):
    """Complete 3D visualization with all original features"""
    if gdf.geom_type[0] not in ['Polygon', 'MultiPolygon']:
        return None
    
    # Add elevation and color columns
    gdf['elevation'] = np.random.randint(100, 1000, len(gdf))
    gdf['color'] = [[255, 140, 0, 200] for _ in range(len(gdf))]
    
    # Define the view state
    view_state = pdk.ViewState(
        latitude=40,
        longitude=-95,
        zoom=3,
        pitch=45,
        bearing=30
    )
    
    # Define the layers
    layers = [
        pdk.Layer(
            "PolygonLayer",
            data=gdf.to_json(),  # Use GeoJSON string directly
            get_polygon="geometry.coordinates",
            get_fill_color="color",
            get_elevation="elevation",
            elevation_scale=100,
            pickable=True,
            extruded=True,
            auto_highlight=True
        )
    ]
    
    # Define the tooltip
    tooltip = {
        "html": "<b>Elevation:</b> {elevation} meters<br><b>Area:</b> {area_km2:.2f} km²" 
                if 'area_km2' in gdf.columns else "<b>Elevation:</b> {elevation} meters",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white"
        }
    }
    
    # Return the Pydeck Deck object
    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='mapbox://styles/mapbox/light-v9'
    )

# --- Main Application ---
def main():
    # Load data
    data = load_data()
    
    # Sidebar controls
    with st.sidebar:
        st.title("🌍 Geospatial Analysis")
        dataset = st.selectbox("Dataset", list(data.keys()))
        gdf = data[dataset].copy()
        
        st.markdown(f"""
        ### Dataset Info
        - **Features:** {len(gdf)}
        - **Geometry Type:** `{gdf.geom_type[0]}`
        - **CRS:** `{gdf.crs}`
        """)
        
        new_crs = st.selectbox("Transform CRS", 
                             ["EPSG:4326 (WGS84)", 
                              "EPSG:3857 (Web Mercator)"])
        gdf = gdf.to_crs(new_crs.split()[0])
        
        analysis_type = st.radio("Analysis Type", 
                               ["Interactive Map", 
                                "Spatial Statistics",
                                "Network Analysis",
                                "3D Visualization"])
    
    # Main content
    st.title("Advanced Geospatial Analysis")
    
    if analysis_type == "Interactive Map":
        st.header("Interactive Mapping")
        m = create_interactive_map(gdf)
        folium_static(m, width=1000, height=600)
        
    elif analysis_type == "Spatial Statistics":
        st.header("Spatial Statistics")
        results = perform_spatial_analysis(gdf)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Metrics")
            for k, v in results.items():
                st.metric(k, v if not isinstance(v, float) else f"{v:.2f}")
        
        with col2:
            st.subheader("Visualization")
            if gdf.geom_type[0] in ['Polygon', 'MultiPolygon']:
                fig, ax = plt.subplots()
                gdf.plot(column='density' if 'density' in gdf.columns else 'area_km2', 
                        legend=True, ax=ax)
                st.pyplot(fig)
            elif gdf.geom_type[0] == 'Point':
                st.map(gdf)
    
    elif analysis_type == "Network Analysis":
        st.header("Network Analysis")
        if gdf.geom_type[0] == 'LineString':
            G = nx.Graph()
            for line in gdf.geometry:
                for i in range(len(line.coords)-1):
                    G.add_edge(line.coords[i], line.coords[i+1])
            
            col1, col2 = st.columns(2)
            col1.metric("Nodes", G.number_of_nodes())
            col2.metric("Edges", G.number_of_edges())
            
            fig, ax = plt.subplots()
            nx.draw(G, pos={n: (n[0], n[1]) for n in G.nodes()}, 
                   ax=ax, node_size=20, width=0.5)
            st.pyplot(fig)
        else:
            st.warning("Network analysis requires line data")
    
    elif analysis_type == "3D Visualization":
        st.header("3D Visualization")
        deck = create_3d_visualization(gdf)
        if deck:
            st.pydeck_chart(deck)
        else:
            st.warning("3D visualization requires polygon data")

if __name__ == "__main__":
    main()