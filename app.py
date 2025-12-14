import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from PIL import Image
import sqlite3
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Real Estate Market Analysis",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #464852;
    }
    .stMetric label {
        color: #fafafa !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #09ab3b !important;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 10px;
    }
    h2 {
        color: #2ca02c;
        padding-top: 20px;
    }
    .reportview-container .main .block-container {
        max-width: 1400px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load the real estate data from SQLite database"""
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('realtor.db')

        # Query to join all three tables and get complete property information
        query = """
        SELECT
            p.property_id,
            p.status,
            p.price,
            p.bed,
            p.bath,
            p.acre_lot,
            p.house_size,
            p.prev_sold_date,
            b.broker_id as brokered_by,
            l.street,
            l.city,
            l.state,
            l.zip_code
        FROM Property p
        INNER JOIN Location l ON p.location_id = l.location_id
        INNER JOIN Broker b ON p.broker_id = b.broker_id
        """

        # Read data from database
        df = pd.read_sql_query(query, conn)

        # Close connection
        conn.close()

        # Convert float columns to integers where appropriate
        int_columns = ['bed', 'bath', 'brokered_by']
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Handle zip_code (might be string in database)
        if 'zip_code' in df.columns:
            df['zip_code'] = pd.to_numeric(df['zip_code'], errors='coerce').fillna(0).astype(int)

        # Handle street (might be float)
        if 'street' in df.columns:
            df['street'] = pd.to_numeric(df['street'], errors='coerce').fillna(0).astype(int)

        # Calculate price per square foot
        df['price_per_sqft'] = df['price'] / df['house_size']

        return df
    except Exception as e:
        st.error(f"Error loading data from database: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_data()

# Check if data loaded successfully
if df.empty:
    st.error("❌ Failed to load data. Please check if realtor.db exists in the correct location.")
    st.stop()

# Sidebar filters
st.sidebar.header("🔍 Filter Properties")

# Price range filter
price_min, price_max = st.sidebar.slider(
    "Price Range ($)",
    min_value=int(df['price'].min()),
    max_value=int(df['price'].max()),
    value=(int(df['price'].min()), int(df['price'].max())),
    step=5000
)

# Bedroom filter
bed_options = sorted(df['bed'].unique())
selected_beds = st.sidebar.multiselect(
    "Bedrooms",
    options=bed_options,
    default=bed_options
)

# Bathroom filter
bath_options = sorted(df['bath'].unique())
selected_baths = st.sidebar.multiselect(
    "Bathrooms",
    options=bath_options,
    default=bath_options
)

# City filter
city_options = sorted(df['city'].unique())
selected_cities = st.sidebar.multiselect(
    "Cities",
    options=city_options,
    default=city_options
)

# House size filter
size_min, size_max = st.sidebar.slider(
    "House Size (sq ft)",
    min_value=int(df['house_size'].min()),
    max_value=int(df['house_size'].max()),
    value=(int(df['house_size'].min()), int(df['house_size'].max())),
    step=100
)

# Apply filters
filtered_df = df[
    (df['price'] >= price_min) &
    (df['price'] <= price_max) &
    (df['bed'].isin(selected_beds)) &
    (df['bath'].isin(selected_baths)) &
    (df['city'].isin(selected_cities)) &
    (df['house_size'] >= size_min) &
    (df['house_size'] <= size_max)
]

# App Title
st.title("🏠 Real Estate Market Analysis Dashboard")
st.markdown("### Transforming Real Estate Data into Actionable Insights")
st.markdown("---")

# Navigation Tabs
page = st.tabs([
    "🏠 Home Dashboard",
    "🗺️ Map View",
    "💰 Price Analysis",
    "🏘️ Property Characteristics",
    "💎 Investment Opportunities",
    "📈 Market Trends",
    "🧮 Affordability Tools",
    "📋 Property Browser"
])

# Home Dashboard Tab
with page[0]:
    # KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Properties", len(filtered_df))

    with col2:
        avg_price = filtered_df['price'].mean()
        st.metric("Avg Price", f"${avg_price:,.0f}")

    with col3:
        avg_price_sqft = filtered_df['price_per_sqft'].mean()
        st.metric("Avg $/sq ft", f"${avg_price_sqft:.2f}")

    with col4:
        most_active_broker = filtered_df['brokered_by'].mode().values[0] if len(filtered_df) > 0 else 'N/A'
        st.metric("Most Active Broker", most_active_broker)

    with col5:
        avg_size = filtered_df['house_size'].mean()
        st.metric("Avg House Size", f"{avg_size:,.0f} sq ft")

    st.markdown("---")

    # Quick insights
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Price Distribution")
        fig = px.histogram(filtered_df, x='price', nbins=60,
                          title="Property Price Distribution",
                          labels={'price': 'Price ($)', 'count': 'Number of Properties'},
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏙️ Properties by City")
        city_counts = filtered_df['city'].value_counts().head(10)
        fig = px.bar(x=city_counts.index, y=city_counts.values,
                    title="Top 10 Cities by Property Count",
                    labels={'x': 'City', 'y': 'Number of Properties'},
                    color=city_counts.values,
                    color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Market Overview
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛏️ Property Configuration")
        config_data = filtered_df.groupby(['bed', 'bath']).size().reset_index(name='count')
        fig = px.scatter(config_data, x='bed', y='bath', size='count',
                        title="Bedroom vs Bathroom Configurations",
                        labels={'bed': 'Bedrooms', 'bath': 'Bathrooms', 'count': 'Count'},
                        color='count', color_continuous_scale='Reds')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💵 Price Range Distribution")
        price_ranges = pd.cut(filtered_df['price'],
                             bins=[0, 75000, 150000, 250000, 500000, 1000000],
                             labels=['<$75K', '$75K-$150K', '$150K-$250K', '$250K-$500K', '>$500K'])
        range_counts = price_ranges.value_counts()
        fig = px.pie(values=range_counts.values, names=range_counts.index,
                    title="Properties by Price Range",
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Additional Altair Visualizations
    st.markdown("---")

    # 1) Price distribution histogram
    st.subheader("📊 Price Distribution (Detailed)")
    if not filtered_df.empty:
        hist = alt.Chart(filtered_df).mark_bar().encode(
            alt.X("price:Q", bin=alt.Bin(maxbins=60), title="Price (USD)"),
            y='count()'
        ).properties(width=800, height=300)
        st.altair_chart(hist, use_container_width=True)
    else:
        st.info("No data for selected filters.")

    # 2) Avg price by state (bar)
    st.subheader("🗺️ Average Price by State")
    if not filtered_df.empty:
        avg_state = filtered_df.groupby("state", dropna=True)['price'].mean().reset_index().sort_values("price", ascending=False).head(25)
        bar = alt.Chart(avg_state).mark_bar().encode(
            x=alt.X("price:Q", title="Average Price"),
            y=alt.Y("state:N", sort='-x', title="State")
        ).properties(width=700, height=400)
        st.altair_chart(bar, use_container_width=True)

    # 3) Scatter: Price vs House size
    st.subheader("📐 Price vs House Size (Scatter)")
    if 'house_size' in filtered_df.columns and not filtered_df.empty:
        scatter = alt.Chart(filtered_df).mark_circle(opacity=0.6).encode(
            x=alt.X("house_size:Q", title="House Size (sqft)"),
            y=alt.Y("price:Q", title="Price (USD)"),
            color=alt.Color("bed:O", title="Beds"),
            tooltip=["price", "house_size", "bed", "bath", "city", "state"]
        ).interactive().properties(width=900, height=450)
        st.altair_chart(scatter, use_container_width=True)

# Map View Tab
with page[1]:
    # Puerto Rico city coordinates (approximate centers)
    city_coords = {
        'Adjuntas': [18.1638, -66.7222],
        'Aguada': [18.3770, -67.1893],
        'Aguadilla': [18.4275, -67.1541],
        'Aguas Buenas': [18.2569, -66.1028],
        'Aibonito': [18.1397, -66.2661],
        'Anasco': [18.2808, -67.1397],
        'Arecibo': [18.4744, -66.7156],
        'Arroyo': [17.9653, -66.0619],
        'Barceloneta': [18.4508, -66.5397],
        'Barranquitas': [18.1869, -66.3061],
        'Bayamon': [18.3989, -66.1555],
        'Cabo Rojo': [18.0869, -67.1456],
        'Caguas': [18.2342, -66.0353],
        'Camuy': [18.4839, -66.8456],
        'Canovanas': [18.3783, -65.9019],
        'Carolina': [18.3808, -65.9567],
        'Catano': [18.4364, -66.1383],
        'Cayey': [18.1119, -66.1661],
        'Ceiba': [18.2636, -65.6433],
        'Ciales': [18.3367, -66.4686],
        'Cidra': [18.1756, -66.1614],
        'Coamo': [18.0797, -66.3564],
        'Comerio': [18.2189, -66.2264],
        'Corozal': [18.3428, -66.3197],
        'Culebra': [18.3089, -65.3036],
        'Dorado': [18.4589, -66.2678],
        'Fajardo': [18.3258, -65.6525],
        'Florida': [18.3628, -66.5628],
        'Guanica': [17.9719, -66.9081],
        'Guayama': [17.9842, -66.1136],
        'Guayanilla': [18.0169, -66.7911],
        'Guaynabo': [18.3675, -66.1142],
        'Gurabo': [18.2544, -65.9731],
        'Hatillo': [18.4864, -66.8256],
        'Hormigueros': [18.1439, -67.1189],
        'Humacao': [18.1497, -65.8272],
        'Isabela': [18.5000, -67.0303],
        'Jayuya': [18.2183, -66.5917],
        'Juana Diaz': [18.0525, -66.5039],
        'Juncos': [18.2275, -65.9211],
        'Lajas': [18.0497, -67.0589],
        'Lares': [18.2947, -66.8778],
        'Las Marias': [18.2514, -66.9889],
        'Las Piedras': [18.1833, -65.8653],
        'Loiza': [18.4328, -65.8797],
        'Luquillo': [18.3728, -65.7164],
        'Manati': [18.4308, -66.4828],
        'Maricao': [18.1808, -66.9797],
        'Maunabo': [18.0078, -65.8997],
        'Mayaguez': [18.2013, -67.1397],
        'Moca': [18.3947, -67.1117],
        'Morovis': [18.3256, -66.4067],
        'Naguabo': [18.2117, -65.7350],
        'Naranjito': [18.3006, -66.2447],
        'Orocovis': [18.2267, -66.3911],
        'Patillas': [18.0006, -66.0133],
        'Penuelas': [18.0569, -66.7231],
        'Ponce': [18.0111, -66.6141],
        'Quebradillas': [18.4739, -66.9386],
        'Rincon': [18.3403, -67.2494],
        'Rio Grande': [18.3803, -65.8314],
        'Sabana Grande': [18.0775, -66.9608],
        'Salinas': [17.9778, -66.2981],
        'San German': [18.0808, -67.0450],
        'San Juan': [18.4655, -66.1057],
        'San Lorenzo': [18.1883, -65.9614],
        'San Sebastian': [18.3386, -66.9900],
        'Santa Isabel': [17.9661, -66.4047],
        'Toa Alta': [18.3883, -66.2478],
        'Toa Baja': [18.4447, -66.2542],
        'Trujillo Alto': [18.3547, -66.0172],
        'Utuado': [18.2656, -66.7003],
        'Vega Alta': [18.4119, -66.3311],
        'Vega Baja': [18.4456, -66.3878],
        'Vieques': [18.1267, -65.4400],
        'Villalba': [18.1272, -66.4922],
        'Yabucoa': [18.0506, -65.8794],
        'Yauco': [18.0350, -66.8497]
    }

    # Add coordinates to filtered data
    filtered_df['latitude'] = filtered_df['city'].map(lambda x: city_coords.get(x, [18.2208, -66.5901])[0])
    filtered_df['longitude'] = filtered_df['city'].map(lambda x: city_coords.get(x, [18.2208, -66.5901])[1])

    # Map options
    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("### Map Settings")
        map_color_by = st.selectbox(
            "Color points by:",
            ["price", "price_per_sqft", "bed", "bath", "house_size"],
            index=0
        )

        map_size_by = st.selectbox(
            "Size points by:",
            ["price", "house_size", "bed", "bath", "acre_lot"],
            index=0
        )

        show_city_labels = st.checkbox("Show city names", value=True)

    with col1:
        st.markdown("### Property Distribution Map")

        # Create scatter mapbox
        fig = px.scatter_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color=map_color_by,
            size=map_size_by,
            hover_name="city",
            hover_data={
                "price": ":$,.0f",
                "bed": True,
                "bath": True,
                "house_size": ":.0f",
                "price_per_sqft": ":$.2f",
                "latitude": False,
                "longitude": False
            },
            color_continuous_scale="Viridis",
            size_max=20,
            zoom=8,
            center={"lat": 18.2208, "lon": -66.5901},
            mapbox_style="open-street-map",
            title=f"Properties colored by {map_color_by.replace('_', ' ').title()}"
        )

        fig.update_layout(
            height=600,
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )

        st.plotly_chart(fig, use_container_width=True)

    # City-level aggregations
    st.markdown("---")
    st.subheader("📊 City-Level Statistics")

    col1, col2 = st.columns(2)

    with col1:
        # Aggregate by city
        city_stats = filtered_df.groupby('city').agg({
            'price': 'mean',
            'house_size': 'mean',
            'brokered_by': 'count'
        }).reset_index()
        city_stats.columns = ['city', 'avg_price', 'avg_size', 'count']
        city_stats['latitude'] = city_stats['city'].map(lambda x: city_coords.get(x, [18.2208, -66.5901])[0])
        city_stats['longitude'] = city_stats['city'].map(lambda x: city_coords.get(x, [18.2208, -66.5901])[1])

        # Bubble map by city
        fig = px.scatter_mapbox(
            city_stats,
            lat="latitude",
            lon="longitude",
            size="count",
            color="avg_price",
            hover_name="city",
            hover_data={
                "avg_price": ":$,.0f",
                "avg_size": ":.0f",
                "count": True,
                "latitude": False,
                "longitude": False
            },
            color_continuous_scale="RdYlGn_r",
            size_max=50,
            zoom=8,
            center={"lat": 18.2208, "lon": -66.5901},
            mapbox_style="open-street-map",
            title="Property Inventory by City (Bubble size = # of properties)"
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Top cities table
        st.markdown("### 🏆 Top Cities by Property Count")
        top_cities = city_stats.nlargest(10, 'count')[['city', 'count', 'avg_price', 'avg_size']]
        st.dataframe(
            top_cities.style.format({
                'avg_price': '${:,.0f}',
                'avg_size': '{:.0f}'
            }),
            use_container_width=True,
            height=400
        )

        st.markdown("### 💰 Most Expensive Cities")
        expensive_cities = city_stats.nlargest(10, 'avg_price')[['city', 'avg_price', 'count']]
        st.dataframe(
            expensive_cities.style.format({
                'avg_price': '${:,.0f}'
            }),
            use_container_width=True,
            height=400
        )

# Price Analysis Tab
with page[2]:
    st.subheader("💰 Price Analysis")

    tab1, tab2, tab3 = st.tabs(["Geographic Analysis", "Price Metrics", "Comparative Analysis"])

    with tab1:
        st.subheader("🗺️ Geographic Price Distribution")

        col1, col2 = st.columns(2)

        with col1:
            # City-wise price comparison
            city_price = filtered_df.groupby('city')['price'].agg(['mean', 'median', 'count']).reset_index()
            city_price = city_price.sort_values('mean', ascending=False)

            fig = px.bar(city_price, x='city', y='mean',
                        title="Average Price by City",
                        labels={'mean': 'Average Price ($)', 'city': 'City'},
                        color='mean', color_continuous_scale='Blues',
                        hover_data=['median', 'count'])
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Box plot for price distribution by city
            fig = px.box(filtered_df, x='city', y='price',
                        title="Price Distribution by City (Box Plot)",
                        labels={'price': 'Price ($)', 'city': 'City'},
                        color='city')
            fig.update_xaxes(tickangle=-45)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Zip code heatmap
        st.subheader("📮 Price by Zip Code")
        zip_price = filtered_df.groupby('zip_code')['price'].mean().reset_index()
        fig = px.bar(zip_price, x='zip_code', y='price',
                    title="Average Price by Zip Code",
                    labels={'price': 'Average Price ($)', 'zip_code': 'Zip Code'},
                    color='price', color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📏 Price per Square Foot Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Cities ranked by $/sq ft
            city_sqft = filtered_df.groupby('city')['price_per_sqft'].mean().reset_index()
            city_sqft = city_sqft.sort_values('price_per_sqft', ascending=False)

            fig = px.bar(city_sqft, x='city', y='price_per_sqft',
                        title="Cities Ranked by Avg Price per Sq Ft",
                        labels={'price_per_sqft': '$ per Sq Ft', 'city': 'City'},
                        color='price_per_sqft', color_continuous_scale='Viridis')
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Price vs House Size scatter
            fig = px.scatter(filtered_df, x='house_size', y='price', color='city',
                           title="Price vs House Size by City",
                           labels={'house_size': 'House Size (sq ft)', 'price': 'Price ($)'},
                           hover_data=['bed', 'bath', 'price_per_sqft'])
            st.plotly_chart(fig, use_container_width=True)

        # Price per sqft distribution
        st.subheader("💲 Price per Sq Ft Distribution")
        fig = px.histogram(filtered_df, x='price_per_sqft', nbins=30,
                          title="Distribution of Price per Square Foot",
                          labels={'price_per_sqft': '$ per Sq Ft', 'count': 'Number of Properties'},
                          color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📊 Statistical Summary by City")

        # Comprehensive city statistics
        city_stats = filtered_df.groupby('city').agg({
            'price': ['mean', 'median', 'min', 'max', 'std'],
            'price_per_sqft': 'mean',
            'house_size': 'mean',
            'bed': 'mean',
            'bath': 'mean',
            'brokered_by': 'count'
        }).round(2)

        city_stats.columns = ['Avg Price', 'Median Price', 'Min Price', 'Max Price',
                              'Std Dev', 'Avg $/sqft', 'Avg Size', 'Avg Beds',
                              'Avg Baths', 'Count']

        st.dataframe(city_stats.style.format({
            'Avg Price': '${:,.0f}',
            'Median Price': '${:,.0f}',
            'Min Price': '${:,.0f}',
            'Max Price': '${:,.0f}',
            'Std Dev': '${:,.0f}',
            'Avg $/sqft': '${:.2f}',
            'Avg Size': '{:.0f}',
            'Avg Beds': '{:.1f}',
            'Avg Baths': '{:.1f}'
        }), use_container_width=True)

# Property Characteristics Tab
with page[3]:
    st.subheader("🏘️ Property Characteristics Analysis")

    tab1, tab2 = st.tabs(["Bed/Bath Analysis", "Size & Lot Analysis"])

    with tab1:
        st.subheader("🛏️ Bedroom and Bathroom Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Average price by bed/bath combination
            bed_bath_price = filtered_df.groupby(['bed', 'bath'])['price'].mean().reset_index()
            pivot_data = bed_bath_price.pivot(index='bed', columns='bath', values='price')

            fig = px.imshow(pivot_data,
                          title="Average Price Heatmap (Bed vs Bath)",
                          labels=dict(x="Bathrooms", y="Bedrooms", color="Avg Price ($)"),
                          color_continuous_scale='RdYlGn',
                          text_auto='.0f')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Property count by configuration
            config_count = filtered_df.groupby(['bed', 'bath']).size().reset_index(name='count')
            config_count['config'] = config_count['bed'].astype(str) + 'BR/' + config_count['bath'].astype(str) + 'BA'
            config_count = config_count.sort_values('count', ascending=False).head(10)

            fig = px.bar(config_count, x='config', y='count',
                        title="Top 10 Property Configurations",
                        labels={'config': 'Configuration', 'count': 'Number of Properties'},
                        color='count', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

        # House size by bedroom count
        st.subheader("📐 House Size Distribution by Bedrooms")
        fig = px.box(filtered_df, x='bed', y='house_size', color='bed',
                    title="House Size Distribution by Number of Bedrooms",
                    labels={'bed': 'Bedrooms', 'house_size': 'House Size (sq ft)'})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🏞️ Lot Size Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Acre lot vs price
            fig = px.scatter(filtered_df, x='acre_lot', y='price', color='city',
                           size='house_size',
                           title="Price vs Lot Size by City",
                           labels={'acre_lot': 'Lot Size (acres)', 'price': 'Price ($)'},
                           hover_data=['bed', 'bath', 'house_size'])
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Lot size distribution
            fig = px.histogram(filtered_df, x='acre_lot', nbins=20,
                             title="Lot Size Distribution",
                             labels={'acre_lot': 'Lot Size (acres)', 'count': 'Number of Properties'},
                             color_discrete_sequence=['#ff7f0e'])
            st.plotly_chart(fig, use_container_width=True)

        # Average metrics by lot size category
        st.subheader("📊 Metrics by Lot Size Category")
        filtered_df['lot_category'] = pd.cut(filtered_df['acre_lot'],
                                             bins=[0, 0.1, 0.25, 0.5, 1, 100],
                                             labels=['<0.1 ac', '0.1-0.25 ac', '0.25-0.5 ac',
                                                    '0.5-1 ac', '>1 ac'])

        lot_metrics = filtered_df.groupby('lot_category').agg({
            'price': 'mean',
            'house_size': 'mean',
            'price_per_sqft': 'mean',
            'brokered_by': 'count'
        }).reset_index()
        lot_metrics.columns = ['Lot Category', 'Avg Price', 'Avg Size', 'Avg $/sqft', 'Count']

        st.dataframe(lot_metrics.style.format({
            'Avg Price': '${:,.0f}',
            'Avg Size': '{:.0f}',
            'Avg $/sqft': '${:.2f}'
        }), use_container_width=True)

# Investment Opportunities Tab
with page[4]:
    st.subheader("💎 Investment Opportunities")

    tab1, tab2 = st.tabs(["Value Finder", "Rental Yield Estimator"])

    with tab1:
        st.subheader("🎯 Value Identification")

        # Calculate value score (properties below regression line are good value)
        from sklearn.linear_model import LinearRegression
        X = filtered_df[['house_size']].values
        y = filtered_df['price'].values
        model = LinearRegression()
        model.fit(X, y)
        filtered_df['predicted_price'] = model.predict(X)
        filtered_df['value_score'] = ((filtered_df['predicted_price'] - filtered_df['price']) /
                                       filtered_df['predicted_price'] * 100)

        col1, col2 = st.columns(2)

        with col1:
            # Scatter plot showing value analysis
            fig = px.scatter(filtered_df, x='house_size', y='price', color='value_score',
                           title="Price Efficiency Analysis",
                           labels={'house_size': 'House Size (sq ft)', 'price': 'Price ($)', 'value_score': 'Value Score (%)'},
                           hover_data=['bed', 'bath', 'city'],
                           color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

            st.info("💡 Properties with higher value scores (greener) represent better value opportunities!")

        with col2:
            # Best value properties
            st.subheader("🏆 Top Value Properties")
            best_value = filtered_df.nlargest(10, 'value_score')[
                ['city', 'price', 'bed', 'bath', 'house_size', 'price_per_sqft', 'value_score']
            ]
            best_value['value_score'] = best_value['value_score'].round(1)

            st.dataframe(best_value.style.format({
                'price': '${:,.0f}',
                'house_size': '{:.0f}',
                'price_per_sqft': '${:.2f}',
                'value_score': '{:.1f}%'
            }),
            use_container_width=True)

        # Best value by city
        st.subheader("🌟 Best Value by City (Lowest $/sq ft)")
        city_best_value = filtered_df.loc[filtered_df.groupby('city')['price_per_sqft'].idxmin()][
            ['city', 'price', 'bed', 'bath', 'house_size', 'price_per_sqft']
        ]

        st.dataframe(city_best_value.style.format({
            'price': '${:,.0f}',
            'house_size': '{:.0f}',
            'price_per_sqft': '${:.2f}'
        }), use_container_width=True)

    with tab2:
        st.subheader("📊 Rental Yield Estimator")

        st.info("💡 This calculator estimates potential rental yields based on typical market rates.")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Input Assumptions")
            rental_rate_per_sqft = st.number_input(
                "Monthly Rent per Sq Ft ($)",
                min_value=0.5,
                max_value=5.0,
                value=1.2,
                step=0.1
            )

            annual_expenses_pct = st.slider(
                "Annual Expenses (% of rent)",
                min_value=0,
                max_value=50,
                value=25,
                step=5
            )

            vacancy_rate = st.slider(
                "Vacancy Rate (%)",
                min_value=0,
                max_value=20,
                value=5,
                step=1
            )

        with col2:
            # Calculate yields for all properties
            filtered_df['estimated_monthly_rent'] = filtered_df['house_size'] * rental_rate_per_sqft
            filtered_df['annual_rental_income'] = filtered_df['estimated_monthly_rent'] * 12
            filtered_df['annual_expenses'] = filtered_df['annual_rental_income'] * (annual_expenses_pct / 100)
            filtered_df['vacancy_loss'] = filtered_df['annual_rental_income'] * (vacancy_rate / 100)
            filtered_df['net_operating_income'] = (filtered_df['annual_rental_income'] -
                                                   filtered_df['annual_expenses'] -
                                                   filtered_df['vacancy_loss'])
            filtered_df['cap_rate'] = (filtered_df['net_operating_income'] / filtered_df['price']) * 100

            # Show top opportunities
            st.markdown("### 🎯 Top Rental Yield Opportunities")
            top_yields = filtered_df.nlargest(10, 'cap_rate')[
                ['city', 'price', 'house_size', 'estimated_monthly_rent',
                 'annual_rental_income', 'net_operating_income', 'cap_rate']
            ]

            st.dataframe(top_yields.style.format({
                'price': '${:,.0f}',
                'house_size': '{:.0f}',
                'estimated_monthly_rent': '${:,.0f}',
                'annual_rental_income': '${:,.0f}',
                'net_operating_income': '${:,.0f}',
                'cap_rate': '{:.2f}%'
            }),
            use_container_width=True)

        # Visualization
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(filtered_df.groupby('city')['cap_rate'].mean().reset_index().sort_values('cap_rate', ascending=False),
                        x='city', y='cap_rate',
                        title="Average Cap Rate by City",
                        labels={'cap_rate': 'Cap Rate (%)', 'city': 'City'},
                        color='cap_rate', color_continuous_scale='RdYlGn')
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(filtered_df, x='price', y='cap_rate', color='city',
                           size='house_size',
                           title="Cap Rate vs Price by City",
                           labels={'price': 'Price ($)', 'cap_rate': 'Cap Rate (%)'},
                           hover_data=['bed', 'bath'])
            st.plotly_chart(fig, use_container_width=True)

# Market Trends Tab
with page[5]:
    st.subheader("📈 Market Trends & Comparisons")

    tab1, tab2 = st.tabs(["City Comparisons", "Inventory Analysis"])

    with tab1:
        st.subheader("🏙️ Multi-Metric City Comparison")

        # Radar chart for city comparison
        city_metrics = filtered_df.groupby('city').agg({
            'price': 'mean',
            'house_size': 'mean',
            'price_per_sqft': 'mean',
            'bed': 'mean',
            'bath': 'mean',
            'brokered_by': 'count'
        }).reset_index()

        # Normalize metrics for radar chart
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        metrics_to_scale = ['price', 'house_size', 'price_per_sqft', 'bed', 'bath', 'brokered_by']
        city_metrics[metrics_to_scale] = scaler.fit_transform(city_metrics[metrics_to_scale])

        # Select cities for comparison
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Select Cities to Compare")
            cities_to_compare = st.multiselect(
                "Choose cities (up to 5):",
                options=sorted(filtered_df['city'].unique()),
                default=sorted(filtered_df['city'].unique())[:3],
                max_selections=5
            )

        with col2:
            if cities_to_compare:
                fig = go.Figure()

                for city in cities_to_compare:
                    city_data = city_metrics[city_metrics['city'] == city].iloc[0]
                    fig.add_trace(go.Scatterpolar(
                        r=[city_data['price'], city_data['house_size'],
                           city_data['price_per_sqft'], city_data['bed'],
                           city_data['bath'], city_data['brokered_by']],
                        theta=['Avg Price', 'Avg Size', '$/sqft', 'Beds', 'Baths', 'Inventory'],
                        fill='toself',
                        name=city
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="City Comparison Radar Chart (Normalized Metrics)"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Detailed comparison table
        st.subheader("📊 Detailed City Metrics")
        comparison_data = filtered_df.groupby('city').agg({
            'price': ['mean', 'median'],
            'house_size': 'mean',
            'price_per_sqft': 'mean',
            'bed': 'mean',
            'bath': 'mean',
            'brokered_by': 'count'
        }).round(2)
        comparison_data.columns = ['Avg Price', 'Median Price', 'Avg Size',
                                   'Avg $/sqft', 'Avg Beds', 'Avg Baths', 'Properties']

        st.dataframe(comparison_data.style.format({
            'Avg Price': '${:,.0f}',
            'Median Price': '${:,.0f}',
            'Avg Size': '{:.0f}',
            'Avg $/sqft': '${:.2f}',
            'Avg Beds': '{:.1f}',
            'Avg Baths': '{:.1f}'
        }), use_container_width=True)

    with tab2:
        st.subheader("📦 Inventory Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Properties by broker
            broker_counts = filtered_df['brokered_by'].value_counts().head(10)
            fig = px.bar(x=broker_counts.index.astype(str), y=broker_counts.values,
                        title="Top 10 Brokers by Listing Count",
                        labels={'x': 'Broker ID', 'y': 'Number of Listings'},
                        color=broker_counts.values, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Average price per broker
            broker_avg_price = filtered_df.groupby('brokered_by')['price'].mean().sort_values(ascending=False).head(10)
            fig = px.bar(x=broker_avg_price.index.astype(str), y=broker_avg_price.values,
                        title="Top 10 Brokers by Average Listing Price",
                        labels={'x': 'Broker ID', 'y': 'Average Price ($)'},
                        color=broker_avg_price.values, color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

        # Supply concentration
        col1, col2 = st.columns(2)

        with col1:
            city_dist = filtered_df['city'].value_counts()
            fig = px.pie(values=city_dist.values, names=city_dist.index,
                        title="Property Distribution by City",
                        hole=0.3)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Property types by city (using bed count as proxy)
            bed_by_city = filtered_df.groupby(['city', 'bed']).size().reset_index(name='count')
            fig = px.bar(bed_by_city, x='city', y='count', color='bed',
                        title="Property Types (Bedrooms) by City",
                        labels={'count': 'Number of Properties', 'bed': 'Bedrooms'},
                        barmode='stack')
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

# Affordability Tools Tab
with page[6]:
    st.subheader("🧮 Affordability Tools")

    tab1, tab2 = st.tabs(["Mortgage Calculator", "Affordability Matrix"])

    with tab1:
        st.subheader("🏦 Mortgage Calculator")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### Property Selection")

            # Select property or enter custom price
            property_selector = st.radio(
                "Choose option:",
                ["Select from listings", "Enter custom price"]
            )

            if property_selector == "Select from listings":
                property_options = filtered_df[['city', 'price', 'bed', 'bath', 'house_size']].copy()
                property_options['label'] = (property_options['city'] + ' - $' +
                                            property_options['price'].apply(lambda x: f'{x:,.0f}') +
                                            ' (' + property_options['bed'].astype(str) + 'BR/' +
                                            property_options['bath'].astype(str) + 'BA)')

                selected_property = st.selectbox(
                    "Select a property:",
                    options=range(len(property_options)),
                    format_func=lambda x: property_options.iloc[x]['label']
                )
                property_price = property_options.iloc[selected_property]['price']
            else:
                property_price = st.number_input(
                    "Property Price ($)",
                    min_value=10000,
                    max_value=10000000,
                    value=150000,
                    step=10000
                )

            st.markdown("### Loan Parameters")
            down_payment_pct = st.slider(
                "Down Payment (%)",
                min_value=0,
                max_value=50,
                value=20,
                step=5
            )

            interest_rate = st.slider(
                "Interest Rate (%)",
                min_value=3.0,
                max_value=8.0,
                value=6.5,
                step=0.25
            )

            loan_term_years = st.selectbox(
                "Loan Term (years)",
                options=[15, 20, 30],
                index=2
            )

        with col2:
            st.markdown("### Calculation Results")

            # Calculate mortgage
            down_payment = property_price * (down_payment_pct / 100)
            loan_amount = property_price - down_payment
            monthly_rate = interest_rate / 100 / 12
            num_payments = loan_term_years * 12

            if monthly_rate > 0:
                monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                                ((1 + monthly_rate)**num_payments - 1)
            else:
                monthly_payment = loan_amount / num_payments

            total_payment = monthly_payment * num_payments
            total_interest = total_payment - loan_amount

            # Display results
            st.metric("Property Price", f"${property_price:,.0f}")
            st.metric("Down Payment", f"${down_payment:,.0f}")
            st.metric("Loan Amount", f"${loan_amount:,.0f}")

            st.markdown("---")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Monthly Payment", f"${monthly_payment:,.2f}",
                         delta=None, delta_color="off")
            with col_b:
                st.metric("Total Interest", f"${total_interest:,.0f}",
                         delta=None, delta_color="off")

            st.metric("Total Amount Paid", f"${total_payment:,.0f}")

            # Amortization visualization
            st.markdown("### 📊 Payment Breakdown")

            # Calculate year-by-year breakdown
            years = []
            principal_paid = []
            interest_paid = []
            remaining_balance = loan_amount

            for year in range(1, loan_term_years + 1):
                year_interest = 0
                year_principal = 0

                for month in range(12):
                    interest_payment = remaining_balance * monthly_rate
                    principal_payment = monthly_payment - interest_payment

                    year_interest += interest_payment
                    year_principal += principal_payment
                    remaining_balance -= principal_payment

                years.append(year)
                principal_paid.append(year_principal)
                interest_paid.append(year_interest)

            # Stacked bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Principal', x=years, y=principal_paid,
                                marker_color='lightblue'))
            fig.add_trace(go.Bar(name='Interest', x=years, y=interest_paid,
                                marker_color='coral'))

            fig.update_layout(
                barmode='stack',
                title='Annual Payment Breakdown (Principal vs Interest)',
                xaxis_title='Year',
                yaxis_title='Amount ($)',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("💰 Affordability Matrix")

        st.info("💡 This matrix shows the maximum affordable property price based on income and interest rates.")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Assumptions")

            debt_to_income = st.slider(
                "Max Debt-to-Income Ratio (%)",
                min_value=25,
                max_value=45,
                value=35,
                step=5
            )

            down_payment_pct_matrix = st.slider(
                "Down Payment (%)",
                min_value=5,
                max_value=30,
                value=20,
                step=5,
                key='matrix_dp'
            )

            loan_term_matrix = st.selectbox(
                "Loan Term (years)",
                options=[15, 20, 30],
                index=2,
                key='matrix_term'
            )

        with col2:
            # Create affordability matrix
            income_levels = [40000, 60000, 80000, 100000, 120000, 150000, 200000]
            interest_rates = [4.0, 5.0, 6.0, 7.0, 8.0]

            affordability_data = []

            for income in income_levels:
                row = []
                max_monthly_payment = (income / 12) * (debt_to_income / 100)

                for rate in interest_rates:
                    monthly_rate = rate / 100 / 12
                    num_payments = loan_term_matrix * 12

                    if monthly_rate > 0:
                        loan_amount = max_monthly_payment * ((1 + monthly_rate)**num_payments - 1) / \
                                     (monthly_rate * (1 + monthly_rate)**num_payments)
                    else:
                        loan_amount = max_monthly_payment * num_payments

                    max_price = loan_amount / (1 - down_payment_pct_matrix / 100)
                    row.append(max_price)

                affordability_data.append(row)

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=affordability_data,
                x=[f'{r}%' for r in interest_rates],
                y=[f'${i:,.0f}' for i in income_levels],
                colorscale='RdYlGn',
                text=[[f'${val:,.0f}' for val in row] for row in affordability_data],
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Max Price ($)")
            ))

            fig.update_layout(
                title=f'Maximum Affordable Home Price<br>(DTI: {debt_to_income}%, Down: {down_payment_pct_matrix}%, Term: {loan_term_matrix}yr)',
                xaxis_title='Interest Rate',
                yaxis_title='Annual Income',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        # Additional insights
        st.markdown("### 💡 Affordability Insights")
        col1, col2, col3 = st.columns(3)

        with col1:
            income_needed_median = filtered_df['price'].median() * (down_payment_pct_matrix / 100)
            monthly_payment_needed = (filtered_df['price'].median() - income_needed_median) * \
                                     (6.5 / 100 / 12) * (1 + 6.5 / 100 / 12)**(30*12) / \
                                     ((1 + 6.5 / 100 / 12)**(30*12) - 1)
            min_income = (monthly_payment_needed * 12) / (debt_to_income / 100)
            st.metric("Min Income for Median Home", f"${min_income:,.0f}/year")

        with col2:
            affordable_count = len(filtered_df[filtered_df['price'] <= affordability_data[2][2]])  # 80k income, 6% rate
            st.metric("Affordable Properties (80K income, 6%)", affordable_count)

        with col3:
            avg_monthly = monthly_payment
            st.metric("Avg Monthly Payment", f"${avg_monthly:,.0f}")

# Property Browser Tab
with page[7]:
    st.subheader("📋 Property Browser & Data Export")

    st.subheader("🔍 Searchable Property Listings")

    # Display options
    col1, col2, col3 = st.columns(3)

    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            options=['price', 'price_per_sqft', 'house_size', 'bed', 'bath', 'city'],
            index=0
        )

    with col2:
        sort_order = st.radio(
            "Order:",
            options=['Ascending', 'Descending'],
            index=1
        )

    with col3:
        rows_to_show = st.selectbox(
            "Rows to display:",
            options=[10, 25, 50, 100, 'All'],
            index=0
        )

    # Sort data
    ascending = sort_order == 'Ascending'
    sorted_df = filtered_df.sort_values(by=sort_by, ascending=ascending)

    # Display data
    if rows_to_show != 'All':
        display_df = sorted_df.head(rows_to_show)
    else:
        display_df = sorted_df

    # Format for display
    display_columns = ['brokered_by', 'city', 'price', 'bed', 'bath',
                      'house_size', 'acre_lot', 'price_per_sqft', 'zip_code']

    st.dataframe(
        display_df[display_columns].style.format({
            'price': '${:,.0f}',
            'house_size': '{:.0f}',
            'acre_lot': '{:.2f}',
            'price_per_sqft': '${:.2f}'
        }),
        use_container_width=True,
        height=400
    )

    # Property comparison tool
    st.markdown("---")
    st.subheader("⚖️ Property Comparison Tool")

    st.info("Select properties from the filtered results to compare side-by-side.")

    # Create property labels for selection
    property_labels = [f"{row['city']} - ${row['price']:,.0f} ({row['bed']}BR/{row['bath']}BA, {row['house_size']:.0f} sqft)"
                      for _, row in sorted_df.iterrows()]

    selected_indices = st.multiselect(
        "Select up to 5 properties to compare:",
        options=range(len(sorted_df)),
        format_func=lambda x: property_labels[x],
        max_selections=5
    )

    if selected_indices:
        comparison_df = sorted_df.iloc[selected_indices][
            ['city', 'price', 'bed', 'bath', 'house_size', 'acre_lot',
             'price_per_sqft', 'brokered_by', 'zip_code']
        ].T

        comparison_df.columns = [f'Property {i+1}' for i in range(len(selected_indices))]

        st.dataframe(
            comparison_df.style.format({
                'price': '${:,.0f}',
                'house_size': '{:.0f}',
                'acre_lot': '{:.2f}',
                'price_per_sqft': '${:.2f}'
            }),
            use_container_width=True
        )

        # Comparison visualization
        if len(selected_indices) > 1:
            col1, col2 = st.columns(2)

            with col1:
                comparison_data = sorted_df.iloc[selected_indices]
                fig = go.Figure()

                for metric in ['price', 'house_size', 'price_per_sqft']:
                    fig.add_trace(go.Bar(
                        name=metric.replace('_', ' ').title(),
                        x=[f'Property {i+1}' for i in range(len(selected_indices))],
                        y=comparison_data[metric]
                    ))

                fig.update_layout(
                    title="Property Metrics Comparison",
                    barmode='group',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Normalized comparison
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()

                metrics_to_compare = ['price', 'house_size', 'bed', 'bath', 'acre_lot']
                normalized_data = comparison_data[metrics_to_compare].copy()
                normalized_data = pd.DataFrame(
                    scaler.fit_transform(normalized_data),
                    columns=metrics_to_compare,
                    index=normalized_data.index
                )

                fig = go.Figure()

                for i, idx in enumerate(selected_indices):
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_data.iloc[i].values,
                        theta=metrics_to_compare,
                        fill='toself',
                        name=f'Property {i+1}'
                    ))

                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Normalized Comparison",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

    # Export options
    st.markdown("---")
    st.subheader("📥 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Export Filtered Data")
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="real_estate_filtered_data.csv",
            mime="text/csv"
        )

    with col2:
        st.markdown("### Export Summary Statistics")
        summary_stats = filtered_df.describe()
        summary_csv = summary_stats.to_csv()
        st.download_button(
            label="📥 Download Summary Stats",
            data=summary_csv,
            file_name="real_estate_summary_stats.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Real Estate Market Analysis Tool | Data-Driven Property Insights</p>
        <p style='font-size: 0.8em;'>Built with Streamlit & Plotly | Database Version | Last Updated: 2025</p>
    </div>
    """, unsafe_allow_html=True)
