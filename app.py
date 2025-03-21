import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime, date, timedelta  # Fixed import for date
import re
import base64
import io
import json
from urllib.parse import unquote, quote
from dateutil.relativedelta import relativedelta

# Set page configuration with Glimpse-inspired styling
st.set_page_config(
    page_title="Wiki Trends - Discover Topic Interest Before It's Trending",
    page_icon="📈",
    layout="wide"
)

# Custom CSS to match Glimpse style
st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary-color: #6247aa;
        --secondary-color: #102b3f;
        --accent-color: #50b8e7;
        --text-color: #333;
        --background-color: #ffffff;
        --card-background: #f9f9f9;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: var(--secondary-color);
        font-weight: 700 !important;
    }
    h1 {
        font-size: 2.5rem !important;
    }
    .main-header {
        background: linear-gradient(90deg, #6247aa, #102b3f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .subheader {
        color: #666;
        font-size: 1.3rem !important;
        font-weight: 400 !important;
        margin-top: 0;
    }
    
    /* Cards */
    .card {
        background-color: var(--card-background);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #f5f7f9;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: var(--secondary-color) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Streamlit components styling */
    .stTextInput, .stDateInput {
        margin-bottom: 15px;
    }
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 5px;
    }
    
    /* Stats boxes */
    .stat-box {
        background-color: var(--card-background);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Footer */
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #eee;
        color: #666;
        font-size: 0.8rem;
    }
    
    /* Custom toggle switch */
    .toggle-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    .toggle-label {
        margin-right: 10px;
        font-weight: 500;
    }
    
    /* Search results */
    .search-result {
        padding: 10px;
        border-radius: 5px;
        transition: all 0.2s ease;
    }
    .search-result:hover {
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("<h1 class='main-header'>Wiki Trends</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Discover interest patterns in Wikipedia topics before they're trending</p>", unsafe_allow_html=True)

# Helper functions
def extract_page_title(url):
    """Extract the page title from a Wikipedia URL"""
    match = re.search(r'wikipedia\.org/wiki/([^#?]+)', url)
    if match:
        return unquote(match.group(1))
    else:
        return None

def search_wikipedia_topics(query, limit=5):
    """Search for Wikipedia topics using the Wikipedia API"""
    if not query or len(query.strip()) < 2:
        return []
        
    url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote(query)}&limit={limit}&namespace=0&format=json"
    
    try:
        response = requests.get(url, headers={'User-Agent': 'WikiInterestComparisonTool/1.0'})
        
        if response.status_code == 200:
            data = response.json()
            # The API returns data in the format [query, [titles], [descriptions], [urls]]
            if len(data) >= 2:
                return data[1]  # Return the list of titles
        
        return []
    except Exception as e:
        st.error(f"Error searching Wikipedia: {e}")
        return []

def fetch_pageviews(page_title, start_date, end_date, project='en.wikipedia.org'):
    """Fetch page view data from the Wikimedia API"""
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/all-agents/{page_title}/daily/{start_str}/{end_str}"
    
    response = requests.get(url, headers={'User-Agent': 'WikiInterestComparisonTool/1.0'})
    
    if response.status_code == 200:
        data = response.json()
        views_data = [(datetime.strptime(item['timestamp'], '%Y%m%d00'), item['views']) 
                      for item in data.get('items', [])]
        df = pd.DataFrame(views_data, columns=['date', 'views'])
        return df
    else:
        st.error(f"Error fetching data: {response.status_code}")
        st.text(response.text)
        return pd.DataFrame(columns=['date', 'views'])

def fetch_related_topics(page_title, limit=5):
    """Fetch related Wikipedia topics using the Wikipedia API"""
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=backlinks&bltitle={page_title}&bllimit={limit}&format=json"
    response = requests.get(url, headers={'User-Agent': 'WikiInterestComparisonTool/1.0'})
    
    related_topics = []
    if response.status_code == 200:
        data = response.json()
        backlinks = data.get('query', {}).get('backlinks', [])
        related_topics = [item['title'] for item in backlinks if 'redirect' not in item]
    
    # If not enough backlinks, get additional recommendations from the "related pages" feature
    if len(related_topics) < limit:
        url = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=page_image_free|wikibase_item&generator=categorymembers&gcmtitle=Category:{page_title.replace('_', ' ')}&gcmlimit={limit*2}&format=json"
        response = requests.get(url, headers={'User-Agent': 'WikiInterestComparisonTool/1.0'})
        if response.status_code == 200:
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                if page_data.get('title') != page_title.replace('_', ' '):
                    related_topics.append(page_data.get('title'))
                if len(related_topics) >= limit:
                    break
    
    return related_topics[:limit]

def get_table_download_link(df, filename, text, file_format='csv'):
    """Generate a download link for the dataframe in various formats"""
    if file_format == 'csv':
        data = df.to_csv(index=False)
        b64 = base64.b64encode(data.encode()).decode()
        mime_type = 'text/csv'
        file_ext = 'csv'
    elif file_format == 'excel':
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.save()
        data = output.getvalue()
        b64 = base64.b64encode(data).decode()
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        file_ext = 'xlsx'
    elif file_format == 'json':
        data = df.to_json(orient='records')
        b64 = base64.b64encode(data.encode()).decode()
        mime_type = 'application/json'
        file_ext = 'json'
        
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}.{file_ext}">{text}</a>'
    return href

def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    """Generate a download link for the matplotlib figure"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Create sidebar for inputs
with st.sidebar:
    st.header("Analysis Parameters")
    
    # Input for Number of Topics (1-5)
    num_topics = st.number_input("Number of topics to compare", min_value=1, max_value=5, value=2)
    
    # Create URL input fields based on the number of topics
    urls = []
    for i in range(num_topics):
        st.subheader(f"Topic {i+1}")
        
        # Default search terms
        default_search = ""
        if i == 0:
            default_search = "Python programming language"
        elif i == 1:
            default_search = "JavaScript"
            
        # Search box for Wikipedia topics
        search_query = st.text_input(
            f"Search Wikipedia topics {i+1}",
            value=default_search,
            key=f"search_{i}"
        )
        
        # Search for matching topics
        matching_topics = []
        if search_query:
            matching_topics = search_wikipedia_topics(search_query)
        
        # Display dropdown for topic selection if results found
        selected_topic = None
        if matching_topics:
            selected_topic = st.selectbox(
                f"Select a topic {i+1}",
                options=matching_topics,
                key=f"topic_{i}"
            )
            
            # Convert selected topic to Wikipedia URL
            if selected_topic:
                wiki_url = f"https://en.wikipedia.org/wiki/{quote(selected_topic.replace(' ', '_'))}"
                st.success(f"Using: {wiki_url}")
                urls.append(wiki_url)
        else:
            # Fallback to direct URL input if no search results or no search query
            if search_query:
                st.info("No matching topics found. Enter a URL directly.")
                
            direct_url = st.text_input(
                f"Or enter Wikipedia URL directly {i+1}",
                value="" if search_query and not matching_topics else ("https://en.wikipedia.org/wiki/Python_(programming_language)" if i == 0 else 
                      "https://en.wikipedia.org/wiki/JavaScript" if i == 1 else ""),
                key=f"url_{i}"
            )
            if direct_url:
                urls.append(direct_url)
    
    # Date range selection
    st.subheader("Date Range")
    end_date_val = datetime.now()
    
    # Predefined date ranges
    date_range_options = {
        "Last 7 days": end_date_val - timedelta(days=7),
        "Last 30 days": end_date_val - timedelta(days=30),
        "Last 90 days": end_date_val - timedelta(days=90),
        "Last 6 months": end_date_val - timedelta(days=180),
        "Last year": end_date_val - timedelta(days=365),
        "Custom": None
    }
    
    selected_range = st.selectbox("Select time period", list(date_range_options.keys()))
    
    if selected_range == "Custom":
        # Custom date range
        min_date_val = datetime(2015, 7, 1)  # Wikimedia pageview API data starts from July 2015
        
        start_date = st.date_input(
            "Start Date",
            value=end_date_val - timedelta(days=30),
            min_value=min_date_val,
            max_value=end_date_val
        )
        
        end_date = st.date_input(
            "End Date",
            value=end_date_val,
            min_value=min_date_val,
            max_value=datetime.now()
        )
    else:
        start_date = date_range_options[selected_range].date()  # Fixed for date object
        end_date = end_date_val.date()  # Fixed for date object
        # Display the selected dates for reference
        st.info(f"From: {start_date.strftime('%Y-%m-%d')} To: {end_date.strftime('%Y-%m-%d')}")
    
    # Convert to datetime objects - Fixed for date object check
    if isinstance(start_date, date):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date):
        end_date = datetime.combine(end_date, datetime.min.time())
    
    # Display options
    st.subheader("Display Options")
    
    # Toggle for absolute vs relative comparison
    show_relative = st.checkbox("Show relative comparison (%)", value=False, 
                              help="Toggle between absolute page views and percentage comparison")
    
    # Select export format
    export_format = st.selectbox(
        "Export data format",
        ["CSV", "Excel", "JSON"],
        help="Choose the format for data export"
    )
    
    # Show related topics option
    show_related = st.checkbox("Show related topics", value=True, 
                             help="Display related Wikipedia topics for each analyzed page")
    
    # Analyze button
    analyze_button = st.button("Analyze Trends", use_container_width=True)

# Main content area
if analyze_button:
    # Validate and extract page titles
    page_titles = []
    valid_urls = []
    
    for url in urls:
        if not url:  # Skip empty URLs
            continue
            
        page_title = extract_page_title(url)
        if page_title:
            page_titles.append(page_title)
            valid_urls.append(url)
        else:
            st.error(f"Invalid Wikipedia URL: {url}. Please use format: https://en.wikipedia.org/wiki/Page_Title")
    
    if not page_titles:
        st.error("No valid Wikipedia URLs provided. Please enter at least one valid URL.")
    else:
        with st.spinner("Fetching data... This may take a moment"):
            # Create progress container
            progress_container = st.empty()
            
            # Fetch data for each page
            dfs = []
            for i, page_title in enumerate(page_titles):
                progress_container.progress((i / len(page_titles)), f"Fetching data for {page_title.replace('_', ' ')}...")
                df = fetch_pageviews(page_title, start_date, end_date)
                
                if not df.empty:
                    display_title = page_title.replace('_', ' ')
                    df.rename(columns={'views': display_title}, inplace=True)
                    dfs.append(df)
                else:
                    st.error(f"Failed to fetch data for {page_title.replace('_', ' ')}.")
            
            progress_container.empty()
            
            if not dfs:
                st.error("Failed to fetch data for any of the specified pages.")
            else:
                # Start with the first dataframe
                merged_df = dfs[0]
                
                # Merge with the rest
                for df in dfs[1:]:
                    merged_df = pd.merge(merged_df, df, on='date', how='outer')
                
                # Fill missing values with 0
                merged_df.fillna(0, inplace=True)
                
                # Sort by date
                merged_df = merged_df.sort_values('date')
                
                # Create a copy for visualization with only the data columns
                display_titles = [title.replace('_', ' ') for title in page_titles]
                plot_df = merged_df.copy()
                
                # Create relative comparison if selected
                if show_relative:
                    # Calculate percentages
                    total_views = plot_df[display_titles].sum(axis=1)
                    for col in display_titles:
                        plot_df[col] = plot_df[col] / total_views * 100
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Visualization", "Data Table", "Statistics"])
                
                with tab1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    
                    # Create visualization
                    fig, ax = plt.subplots(figsize=(12, 7))
                    
                    # Set style similar to Glimpse
                    plt.style.use('seaborn-v0_8-whitegrid')
                    ax.set_facecolor('#f9f9f9')
                    fig.patch.set_facecolor('#ffffff')
                    
                    # Plot each line with a distinct color from a colorblind-friendly palette
                    colors = sns.color_palette("husl", len(display_titles))
                    
                    for i, title in enumerate(display_titles):
                        ax.plot(plot_df['date'], plot_df[title], color=colors[i], linewidth=2.5, label=title)
                    
                    # Set title and labels
                    if show_relative:
                        plt.title('Relative Interest Comparison', fontsize=16, fontweight='bold')
                        plt.ylabel('Percentage of Total Views (%)', fontsize=12)
                    else:
                        plt.title('Wikipedia Page Views Comparison', fontsize=16, fontweight='bold')
                        plt.ylabel('Number of Views', fontsize=12)
                    
                    plt.xlabel('Date', fontsize=12)
                    
                    # Format x-axis dates
                    date_format = plt.matplotlib.dates.DateFormatter('%b %d, %Y')
                    ax.xaxis.set_major_formatter(date_format)
                    fig.autofmt_xdate()
                    
                    # Add legend
                    plt.legend(fontsize=10, frameon=True, framealpha=0.8, 
                              title="Wikipedia Topics", title_fontsize=11)
                    
                    # Tight layout
                    plt.tight_layout()
                    
                    # Show the plot
                    st.pyplot(fig)
                    
                    # Download image button
                    plot_download = get_image_download_link(fig, 
                                                          "wikipedia_interest_comparison.png", 
                                                          "⬇️ Download Plot as PNG")
                    st.markdown(plot_download, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with tab2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    
                    # Display the dataframe
                    st.dataframe(merged_df, use_container_width=True)
                    
                    # Format for export based on selection
                    format_lower = export_format.lower()
                    download_link = get_table_download_link(
                        merged_df, 
                        f"wikipedia_pageviews_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}", 
                        f"⬇️ Download data as {export_format}", 
                        format_lower
                    )
                    st.markdown(download_link, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tab3:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    
                    # Calculate statistics for each topic
                    stats_data = []
                    
                    for title in display_titles:
                        total_views = int(merged_df[title].sum())
                        avg_views = int(merged_df[title].mean())
                        max_views = int(merged_df[title].max())
                        max_date = merged_df.loc[merged_df[title].idxmax(), 'date']
                        
                        # Calculate growth metrics
                        if len(merged_df) > 1:
                            first_half = merged_df.iloc[:len(merged_df)//2]
                            second_half = merged_df.iloc[len(merged_df)//2:]
                            
                            first_avg = first_half[title].mean()
                            second_avg = second_half[title].mean()
                            
                            if first_avg > 0:
                                growth_rate = ((second_avg - first_avg) / first_avg) * 100
                            else:
                                growth_rate = 0
                        else:
                            growth_rate = 0
                        
                        stats_data.append({
                            'Topic': title,
                            'Total Views': total_views,
                            'Average Daily Views': avg_views,
                            'Peak Views': max_views,
                            'Peak Date': max_date.strftime('%Y-%m-%d'),
                            'Trend': f"{growth_rate:.1f}%"
                        })
                    
                    stats_df = pd.DataFrame(stats_data)
                    
                    # Display statistics
                    st.subheader("Topic Statistics")
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Create statistics cards for each topic
                    st.subheader("Key Metrics")
                    
                    # Display metrics in rows of 3
                    for i in range(0, len(display_titles), 3):
                        cols = st.columns(3)
                        for j in range(3):
                            if i + j < len(display_titles):
                                title = display_titles[i + j]
                                with cols[j]:
                                    st.markdown(f"<h3 style='text-align: center;'>{title}</h3>", unsafe_allow_html=True)
                                    
                                    stat = stats_data[i + j]
                                    
                                    # Total views
                                    st.markdown(
                                        f"""
                                        <div class='stat-box'>
                                            <p class='stat-value'>{stat['Total Views']:,}</p>
                                            <p class='stat-label'>Total Views</p>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Average views
                                    st.markdown(
                                        f"""
                                        <div class='stat-box' style='margin-top: 15px;'>
                                            <p class='stat-value'>{stat['Average Daily Views']:,}</p>
                                            <p class='stat-label'>Average Daily Views</p>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                                    
                                    # Growth trend
                                    trend_color = "#4CAF50" if float(stat['Trend'].strip('%')) > 0 else "#F44336"
                                    trend_arrow = "↗" if float(stat['Trend'].strip('%')) > 0 else "↘"
                                    st.markdown(
                                        f"""
                                        <div class='stat-box' style='margin-top: 15px;'>
                                            <p class='stat-value' style='color: {trend_color};'>{trend_arrow} {stat['Trend']}</p>
                                            <p class='stat-label'>Growth Trend</p>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display related topics if enabled
                if show_related:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Related Topics You Might Be Interested In")
                    
                    related_topics_all = {}
                    
                    for page_title in page_titles:
                        with st.spinner(f"Finding related topics for {page_title.replace('_', ' ')}..."):
                            related = fetch_related_topics(page_title)
                            if related:
                                related_topics_all[page_title] = related
                    
                    if related_topics_all:
                        for page_title, related in related_topics_all.items():
                            display_title = page_title.replace('_', ' ')
                            st.markdown(f"#### For {display_title}:")
                            
                            # Display in a horizontal row
                            cols = st.columns(len(related))
                            for i, topic in enumerate(related):
                                with cols[i]:
                                    topic_url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
                                    st.markdown(f"[{topic}]({topic_url})")
                    else:
                        st.info("No related topics found.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
else:
    # Display welcome screen
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        # Discover Interest Patterns in Wikipedia Topics
        
        Use this tool to track and compare interest levels in up to 5 Wikipedia topics over time,
        based on actual page view data from the Wikimedia API.
        
        ### What You Can Do:
        - Compare multiple Wikipedia topics simultaneously
        - View absolute or relative interest levels
        - Download data in multiple formats
        - Explore related topics for further research
        
        ### Getting Started:
        1. Search for Wikipedia topics in the sidebar
        2. Select your desired date range
        3. Click "Analyze Trends" to begin
        """)
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Pan_Blue_Circle.svg/240px-Pan_Blue_Circle.svg.png", width=200)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Example topics (like Glimpse's format)
    st.markdown("<h3>Popular Wikipedia Topics</h3>", unsafe_allow_html=True)
    
    example_topics = [
        {"name": "Artificial Intelligence", "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"},
        {"name": "Machine Learning", "url": "https://en.wikipedia.org/wiki/Machine_learning"},
        {"name": "Data Science", "url": "https://en.wikipedia.org/wiki/Data_science"},
        {"name": "Blockchain", "url": "https://en.wikipedia.org/wiki/Blockchain"},
        {"name": "Climate Change", "url": "https://en.wikipedia.org/wiki/Climate_change"},
        {"name": "Renewable Energy", "url": "https://en.wikipedia.org/wiki/Renewable_energy"},
        {"name": "Quantum Computing", "url": "https://en.wikipedia.org/wiki/Quantum_computing"},
        {"name": "Virtual Reality", "url": "https://en.wikipedia.org/wiki/Virtual_reality"}
    ]
    
    # Display in a grid of 4 columns
    cols = st.columns(4)
    for i, topic in enumerate(example_topics):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style='background-color: #f5f7f9; padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
                    <h4 style='margin: 0;'>{topic['name']}</h4>
                    <a href="{topic['url']}" target="_blank" style='color: #6247aa;'>View on Wikipedia</a>
                </div>
                """, 
                unsafe_allow_html=True
            )

# Footer
st.markdown("""
<div class='footer'>
    <p>Data source: Wikimedia Pageviews API • Created with Streamlit</p>
    <p>© 2025 Wiki Trends - Discover interest patterns in Wikipedia topics</p>
</div>
""", unsafe_allow_html=True)
