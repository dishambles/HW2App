import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime, date, timedelta
import re
import base64
import io
import json
from urllib.parse import unquote, quote
from dateutil.relativedelta import relativedelta
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import mwoauth
import requests_oauthlib
from requests_oauthlib import OAuth1Session
from streamlit_plotly_events import plotly_events
from streamlit_elements import elements
import tempfile
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from collections import Counter
import os
from PIL import Image
import html

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

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
    
    /* Network graph */
    .network-card {
        height: 600px;
        background-color: white;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Auth section */
    .auth-section {
        background-color: #f0f8ff;
        border-left: 4px solid var(--accent-color);
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 0 5px 5px 0;
    }
    
    /* Content editor */
    .edit-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        background-color: #fafafa;
    }
    
    /* Gap analysis */
    .gap-item {
        background-color: #fff;
        border-left: 3px solid #ff9800;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 0 5px 5px 0;
    }
    
    /* News events */
    .news-event {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .news-event:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .news-date {
        color: #777;
        font-size: 0.8rem;
    }
    .news-title {
        font-weight: 600;
        color: var(--secondary-color);
        margin: 5px 0;
    }
    .news-source {
        color: var(--primary-color);
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'selected_peak' not in st.session_state:
    st.session_state.selected_peak = None
if 'topic_data' not in st.session_state:
    st.session_state.topic_data = {}
if 'pageview_data' not in st.session_state:
    st.session_state.pageview_data = None

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

def fetch_related_topics(page_title, limit=8):
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

def get_page_content(page_title, section=None):
    """Fetch the content of a Wikipedia page"""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": page_title.replace("_", " "),
        "format": "json",
        "prop": "text",
        "formatversion": "2"
    }
    
    if section is not None:
        params["section"] = section
    
    response = requests.get(url, params=params, headers={'User-Agent': 'WikiInterestComparisonTool/1.0'})
    
    if response.status_code == 200:
        data = response.json()
        if 'parse' in data and 'text' in data['parse']:
            html_content = data['parse']['text']
            return html_content
        else:
            return None
    else:
        return None

def get_page_sections(page_title):
    """Get the sections of a Wikipedia page"""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": page_title.replace("_", " "),
        "prop": "sections",
        "format": "json"
    }
    
    response = requests.get(url, params=params, headers={'User-Agent': 'WikiInterestComparisonTool/1.0'})
    
    if response.status_code == 200:
        data = response.json()
        if 'parse' in data and 'sections' in data['parse']:
            sections = data['parse']['sections']
            return [(section['index'], section['line']) for section in sections]
        else:
            return []
    else:
        return []

def analyze_content_gaps(page_title):
    """Analyze content gaps in a Wikipedia article"""
    html_content = get_page_content(page_title)
    if not html_content:
        return {
            "missing_citations": [],
            "content_gaps": ["Could not retrieve article content"],
            "improvement_areas": []
        }
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Look for citation needed tags
    citation_needed = soup.find_all('span', {'class': 'citation-needed'})
    missing_citations = [
        f"'{parent.get_text()}'" 
        for tag in citation_needed 
        if (parent := tag.parent)
    ]
    
    # Look for short sections (potential content gaps)
    sections = soup.find_all(['h2', 'h3'])
    content_gaps = []
    
    for section in sections:
        section_title = section.get_text().strip()
        next_section = section.find_next(['h2', 'h3'])
        
        # Get all content between this section and the next
        content = []
        current = section.next_sibling
        while current and current != next_section:
            if current.name and current.get_text().strip():
                content.append(current.get_text())
            current = current.next_sibling
        
        # If content is minimal, flag as a gap
        content_text = ' '.join(content)
        if len(content_text.split()) < 50 and section_title:  # Fewer than 50 words
            content_gaps.append(f"Section '{section_title}' is very brief and could be expanded")
    
    # Look for general improvement areas
    improvement_areas = []
    
    # Check for images
    images = soup.find_all('img')
    if len(images) < 2:
        improvement_areas.append("Article could benefit from additional images")
    
    # Check for references
    references = soup.find_all('ol', {'class': 'references'})
    if not references:
        improvement_areas.append("Article lacks references")
    else:
        ref_items = references[0].find_all('li')
        if len(ref_items) < 5:
            improvement_areas.append(f"Article has only {len(ref_items)} references (more recommended)")
    
    # Check for external links
    external_links = soup.find_all('a', {'class': 'external'})
    if len(external_links) < 3:
        improvement_areas.append("Few external links provided")
    
    # If we didn't find specific gaps, add some generic suggestions
    if not content_gaps:
        sections = get_page_sections(page_title)
        standard_sections = ["History", "Development", "Features", "Reception", "Impact", "See also", "References"]
        
        section_titles = [s[1] for s in sections]
        for std_section in standard_sections:
            if not any(std_section in title for title in section_titles):
                content_gaps.append(f"No '{std_section}' section")
    
    return {
        "missing_citations": missing_citations[:5],  # Limit to 5 for display
        "content_gaps": content_gaps[:5],
        "improvement_areas": improvement_areas[:5]
    }

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

def create_topic_similarity_network(page_titles, merged_df):
    """Create an interactive network visualization of topic relationships"""
    # Create a network graph
    G = nx.Graph()
    
    # Add nodes for each topic
    for title in page_titles:
        display_title = title.replace('_', ' ')
        G.add_node(display_title)
    
    # Calculate correlation between topics
    if merged_df is not None and not merged_df.empty:
        display_titles = [title.replace('_', ' ') for title in page_titles]
        correlation_matrix = merged_df[display_titles].corr()
        
        # Add edges based on correlation strength
        for i, topic1 in enumerate(display_titles):
            for j, topic2 in enumerate(display_titles):
                if i < j:  # Avoid duplicate edges and self-loops
                    correlation = correlation_matrix.loc[topic1, topic2]
                    # Only add edges for correlations above a threshold
                    if correlation > 0.3:
                        weight = correlation * 10  # Scale for better visualization
                        G.add_edge(topic1, topic2, weight=weight)
    
    # Get related topics for each page title to expand the network
    for page_title in page_titles:
        display_title = page_title.replace('_', ' ')
        related = fetch_related_topics(page_title, limit=3)
        
        for related_topic in related:
            G.add_node(related_topic)
            G.add_edge(display_title, related_topic, weight=5)  # Add edge with default weight
    
    # Create an interactive network visualization
    nt = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="#333333")
    
    # Set node colors: main topics in purple, related topics in blue
    for node in G.nodes():
        if node in [t.replace('_', ' ') for t in page_titles]:
            nt.add_node(node, color="#6247aa", size=25, title=node)
        else:
            nt.add_node(node, color="#50b8e7", size=15, title=node)
    
    # Add edges with weights determining thickness
    for edge in G.edges(data=True):
        nt.add_edge(edge[0], edge[1], value=edge[2].get('weight', 1), title=f"Correlation: {edge[2].get('weight', 0)/10:.2f}")
    
    # Set physics layout
    nt.barnes_hut(gravity=-10000, central_gravity=0.3, spring_length=200)
    
    # Generate the HTML file
    html_file = "topic_network.html"
    nt.save_graph(html_file)
    
    return html_file

def fetch_news_events(topic, date, limit=5):
    """Fetch news events related to a topic around a specific date"""
    # In a real implementation, you would integrate with a news API
    # Here, we'll simulate results based on Wikipedia page content
    
    # Format for display
    display_date = date.strftime('%Y-%m-%d')
    
    # Try to get real article content for context
    html_content = get_page_content(topic)
    mock_news = []
    
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        paragraphs = soup.find_all('p')
        
        # Extract text from paragraphs
        text_content = ' '.join([p.get_text() for p in paragraphs])
        
        # Extract significant phrases (simulating news headlines)
        words = nltk.word_tokenize(text_content)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words and len(word) > 3]
        
        # Get phrases from the content
        phrases = []
        for i in range(len(words) - 4):
            if all(w.isalpha() for w in words[i:i+4]):
                phrase = ' '.join(words[i:i+4])
                if len(phrase) > 15:
                    phrases.append(phrase)
        
        # If we have phrases, use them for mock news
        if phrases:
            # Generate some mock news entries
            import random
            selected_phrases = random.sample(phrases, min(limit, len(phrases)))
            
            sources = ["The Tech Chronicle", "Digital Times", "Science Today", "Innovation Weekly", "Data Report"]
            
            # Create mock news around the date
            for i, phrase in enumerate(selected_phrases):
                # Create a date within 3 days of the target date
                news_date = date + timedelta(days=random.randint(-3, 3))
                
                mock_news.append({
                    "title": phrase + "...",
                    "source": random.choice(sources),
                    "date": news_date.strftime("%Y-%m-%d"),
                    "url": f"https://example.com/news/{i}"
                })
    
    # If we couldn't generate realistic mock news, use these defaults
    if not mock_news:
        mock_news = [
            {
                "title": f"New developments in {topic.replace('_', ' ')} announced",
                "source": "Tech News Daily",
                "date": (date - timedelta(days=1)).strftime("%Y-%m-%d"),
                "url": "https://example.com/news/1"
            },
            {
                "title": f"Experts discuss the future of {topic.replace('_', ' ')}",
                "source": "Science Weekly",
                "date": date.strftime("%Y-%m-%d"),
                "url": "https://example.com/news/2"
            },
            {
                "title": f"Research breakthrough related to {topic.replace('_', ' ')}",
                "source": "Research Digest",
                "date": (date + timedelta(days=1)).strftime("%Y-%m-%d"),
                "url": "https://example.com/news/3"
            }
        ]
    
    return mock_news

def detect_pageview_peaks(df, topic_column, sensitivity=2.0):
    """Detect significant peaks in the pageview data"""
    if topic_column not in df.columns or df.empty:
        return []
    
    # Calculate the mean and standard deviation of the views
    mean_views = df[topic_column].mean()
    std_views = df[topic_column].std()
    
    # Calculate the threshold for what constitutes a peak
    threshold = mean_views + (sensitivity * std_views)
    
    # Find dates where views exceed the threshold
    peaks = df[df[topic_column] > threshold]
    
    # Sort by view count (descending) and get the top 5 peaks
    peaks = peaks.sort_values(by=topic_column, ascending=False).head(5)
    
    # Return the peak dates and view counts
    return [
        {'date': row['date'], 'views': row[topic_column]} 
        for _, row in peaks.iterrows()
    ]

def setup_wikipedia_oauth():
    """Set up OAuth authentication with Wikipedia"""
    # In a production app, you'd get these from your OAuth registration
    consumer_key = st.secrets.get("WIKI_CONSUMER_KEY", "your-consumer-key")
    consumer_secret = st.secrets.get("WIKI_CONSUMER_SECRET", "your-consumer-secret")
    
    # Mock authentication for demo purposes
    st.info("🔑 In a production app, this would connect to Wikipedia's OAuth")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input("Wikipedia Username", placeholder="Enter your Wikipedia username")
        st.text_input("Password", type="password", placeholder="Enter your password")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Login to Wikipedia"):
            st.session_state.authenticated = True
            st.success("Authentication successful (simulated)")
            st.rerun()
    
    st.markdown("""
    <div style="font-size: 0.8rem; color: #666;">
        Note: In the production version, this would securely connect to Wikipedia's OAuth system
        without storing your password and following API guidelines for third-party editors.
    </div>
    """, unsafe_allow_html=True)

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
        start_date = date_range_options[selected_range].date()
        end_date = end_date_val.date()
        # Display the selected dates for reference
        st.info(f"From: {start_date.strftime('%Y-%m-%d')} To: {end_date.strftime('%Y-%m-%d')}")
    
    # Convert to datetime objects
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
    
    # Show advanced features
    show_network = st.checkbox("Show topic similarity network", value=True, 
                             help="Display interactive network of topic relationships")
    
    show_peaks = st.checkbox("Enable click-through peak analysis", value=True,
                           help="Click on peaks to see related news events")
    
    show_editor = st.checkbox("Enable content gap analyzer & editor", value=True,
                            help="Analyze and edit Wikipedia content")
    
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
                    
                    # Store page title and display title mapping
                    st.session_state.topic_data[display_title] = page_title
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
                
                # Store in session state for use in other tabs
                st.session_state.pageview_data = merged_df
                
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
                tab_list = ["Visualization", "Interactive Timeline", "Data Table", "Statistics"]
                
                if show_network:
                    tab_list.append("Topic Network")
                
                if show_editor and st.session_state.authenticated:
                    tab_list.append("Content Editor")
                elif show_editor:
                    tab_list.append("Content Analyzer")
                
                tabs = st.tabs(tab_list)
                
                with tabs[0]:  # Visualization tab
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
                
                with tabs[1]:  # Interactive Timeline tab
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Interactive Zoomable Timeline")
                    st.markdown("Zoom in by selecting an area, double-click to reset, click on peaks to see news events.")
                    
                    # Create an interactive Plotly figure
                    fig = go.Figure()
                    
                    for i, title in enumerate(display_titles):
                        fig.add_trace(go.Scatter(
                            x=plot_df['date'],
                            y=plot_df[title],
                            mode='lines',
                            name=title,
                            line=dict(width=3),
                            hovertemplate='%{x}<br>Views: %{y:,.0f}<extra></extra>'
                        ))
                        
                        # If peak analysis is enabled, mark the peaks
                        if show_peaks:
                            # Detect peaks for this topic
                            peaks = detect_pageview_peaks(plot_df, title)
                            
                            # Add markers for the peaks
                            peak_dates = [peak['date'] for peak in peaks]
                            peak_views = [peak['views'] for peak in peaks]
                            
                            fig.add_trace(go.Scatter(
                                x=peak_dates,
                                y=peak_views,
                                mode='markers',
                                marker=dict(size=10, symbol='circle', line=dict(width=2, color='white')),
                                name=f"{title} Peaks",
                                hovertemplate='%{x}<br>Peak: %{y:,.0f} views<br>Click for news events<extra></extra>'
                            ))
                    
                    # Set layout
                    fig.update_layout(
                        height=600,
                        xaxis_title="Date",
                        yaxis_title="Views" if not show_relative else "Percentage (%)",
                        hovermode="closest",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    # Add range slider
                    fig.update_layout(
                        xaxis=dict(
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=7, label="1w", step="day", stepmode="backward"),
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(count=3, label="3m", step="month", stepmode="backward"),
                                    dict(step="all")
                                ])
                            ),
                            rangeslider=dict(visible=True),
                            type="date"
                        )
                    )
                    
                    # Make the plot
                    selected_points = plotly_events(fig, click_event=True, hover_event=False)
                    
                    # Handle peak selection for news events
                    if selected_points and show_peaks:
                        selected_date = pd.to_datetime(selected_points[0]['x'])
                        selected_value = selected_points[0]['y']
                        
                        # Find which topic this point belongs to
                        for title in display_titles:
                            if abs(plot_df.loc[plot_df['date'] == selected_date, title].values[0] - selected_value) < 0.01:
                                selected_topic = st.session_state.topic_data.get(title)
                                
                                if selected_topic:
                                    st.markdown("### 📰 News Events Around This Peak")
                                    
                                    events = fetch_news_events(selected_topic, selected_date)
                                    
                                    for event in events:
                                        st.markdown(f"""
                                        <div class="news-event">
                                            <div class="news-date">{event['date']}</div>
                                            <div class="news-title">{event['title']}</div>
                                            <div class="news-source">{event['source']}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.markdown("---")
                                    st.markdown("ℹ️ *Note: In a production app, these would be real news events from external APIs*")
                                break
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tabs[2]:  # Data Table tab
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
                
                with tabs[3]:  # Statistics tab
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
                                    
                                    # Day of Week Analysis
                                    if len(merged_df) >= 7:  # Only show if we have at least a week of data
                                        # Calculate average views by day of week
                                        merged_df['day_of_week'] = merged_df['date'].dt.day_name()
                                        dow_avg = merged_df.groupby('day_of_week')[title].mean()
                                        
                                        # Create a small bar chart for day of week pattern
                                        fig, ax = plt.subplots(figsize=(6, 3))
                                        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                                        day_data = [dow_avg.get(day, 0) for day in days]
                                        
                                        ax.bar(days, day_data, color="#6247aa", alpha=0.7)
                                        ax.set_title(f"Day of Week Pattern", fontsize=12)
                                        ax.set_xticklabels(days, rotation=45, ha='right')
                                        ax.set_ylabel("Avg. Views")
                                        plt.tight_layout()
                                        
                                        st.pyplot(fig)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                if show_network and len(tab_list) > 4:
                    with tabs[4]:  # Topic Network tab
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.subheader("Topic Similarity Network")
                        st.markdown("This network shows relationships between the analyzed topics and related Wikipedia pages.")
                        
                        # Create the network visualization
                        network_file = create_topic_similarity_network(page_titles, merged_df)
                        
                        # Display the network in an iframe
                        st.components.v1.html(open(network_file, 'r').read(), height=600)
                        
                        # Explanation of the visualization
                        st.markdown("""
                        ### Understanding the Network
                        
                        - **Purple nodes**: Your selected topics
                        - **Blue nodes**: Related Wikipedia topics
                        - **Edges (connections)**: Indicate relationship strength between topics
                        - **Thicker edges**: Stronger relationships based on pageview correlation
                        
                        You can drag nodes to explore the network, zoom in/out, and hover over nodes for topic names.
                        """)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Content Gap Analyzer & Editor
                if show_editor and ((st.session_state.authenticated and len(tab_list) > 5) or 
                                   (not st.session_state.authenticated and len(tab_list) > 4)):
                    tab_index = 5 if st.session_state.authenticated else 4
                    
                    with tabs[tab_index]:  # Content Analyzer/Editor tab
                        if not st.session_state.authenticated:
                            # Show authentication form if not logged in
                            st.markdown("<div class='auth-section'>", unsafe_allow_html=True)
                            st.subheader("Login to Enable Content Editing")
                            st.markdown("To edit Wikipedia content, you need to authenticate with your Wikipedia account.")
                            
                            setup_wikipedia_oauth()
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Always show content analysis
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.header("Content Gap Analyzer")
                        st.markdown("This tool identifies potential improvements for Wikipedia articles.")
                        
                        # Select topic to analyze
                        selected_topic_display = st.selectbox(
                            "Select topic to analyze",
                            options=display_titles,
                            key="content_topic"
                        )
                        
                        if selected_topic_display:
                            # Get the corresponding page title
                            selected_topic = st.session_state.topic_data.get(selected_topic_display)
                            
                            if selected_topic:
                                with st.spinner(f"Analyzing content for {selected_topic_display}..."):
                                    # Perform content gap analysis
                                    gap_analysis = analyze_content_gaps(selected_topic)
                                    
                                    # Display the results
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.subheader("Missing Citations")
                                        if gap_analysis["missing_citations"]:
                                            for citation in gap_analysis["missing_citations"]:
                                                st.markdown(f"""
                                                <div class="gap-item">
                                                    <strong>Needs Citation:</strong> {citation}
                                                </div>
                                                """, unsafe_allow_html=True)
                                        else:
                                            st.info("No missing citations detected.")
                                    
                                    with col2:
                                        st.subheader("Content Gaps")
                                        if gap_analysis["content_gaps"]:
                                            for gap in gap_analysis["content_gaps"]:
                                                st.markdown(f"""
                                                <div class="gap-item">
                                                    <strong>Gap:</strong> {gap}
                                                </div>
                                                """, unsafe_allow_html=True)
                                        else:
                                            st.info("No significant content gaps detected.")
                                    
                                    st.subheader("General Improvement Areas")
                                    if gap_analysis["improvement_areas"]:
                                        for improvement in gap_analysis["improvement_areas"]:
                                            st.markdown(f"""
                                            <div class="gap-item" style="border-left-color: #4CAF50;">
                                                <strong>Improvement:</strong> {improvement}
                                            </div>
                                            """, unsafe_allow_html=True)
                                    else:
                                        st.info("No general improvements needed.")
                                        
                                    # Editor section (if authenticated)
                                    if st.session_state.authenticated:
                                        st.markdown("---")
                                        st.header("Content Editor")
                                        st.markdown("Make improvements to the Wikipedia article based on the analysis.")
                                        
                                        # Get page sections
                                        sections = get_page_sections(selected_topic)
                                        section_options = ["(Entire article)"] + [s[1] for s in sections]
                                        
                                        selected_section = st.selectbox(
                                            "Choose section to edit",
                                            options=section_options,
                                            key="edit_section"
                                        )
                                        
                                        section_index = None
                                        if selected_section != "(Entire article)":
                                            # Find the section index
                                            for s in sections:
                                                if s[1] == selected_section:
                                                    section_index = s[0]
                                                    break
                                        
                                        # Get current content
                                        content_html = get_page_content(selected_topic, section_index)
                                        
                                        if content_html:
                                            # Convert HTML to text for editing
                                            soup = BeautifulSoup(content_html, 'html.parser')
                                            content_text = soup.get_text()
                                            
                                            # Display in a text area
                                            st.markdown("<div class='edit-box'>", unsafe_allow_html=True)
                                            edited_content = st.text_area(
                                                "Edit content",
                                                value=content_text,
                                                height=300,
                                                key="content_editor"
                                            )
                                            st.markdown("</div>", unsafe_allow_html=True)
                                            
                                            # Edit summary
                                            edit_summary = st.text_input(
                                                "Edit summary",
                                                placeholder="Describe your changes",
                                                key="edit_summary"
                                            )
                                            
                                            # Minor edit checkbox
                                            minor_edit = st.checkbox("This is a minor edit", value=False, key="minor_edit")
                                            
                                            # Submit button
                                            if st.button("Submit Changes to Wikipedia", use_container_width=True):
                                                # In a real app, this would send the edits to Wikipedia API
                                                st.success("Changes submitted successfully (simulated)")
                                                st.info("""
                                                In a production app, these changes would be submitted to Wikipedia using the API.
                                                The actual API calls would:
                                                1. Get an edit token from the API
                                                2. Submit the edited content with proper authentication
                                                3. Track the edit in Wikipedia's edit history
                                                """)
                                        else:
                                            st.error("Failed to retrieve content for editing.")
                        
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
        - Explore interactive visualizations including topic networks
        - Analyze content gaps and improve Wikipedia articles
        - Discover news events related to traffic spikes
        - Download data in multiple formats
        
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
    
    # Feature highlights
    st.markdown("<h3>Advanced Features</h3>", unsafe_allow_html=True)
    
    feature_cols = st.columns(3)
    
    with feature_cols[0]:
        st.markdown(
            """
            <div style='background-color: #eee6ff; padding: 20px; border-radius: 10px; height: 200px;'>
                <h4>Topic Similarity Network</h4>
                <p>Visualize relationships between topics with an interactive network graph showing connections and similarities.</p>
                <p style='color: #6247aa;'><strong>✓ Included</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with feature_cols[1]:
        st.markdown(
            """
            <div style='background-color: #e6f7ff; padding: 20px; border-radius: 10px; height: 200px;'>
                <h4>Zoomable Interactive Timeline</h4>
                <p>Explore pageview data with an interactive timeline that allows zooming, panning, and clicking on peaks.</p>
                <p style='color: #50b8e7;'><strong>✓ Included</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with feature_cols[2]:
        st.markdown(
            """
            <div style='background-color: #e6ffee; padding: 20px; border-radius: 10px; height: 200px;'>
                <h4>Content Gap Analyzer & Editor</h4>
                <p>Identify improvements needed in Wikipedia articles and edit them directly through the app.</p>
                <p style='color: #4CAF50;'><strong>✓ Included</strong></p>
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
