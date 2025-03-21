# Wiki Trends - Advanced Wikipedia Interest Tracker

This Streamlit application allows you to track, compare, and analyze interest levels in Wikipedia topics over time based on page view data.

## Features

### Core Features
- Compare up to 5 Wikipedia topics simultaneously
- Search for Wikipedia topics directly within the app
- View absolute or relative interest levels
- Visualize trends with interactive charts
- Download data in multiple formats (CSV, Excel, JSON)
- Explore related topics for further research

### Advanced Features
- **Topic Similarity Network**: Interactive graph showing relationships between topics
- **Zoomable Timeline**: Interactive timeline that allows zooming in on specific periods
- **Click-Through Analysis**: Click on peaks to see related news events from that day
- **Content Gap Analyzer & Editor**: Identify and fix content gaps in Wikipedia articles

## Deployment

This app is deployed using Streamlit Cloud. You can access it at [your-app-url-here].

## Local Development

To run this app locally:

1. Clone this repository
2. Install the requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Data Sources

- **Pageview Data**: [Wikimedia Pageviews API](https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews)
- **Article Content**: Wikipedia API
- **Topic Relationships**: Generated using Wikipedia API and pageview correlations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
