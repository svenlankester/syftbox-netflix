import json
from pathlib import Path
from datetime import datetime

def populate_html_template(json_path: Path, template_path: Path, output_path: Path, num_participants: int, recommendations: list = None):
    """
    Populates an HTML template with data from a JSON file and saves the result.
    
    Args:
        json_path (Path): Path to the JSON file containing the top series data.
        template_path (Path): Path to the HTML template file.
        output_path (Path): Path to save the populated HTML file.
        num_participants (int): Total number of participants.
        recommendations (list, optional): List of recommended series dictionaries.
    """
    try:
        # Load the JSON data for top watched series
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load the HTML template
        with open(template_path, 'r', encoding='utf-8') as f:
            html_template = f.read()

        # Generate the series cards for top watched series
        series_cards = ""
        for item in data:
            series_cards += f"""
            <div class="series-item">
                <img width="110" height="155" src="{item['img']}" alt="{item['name']}">
                <div><strong>{item['name']}</strong></div>
                <p>Language: {item['language']}</p>
                <p>Rating: {item['rating']}</p>
                <p>IMDB: {item['imdb']}</p>
            </div>
            """

        # Generate the series cards for recommended series (if provided)
        recommended_cards = ""
        if recommendations:
            for item in recommendations:
                recommended_cards += f"""
                <div class="series-item">
                    <img width="110" height="155" src="{item['img']}" alt="{item['name']}">
                    <div><strong>{item['name']}</strong></div>
                    <p>Language: {item['language']}</p>
                    <p>Rating: {item['rating']}</p>
                    <p>IMDB: {item['imdb']}</p>
                </div>
                """

        # Replace the placeholders in the template
        populated_html = html_template.replace(
            '<!-- Top 5 most viewed series will be populated dynamically -->', 
            series_cards
        )

        populated_html = populated_html.replace(
            '<!-- Top 5 recommended series will be populated dynamically -->', 
            recommended_cards
        )

        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M')
        populated_html = populated_html.replace(
            'Total of Participants: #',
            f'Total of Participants: {num_participants} | Last Update: {current_datetime}'
        )

        # Save the populated HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(populated_html)

        print(f"Populated HTML saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")