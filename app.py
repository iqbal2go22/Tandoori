from flask import Flask, render_template
from markupsafe import Markup  # Import Markup from markupsafe instead
import pyodbc
import pandas as pd
import os
from flask.testing import FlaskClient
import re
import os
import shutil
import re
from flask.testing import FlaskClient

app = Flask(__name__)

# SQL Server connection information
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost\\TONYSQL;"
    "DATABASE=TonyDB;"
    "Trusted_Connection=yes;"
)

@app.route('/weekly_scores')
def weekly_scores():
    try:
        conn = pyodbc.connect(conn_str)
        query = "SELECT * FROM TandooriWeeklyScores"
        df = pd.read_sql(query, conn)
        conn.close()

        df = df.round(2)

        # Calculate top 3 scores for each week excluding Total and AdjustedTotal columns
        top_weekly_scores = {}
        for week in [col for col in df.columns if col.startswith('Week')]:
            top_weekly_scores[week] = df.loc[df['TeamName'] != 'Total', week].nlargest(3).tolist()

        # Calculate top 3 scores for the season excluding AdjustedTotal
        season_columns = [col for col in df.columns if col.startswith('Week')]
        top_season_scores = df.loc[df['TeamName'] != 'Total', season_columns].values.flatten()
        top_season_scores = pd.Series(top_season_scores).nlargest(3).tolist()

        columns = df.columns.tolist()
        data = df.to_dict(orient='records')
        return render_template(
            'weekly_scores.html', 
            columns=columns, 
            data=data, 
            highlight_totals=True, 
            top_weekly_scores=top_weekly_scores, 
            top_season_scores=top_season_scores
        )
    except Exception as e:
        print("An error occurred:", str(e))
        return "An error occurred while fetching weekly scores.", 500

@app.route('/weekly_payouts')
def weekly_payouts():
    try:
        conn = pyodbc.connect(conn_str)
        query_scores = "SELECT * FROM TandooriWeeklyScores"
        query_matchups = "SELECT * FROM WeeklyMatchupsFinal"

        df_scores = pd.read_sql(query_scores, conn)
        df_matchups = pd.read_sql(query_matchups, conn)
        conn.close()

        df_scores = df_scores.round(2)
        df_matchups = df_matchups.round(2)

        # Calculate point differential
        df_matchups['PointDifferential'] = (df_matchups['Team1Score'] - df_matchups['Team2Score']).abs()

        # Generate top 3 teams for each week
        weekly_payouts_data = {}
        for week in [col for col in df_scores.columns if col.startswith('Week')]:
            top_3 = df_scores.loc[df_scores['TeamName'] != 'Total', ['TeamName', week]].nlargest(3, week)
            top_3['PayoutAmount'] = [50, 25, 15]
            weekly_payouts_data[week] = top_3.to_dict(orient='records')

        # Generate weekly matchups data
        weekly_matchups_data = {}
        for week in df_matchups['Week'].unique():
            matchups = df_matchups[df_matchups['Week'] == week]
            weekly_matchups_data[week] = matchups[['Team1', 'Team2', 'Team1Score', 'Team2Score', 'PointDifferential', 'PayoutAmount']].rename(columns={'Team1Score': 'Team1Points', 'Team2Score': 'Team2Points'}).to_dict(orient='records')

        return render_template('weekly_payouts.html', weekly_payouts_data=weekly_payouts_data, weekly_matchups_data=weekly_matchups_data)
    except Exception as e:
        print("An error occurred:", str(e))
        return "An error occurred while fetching weekly payouts.", 500

@app.route('/summary')
def summary():
    try:
        conn = pyodbc.connect(conn_str)

        # Query for Adjusted Standings
        query_adjusted = """
        SELECT  
            FORMAT(RANK() OVER (ORDER BY AdjustedTotal DESC), '00') AS PayoutRank,
            TeamName,
            ROUND(AdjustedTotal, 2) AS AdjustedTotal,
            ROUND((SELECT MAX(AdjustedTotal) FROM TandooriWeeklyScores WHERE TeamName <> 'Total') - AdjustedTotal, 2) AS PointsBehind,
            CASE RANK() OVER (ORDER BY AdjustedTotal DESC)
                WHEN 1 THEN 225
                WHEN 2 THEN 150
                WHEN 3 THEN 100
                ELSE 0
            END AS PayoutAmount
        FROM TandooriWeeklyScores
        WHERE TeamName <> 'Total';
        """
        df_adjusted = pd.read_sql(query_adjusted, conn)
        print("Adjusted Standings DataFrame:")
        print(df_adjusted)

        # Query for Head to Head Standings
        query_head_to_head = """
        SELECT TeamName, WinPercentage, TotalPayout
        FROM StandingsFinal
        ORDER BY WinPercentage DESC;
        """
        df_head_to_head = pd.read_sql(query_head_to_head, conn)
        print("Head to Head DataFrame:")
        print(df_head_to_head)

        # Processing weekly payouts as done in the standings route
        df = pd.read_sql("SELECT * FROM TandooriWeeklyScores", conn)
        print("Weekly Scores DataFrame:")
        print(df)

        weekly_payouts = []
        for week in [col for col in df.columns if col.startswith('Week')]:
            # Get top 3 teams for each week, exclude the 'Total' row
            top_3_weekly = df.loc[df['TeamName'] != 'Total', ['TeamName', week]].nlargest(3, week)
            # Assign payouts to the top 3 teams
            payout_amounts = [50.0, 25.0, 15.0]
            top_3_weekly['PayoutAmount'] = [float(amount) for amount in payout_amounts[:len(top_3_weekly)]]
            weekly_payouts.append(top_3_weekly[['TeamName', 'PayoutAmount']])

        if weekly_payouts:
            weekly_payouts_df = pd.concat(weekly_payouts, ignore_index=True)
        else:
            weekly_payouts_df = pd.DataFrame(columns=['TeamName', 'PayoutAmount'])

        weekly_payouts_df['PayoutAmount'] = weekly_payouts_df['PayoutAmount'].astype(float)
        weekly_payouts_df = weekly_payouts_df.groupby('TeamName')['PayoutAmount'].sum().reset_index()
        print("Weekly Payouts DataFrame:")
        print(weekly_payouts_df)

        # Processing top 3 individual scores as done in the standings route
        df = df.drop(columns=['Total', 'AdjustedTotal'], errors='ignore')  # Drop Total and AdjustedTotal columns
        df_melted = df.melt(id_vars=['TeamName'], var_name='Week', value_name='Score')
        top_3_scores = df_melted.loc[df_melted['TeamName'] != 'Total'].nlargest(3, 'Score')
        top_3_scores['PayoutAmount'] = [225, 125, 100]
        print("Top 3 Individual Scores DataFrame:")
        print(top_3_scores)

        # Update Adjusted Payout Amount
        df_adjusted['PayoutAmount'] = df_adjusted['PayoutAmount'].apply(lambda x: 0 if pd.isna(x) else int(x))

        # Head-to-Head Payout from StandingsFinal table (using TotalPayout column)
        df_head_to_head['PayoutAmount'] = df_head_to_head['TotalPayout']

        # Combine all payout data
        total_payout_df = pd.concat([
            df_adjusted[['TeamName', 'PayoutAmount']],  # Adjusted Payout
            top_3_scores[['TeamName', 'PayoutAmount']],  # Yearly Top Individual Scores
            weekly_payouts_df[['TeamName', 'PayoutAmount']],  # Weekly Top 3 Payouts
            df_head_to_head[['TeamName', 'PayoutAmount']]  # Head-to-Head Payout
        ], ignore_index=True)
        print("Combined Payout DataFrame before grouping:")
        print(total_payout_df)

        # Sum all payouts by team
        total_payout_df = total_payout_df.groupby('TeamName')['PayoutAmount'].sum().reset_index()
        total_payout_df = total_payout_df.sort_values(by='PayoutAmount', ascending=False).reset_index(drop=True)
        total_payout_df.rename(columns={'PayoutAmount': 'TotalPayout'}, inplace=True)
        print("Total Payout DataFrame after grouping:")
        print(total_payout_df)

        # Extracting top three teams for the summary page with TotalPayout
        top_three_teams = total_payout_df.nlargest(3, 'TotalPayout')[['TeamName', 'TotalPayout']].to_dict(orient='records')
        print("Top Three Teams:")
        print(top_three_teams)

        # Assign images based on team name
        for team in top_three_teams:
            # Replace spaces and special characters with underscores, convert to lowercase
            image_filename = team['TeamName'].lower().replace(" ", "_") + ".jpg"
            team['image'] = image_filename
            team['TotalPayout'] = f"${team['TotalPayout']:.2f}"  # Format payout amount as currency

        # Rename columns for easier rendering
        columns = df_adjusted.columns.tolist()
        data = df_adjusted.to_dict(orient='records')

        print("Data passed to HTML template:")
        print("Columns:", columns)
        print("Data:", data)
        print("Top Three Teams with Images and Payouts:", top_three_teams)

        # Database connection (assuming you're using the same connection)
        cursor = conn.cursor()

        # Fetch newsletter content from database
        cursor.execute("SELECT content FROM summary")  # Your SQL query to fetch newsletter content
        newsletter_content = cursor.fetchone()[0]  # Fetch the first row

        conn.close()  # Close the connection

        return render_template('summary.html', 
                               columns=columns, 
                               data=data, 
                               top_three_teams=top_three_teams, 
                               newsletter_content=Markup(newsletter_content))  # Pass the newsletter content to the template

    except Exception as e:
        print("An error occurred:", str(e))
        return "An error occurred while fetching data.", 500







@app.route('/standings')
def standings():
    try:
        conn = pyodbc.connect(conn_str)

        # Query for Adjusted Standings
        query_adjusted = """
        SELECT 
            FORMAT(RANK() OVER (ORDER BY AdjustedTotal DESC), '00') AS PayoutRank,
            TeamName,
            ROUND(AdjustedTotal, 2) AS AdjustedTotal,
            ROUND((SELECT MAX(AdjustedTotal) FROM TandooriWeeklyScores WHERE TeamName <> 'Total') - AdjustedTotal, 2) AS PointsBehind,
            CASE RANK() OVER (ORDER BY AdjustedTotal DESC)
                WHEN 1 THEN 225
                WHEN 2 THEN 150
                WHEN 3 THEN 100
                ELSE 0
            END AS PayoutAmount
        FROM TandooriWeeklyScores
        WHERE TeamName <> 'Total';
        """

        # Query for Total Standings
        query_total = """
        SELECT 
            FORMAT(RANK() OVER (ORDER BY Total DESC), '00') AS TotalRank,
            TeamName,
            ROUND(Total, 2) AS Total,
            ROUND((SELECT MAX(Total) FROM TandooriWeeklyScores WHERE TeamName <> 'Total') - Total, 2) AS PointsBehind
        FROM TandooriWeeklyScores 
        WHERE TeamName <> 'Total';
        """

        # Query for Head to Head Standings
        query_head_to_head = """
        SELECT TeamName, WinPercentage, TotalPayout
        FROM StandingsFinal
        ORDER BY WinPercentage DESC;
        """

        print("Executing adjusted query...")
        df_adjusted = pd.read_sql(query_adjusted, conn)
        print("Executing total query...")
        df_total = pd.read_sql(query_total, conn)
        print("Executing head to head query...")
        df_head_to_head = pd.read_sql(query_head_to_head, conn)

        print("Fetching all data...")
        df = pd.read_sql("SELECT * FROM TandooriWeeklyScores", conn)

        print("Processing weekly payouts...")
        weekly_payouts = []

        # Iterate over each week column to calculate weekly payouts
        for week in [col for col in df.columns if col.startswith('Week')]:
            # Get top 3 teams for each week, exclude the 'Total' row
            top_3_weekly = df.loc[df['TeamName'] != 'Total', ['TeamName', week]].nlargest(3, week)
            # Assign payouts to the top 3 teams
            payout_amounts = [50.0, 25.0, 15.0]  # Explicitly using floats
            top_3_weekly['PayoutAmount'] = [float(amount) for amount in payout_amounts[:len(top_3_weekly)]]
            weekly_payouts.append(top_3_weekly[['TeamName', 'PayoutAmount']])

        # Concatenate all weekly payouts into a single DataFrame
        if weekly_payouts:
            weekly_payouts_df = pd.concat(weekly_payouts, ignore_index=True)
        else:
            weekly_payouts_df = pd.DataFrame(columns=['TeamName', 'PayoutAmount'])

        # Ensure PayoutAmount is float
        weekly_payouts_df['PayoutAmount'] = weekly_payouts_df['PayoutAmount'].astype(float)

        # Ensure we sum all weekly payouts by team
        weekly_payouts_df = weekly_payouts_df.groupby('TeamName')['PayoutAmount'].sum().reset_index()

        print("Weekly Payouts Aggregation:")
        print(weekly_payouts_df)
        print(weekly_payouts_df.dtypes)  # Print data types to verify

        conn.close()

        print("Processing data...")
        # Drop Total and AdjustedTotal for further processing
        df = df.drop(columns=['Total', 'AdjustedTotal'], errors='ignore')
        df_melted = df.melt(id_vars=['TeamName'], var_name='Week', value_name='Score')
        top_3_scores = df_melted.loc[df_melted['TeamName'] != 'Total'].nlargest(3, 'Score')
        top_3_scores['PayoutAmount'] = [225, 125, 100]

        # Update Adjusted Payout Amount
        df_adjusted['PayoutAmount'] = df_adjusted['PayoutAmount'].apply(lambda x: 0 if pd.isna(x) else int(x))

        # Head-to-Head Payout from StandingsFinal table (using TotalPayout column)
        df_head_to_head['PayoutAmount'] = df_head_to_head['TotalPayout']

        # Combine all payout data
        total_payout_df = pd.concat([
            df_adjusted[['TeamName', 'PayoutAmount']],  # Adjusted Payout
            top_3_scores[['TeamName', 'PayoutAmount']],  # Yearly Top Individual Scores
            weekly_payouts_df[['TeamName', 'PayoutAmount']],  # Weekly Top 3 Payouts
            df_head_to_head[['TeamName', 'PayoutAmount']]  # Head-to-Head Payout
        ], ignore_index=True)

        # Sum all payouts by team
        total_payout_df = total_payout_df.groupby('TeamName')['PayoutAmount'].sum().reset_index()
        total_payout_df = total_payout_df.sort_values(by='PayoutAmount', ascending=False).reset_index(drop=True)
        total_payout_df.rename(columns={'PayoutAmount': 'TotalPayout'}, inplace=True)

        # Add rank to the total payout
        total_payout_df['Rank'] = total_payout_df.index + 1
        total_payout_df['Rank'] = total_payout_df['Rank'].apply(lambda x: f"{x:02d}")

        print("Preparing data for template...")
        adjusted_data = df_adjusted.to_dict(orient='records')
        total_data = df_total.to_dict(orient='records')
        top_3_scores_data = top_3_scores.to_dict(orient='records')
        total_payout_data = total_payout_df.to_dict(orient='records')
        head_to_head_data = df_head_to_head.to_dict(orient='records')
        weekly_payouts_agg_data = weekly_payouts_df.to_dict(orient='records')

        print("Rendering template...")
        return render_template('standings.html', adjusted_data=adjusted_data, total_data=total_data, top_3_scores_data=top_3_scores_data, total_payout_data=total_payout_data, head_to_head_data=head_to_head_data, weekly_payouts_agg_data=weekly_payouts_agg_data)

    except Exception as e:
        print("An error occurred in standings():", str(e))
        import traceback
        traceback.print_exc()
        return "An error occurred while fetching data. Please check the server logs for more information.", 500


def generate_static_files(app):
    with app.test_client() as client:
        routes = ['/summary', '/weekly_scores', '/weekly_payouts', '/standings']
        for route in routes:
            response = client.get(route)
            filename = f"{route.strip('/')}.html" if route != '/' else 'index.html'
            os.makedirs('static_html', exist_ok=True)
            with open(os.path.join('static_html', filename), 'wb') as f:
                f.write(response.data)
    
    print("Static HTML files generated successfully.")

def update_html_links(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.html'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                content = file.read()
            
            # Update navigation links
            content = content.replace("{{ url_for('summary') }}", "summary.html")
            content = content.replace("{{ url_for('weekly_scores') }}", "weekly_scores.html")
            content = content.replace("{{ url_for('weekly_payouts') }}", "weekly_payouts.html")
            content = content.replace("{{ url_for('standings') }}", "standings.html")
            
            # Update static file links
            content = re.sub(r"\{\{\s*url_for\('static',\s*filename='([^']*)'\)\s*\}\}", r"\1", content)
            
            with open(filepath, 'w') as file:
                file.write(content)
    
    print("HTML links updated successfully.")
def generate_static_files(app):
    with app.test_client() as client:
        routes = ['/summary', '/weekly_scores', '/weekly_payouts', '/standings']
        for route in routes:
            response = client.get(route)
            filename = f"{route.strip('/')}.html" if route != '/' else 'index.html'
            os.makedirs('static_html', exist_ok=True)
            with open(os.path.join('static_html', filename), 'wb') as f:
                f.write(response.data)
    
    # Copy static files (including images)
    static_folder = os.path.join(app.root_path, 'static')
    static_html_folder = os.path.join('static_html', 'static')
    if os.path.exists(static_folder):
        shutil.copytree(static_folder, static_html_folder, dirs_exist_ok=True)
    
    print("Static HTML files and images generated successfully.")


def update_html_links(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.html'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Update navigation links to be relative
            content = re.sub(r'href="/(\w+)"', r'href="\1.html"', content)
            
            # Update static file links (including images)
            content = re.sub(r"\{\{\s*url_for\('static',\s*filename='([^']*)'\)\s*\}\}", r"static/\1", content)
            
            # Update image links that use Python string formatting
            content = re.sub(r"\{\{\s*url_for\('static',\s*filename=([^}]+)\)\s*\}\}", r"static/\1", content)
            
            # Remove leading slash from /static/ paths
            content = content.replace('src="/static/', 'src="static/')
            
            # Remove any remaining Jinja2 expressions
            content = re.sub(r"\{\{.*?\}\}", "", content)
            
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(content)
    
    print("HTML links updated successfully.")

# Modify your if __name__ == '__main__': block to look like this:
if __name__ == '__main__':
    generate_static_files(app)
    update_html_links('static_html')
