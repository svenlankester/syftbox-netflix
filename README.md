# SyftBox App for Netflix Viewing History 🍿

This project is a proof of concept utilizing [SyftBox](https://syftbox-documentation.openmined.org/) from [OpenMined](https://openmined.org/) to process 🔒 private data. The use case focuses on analyzing the [Netflix viewing history](https://help.netflix.com/en/node/101917) provided by users. This effort is part of the [#30DaysOfFLCode](https://info.openmined.org/30daysofflcode) initiative.

> ✋ If you are interested in joining this work, check out the Matchmaking spreadsheet for #30DaysOfFLCode [here](https://docs.google.com/spreadsheets/d/1euxZMxQXwctjRt_MVLqnqkuBqpXKuGagLReYANXj1i8/edit?gid=78639164#gid=78639164).

[![Join OpenMined on Slack](https://img.shields.io/badge/Join%20Us%20on-Slack-blue)](https://slack.openmined.org/)

## 🎯 Goals

The primary aim is to apply 🛡️ privacy-enhancing technologies to derive aggregate information from Netflix viewing history while safeguarding personal details. Some possible insights include (**_ideas are welcome_**):

- **Most common show viewed in the last week**
- **Viewing trends among participants**
- **Am I watching too much in comparison with others?**
- **Watching more due to sickness/injury?** [(source)](https://www.kaggle.com/code/nachoco/netflix-viewing-analysis-with-injury)

---

## Installation & Requirements
**_Tested on Linux and macOS._**

### 1. Install ChromeDriver
To retrieve your Netflix viewing history automatically:

- **For macOS:**
  ```bash
  brew install chromedriver
   ```

- **For Ubuntu/Linux:**
   ```bash
   sudo apt-get install chromium-driver
   ```

> **Note:** If you prefer, you can skip this step and manually download your Netflix viewing history as a CSV. See [How to download your Netflix viewing history](https://help.netflix.com/en/node/101917). Once downloaded, place the CSV file in the `OUTPUT_DIR` specified in the `.env` file.

### 2. Start SyftBox
Install and start SyftBox by running this command:

   ```bash
   curl -LsSf https://syftbox.openmined.org/install.sh | sh
   ```
### 3. Install app on SyftBox
From terminal, once SyftBox is running:
   ```bash
   syftbox app install gubertoli/syftbox-netflix
   ```

### 4. Set Up the Environment
Configure this app inside SyftBox.

1. Navigate to the `syftbox-netflix` directory:
   ```bash
   cd /SyftBox/apis/syftbox-netflix
   ```
2. Open the `.env` file in a text editor and **define at least** `OUTPUT_DIR`. This is the directory to make available your `NetflixViewingHistory.csv` downloaded manually, if not available, a dummy file will be created. Optionally, fill in your Netflix account information to automate viewing history retrieval.

### Data format (Netflix)
The data provided by Netflix (Viewing History) is a comma-separated file (CSV), organized by Title and Date:

   ```
   Title,Date
   "Show Name: Season X: Episode Name","DD/MM/YYYY"
   ...
   ```
---

Feel free to reach out with questions or suggestions to improve this project.
