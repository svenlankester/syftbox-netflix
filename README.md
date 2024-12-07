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
### 3. Copy this repository to SyftBox
Move this repository into the apis folder of your SyftBox:
1. Go to your SyftBox `apis` directory:
   ```bash
   cd ~/SyftBox/apis
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/gubertoli/syftbox-netflix.git netflix_trend
   ```
### 4. Set Up the Environment
Configure this app inside SyftBox.

1. Navigate to the `netflix_trend` directory:
   ```bash
   cd netflix_trend
   ```
2. Create an environment configuration file:
   ```bash
   cp .env.example .env
   ```

3. Open the `.env` file in a text editor and fill in your Netflix account information (_if you want to automate viewing history retrieval_) and the required folders details.

### Data format (Netflix)
The data provided by Netflix (Viewing History) is a comma-separated file (CSV), organized by Title and Date:

   ```
   Title,Date
   "The Blacklist: Season 1: Wujing (No. 84)","21/11/2024"
   "Buy Now: The Shopping Conspiracy","20/11/2024"
   "Jake Paul vs. Mike Tyson","16/11/2024"
   "Murder Mindfully: Breathing","15/11/2024"
   ...
   ```

---

## 📁 Generated Files

1. **Aggregated / PET Files:**

   - 📂 Path: `/SyftBox/datasites/<your-email>/api_data/netflix_trend/`
   - This folder contains the aggregated and/or processed (privacy enhanced) Netflix viewing history, **accessible to aggregator**. For instance, parameters of machine learning models for federated learning or differential private data.

2. **Private Processed Files:**

   - 📂 File: `/SyftBox/datasites/<your-email>/private/netflix_data/netflix_full.npy`
   - Contains the full version of the Netflix viewing history, stored privately and **not accessible to others**. This could be used as a starting point for PETs processing.

   - 📂 File: `/SyftBox/datasites/<your-email>/private/netflix_data/tvseries_views_sparse_vector.npy`
   - It is a sparse one-hot encoded vectors of TV series and number of episodes seen, stored privately and **not accessible to other**. The vocabulary to check which TV series represent certain index is available from the aggregator only to the participants of this app.

---

Feel free to reach out with questions or suggestions to improve this project.
