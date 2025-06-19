# README in progress...

## 📌 Project Title 
End-to-End MLB Moneyline Betting System

## 📖 Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Data Sources](#data-sources)
4. [ETL Pipeline](#etl-pipeline)
5. [Database Schema](#database-schema)
6. [Data Validation & Quality Checks](#data-validation--quality-checks)
7. [Modeling Pipeline](#modeling-pipeline)
8. [How to Run](#how-to-run)
9. [Environment Setup](#environment-setup)
10. [Future Work](#future-work)
11. [Acknowledgments](#acknowledgments)
12. [License](#license)

## 🧠 Overview
- __What does the project do?__
This project aims to create a production-level (or close to production level) system to aid in MLB moneyline betting. As a result, I hope that this project becomes fully self-contained from data storage to ETL/ELT, modeling, backtesting, monitoring, and deploying. The initial version of this project will be completed within a span of 2 months, so while I don't expect the modeling to be optimal, I hope to get something "good enough" while standing up the deployment aspect. Additional model refinement can come later. 

While this is a personal project, which I plan to test out with "skin in the game", I suspect that the insights from this work will be useful for teaching (I teach "Data Science for Sports" at the GW School of Business).

Another aim of this project is to gain additional experience in Docker, ETL/ELT, and deployment to supplement my current role as a data scientist. 

## 🗂️ Project Structure
end-to-end-mlb-betting/
├── data/
│   ├── raw_games.csv
│   ├── raw_team_stats.csv
│   └── raw_player_stats.csv
├── dags/
│   ├── airflow_dag.py
├── db/
│   ├── schema.sql
│   ├── data_validation_checks.sql
├── docker/
├── etl/
│   ├── utils.py
│   ├── extract_games.py
│   ├── extract_team_stats.py
│   ├── extract_player_stats.py
│   ├── extract_odds.py
│   ├── load_to_db.py
│   ├── transform_games_clean.py
│   └── update_all_data.py
├── models/
│   ├── evaluate.py
│   ├── train_model.py
├── serving/
│   ├── api.py
│   ├── Dockerfile
├── tracking/
│   ├── mlflow_config
├── validate/
│   └── run_data_checks.py

## ⚾ Data Sources
- MLBStats API
- See documentation here: ...

## 🔁 ETL Pipeline
- __Break into steps: extract, transform, load__
- __Briefly describe each stage, CLI args, file outputs.__

## 🧱 Database Schema
__Diagrams and/or descriptions of your tables: games, team stats, player stats, etc.__

## ✅ Data Validation & Quality Checks
__Describe the validation logic, examples of checks, and how to run them__

## 📈 Modeling Pipeline
__Overview of modeling workflow: features, target variable, cross-validation approach__

## 🚀 How to Run
__Commands to run:__
    - __historical extract__
    - __load to DB__
    - __validation__
    - __modeling__
__Include CLI examples and notes__

## 🛠️ Environment Setup
__Required packages, Python version, virtualenv/conda, Docker (if used)__

## 🔮 Future Work
__Ideas for extending the pipeline or improving the model (e.g., lineup changes, player absences)__

## 🙏 Acknowledgments
__Credits to data providers, libraries, or collaborators__

## 📄 License
__If public, state license type.__
