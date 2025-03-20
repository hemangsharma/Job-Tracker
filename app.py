!pip install --upgrade pip

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
from PIL import Image
import io
import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords

# Ensure you have the NLTK stopwords downloaded
nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="Job Application Tracker",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define file path for the database
DB_PATH = "job_applications.csv"

# Define the application statuses
APPLICATION_STATUSES = [
    "Applied",
    "Resume Screened",
    "Phone Interview",
    "Technical Interview",
    "Onsite/Final Interview",
    "Offer Received",
    "Offer Accepted",
    "Offer Declined",
    "Rejected",
    "Withdrawn"
]

# Define common job platforms
JOB_PLATFORMS = [
    "LinkedIn",
    "Indeed",
    "Seek",
    "Monster",
    "Naukri",
    "Worfforce AU",
    "Glassdoor",
    "Company Website",
    "Referral",
    "Job Fair",
    "Recruiter",
    "Other"
]

# Function to create database if it doesn't exist
def create_database():
    if not os.path.exists(DB_PATH):
        df = pd.DataFrame(columns=[
            "date_applied",
            "company_name",
            "position_title",
            "salary_min",
            "salary_max",
            "platform",
            "job_description",
            "status",
            "location",
            "job_type",
            "last_updated",
            "notes",
            "job_posting_link"  # New column for job posting link
        ])
        df.to_csv(DB_PATH, index=False)
        return df
    return pd.read_csv(DB_PATH)

# Function to add a new job application
def add_job_application(
    date_applied,
    company_name,
    position_title,
    salary_min,
    salary_max,
    platform,
    job_description,
    status,
    location,
    job_type,
    notes,
    job_posting_link  # New parameter for job posting link
):
    df = pd.read_csv(DB_PATH)
    
    new_row = {
        "date_applied": date_applied,
        "company_name": company_name,
        "position_title": position_title,
        "salary_min": salary_min,
        "salary_max": salary_max,
        "platform": platform,
        "job_description": job_description,
        "status": status,
        "location": location,
        "job_type": job_type,
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "notes": notes,
        "job_posting_link": job_posting_link  # Save the job posting link
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(DB_PATH, index=False)
    return df

# Function to update a job application
def update_job_application(index, updated_data):
    df = pd.read_csv(DB_PATH)
    
    for key, value in updated_data.items():
        df.at[index, key] = value
    
    df.at[index, "last_updated"] = datetime.now().strftime("%Y-%m-%d")
    df.to_csv(DB_PATH, index=False)
    return df

# Function to delete a job application
def delete_job_application(index):
    df = pd.read_csv(DB_PATH)
    df = df.drop(index=index).reset_index(drop=True)
    df.to_csv(DB_PATH, index=False)
    return df

# Function to generate word cloud from job descriptions
def generate_word_cloud(text):
    wordcloud = WordCloud(
        background_color="white",
        max_words=100,
        contour_width=3,
        contour_color="steelblue",
        width=800,
        height=400
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt

# Function to extract common skills from job descriptions
def extract_common_skills(text):
    common_skills = [
        "python", "java", "javascript", "html", "css", "sql", "nosql", "aws", 
        "azure", "react", "angular", "vue", "node", "express", "django", "flask",
        "docker", "kubernetes", "ci/cd", "git", "agile", "scrum", "data analysis",
        "machine learning", "ai", "tableau", "power bi", "excel", "r", "hadoop",
        "spark", "tensorflow", "pytorch", "nlp", "data science", "cloud computing",
        "devops", "sre", "product management", "ux", "ui", "figma", "sketch",
        "adobe", "photoshop", "illustrator", "project management", "jira",
        "full stack", "frontend", "backend", "mobile", "ios", "android", "api",
        "microservices", "architecture", "leadership", "communication"
    ]
    
    found_skills = {}
    text_lower = text.lower()
    
    for skill in common_skills:
        count = len(re.findall(r'\b' + re.escape(skill) + r'\b', text_lower))
        if count > 0:
            found_skills[skill] = count
    
    return found_skills

# Main application
def main():
    st.sidebar.title("Job Application Tracker")
    
    df = create_database()
    
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Add Job Application", "View/Update Applications", "Data Analysis"]
    )
    
    if page == "Dashboard":
        display_dashboard(df)
    
    elif page == "Add Job Application":
        add_job_page(df)
    
    elif page == "View/Update Applications":
        view_update_page(df)
    
    elif page == "Data Analysis":
        data_analysis_page(df)

# Dashboard page
def display_dashboard(df):
    st.title("📊 Job Application Dashboard")
    
    if df.empty:
        st.info("No job applications to display. Start by adding a job application.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Applications", value=len(df))
    
    with col2:
        active_apps = len(df[~df['status'].isin(['Rejected', 'Withdrawn', 'Offer Declined', 'Offer Accepted'])])
        st.metric(label="Active Applications", value=active_apps)
    
    with col3:
        offers = len(df[df['status'].isin(['Offer Received', 'Offer Accepted', 'Offer Declined'])])
        st.metric(label="Offers Received", value=offers)
    
    with col4:
        success_rate = round((offers / len(df)) * 100, 1) if len(df) > 0 else 0
        st.metric(label="Success Rate", value=f"{success_rate}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Application Status")
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig = px.bar(
            status_counts,
            x='Status',
            y='Count',
            color='Status',
            title='Applications by Status'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Application Platforms")
        platform_counts = df['platform'].value_counts().reset_index()
        platform_counts.columns = ['Platform', 'Count']
        
        fig = px.pie(
            platform_counts,
            values='Count',
            names='Platform',
            title='Applications by Platform'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Application Timeline")
    df['date_applied'] = pd.to_datetime(df['date_applied'])
    timeline_data = df.groupby(df['date_applied'].dt.strftime('%Y-%m-%d')).size().reset_index(name='count')
    timeline_data.columns = ['Date', 'Applications']
    timeline_data['Date'] = pd.to_datetime(timeline_data['Date'])
    timeline_data = timeline_data.sort_values('Date')
    
    fig = px.line(
        timeline_data,
        x='Date',
        y='Applications',
        title='Applications Over Time',
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Applications")
    recent_df = df.sort_values('date_applied', ascending=False).head(5)
    
    display_df = recent_df[['date_applied', 'company_name', 'position_title', 'status', 'job_posting_link']].copy()
    display_df.columns = ['Date Applied', 'Company', 'Position', 'Status', 'Job Posting Link']
    
    st.table(display_df)

# Add job application page
def add_job_page(df):
    st.title("🖊️ Add New Job Application")
    
    with st.form("job_application_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            date_applied = st.date_input("Date Applied", datetime.now())
            company_name = st.text_input("Company Name")
            position_title = st.text_input("Position Title")
            location = st.text_input("Location")
            job_type = st.selectbox("Job Type", ["Full-time", "Part-time", "Contract", "Internship", "Freelance"])
            job_posting_link = st.text_input("Job Posting Link")  # New input for job posting link
            
        with col2:
            selected_platform = st.selectbox("Application Platform", JOB_PLATFORMS)
            
            if selected_platform == "Other":
                custom_platform = st.text_input("Specify Platform")
                platform = custom_platform if custom_platform else "Other"
            else:
                platform = selected_platform
                
            status = st.selectbox("Application Status", APPLICATION_STATUSES)
            salary_min = st.number_input("Minimum Salary (0 if unknown)", min_value=0, value=0)
            salary_max = st.number_input("Maximum Salary (0 if unknown)", min_value=0, value=0)
        
        job_description = st.text_area("Job Description")
        notes = st.text_area("Notes")
        
        submitted = st.form_submit_button("Add Job Application")
        
        if submitted:
            if not company_name or not position_title:
                st.error("Company name and position title are required.")
            elif selected_platform == "Other" and not platform:
                st.error("Please specify a platform name.")
            else:
                df = add_job_application(
                    date_applied.strftime("%Y-%m-%d"),
                    company_name,
                    position_title,
                    salary_min,
                    salary_max,
                    platform,
                    job_description,
                    status,
                    location,
                    job_type,
                    notes,
                    job_posting_link  # Pass the job posting link
                )
                st.success(f"Added application for {position_title} at {company_name}")

# View and update applications page
def view_update_page(df):
    st.title("👁️ View and Update Applications")
    
    if df.empty:
        st.info("No job applications to display. Start by adding a job application.")
        return
    
    st.subheader("Search and Filter")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_term = st.text_input("Search by company or position", "")
    
    with col2:
        status_filter = st.multiselect("Filter by status", ["All"] + APPLICATION_STATUSES, default=["All"])
    
    with col3:
        unique_platforms = ["All"] + sorted(df['platform'].unique().tolist())
        platform_filter = st.multiselect("Filter by platform", unique_platforms, default=["All"])
    
    filtered_df = df.copy()
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df['company_name'].str.contains(search_term, case=False) |
            filtered_df['position_title'].str.contains(search_term, case=False)
        ]
    
    if "All" not in status_filter:
        filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
    
    if "All" not in platform_filter:
        filtered_df = filtered_df[filtered_df['platform'].isin(platform_filter)]
    
    st.subheader("Applications")
    
    if filtered_df.empty:
        st.info("No job applications match your filters.")
        return
    
    items_per_page = st.slider("Items per page", min_value=5, max_value=50, value=10, step=5)
    total_pages = (len(filtered_df) + items_per_page - 1) // items_per_page
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        current_page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1)
    
    start_idx = (current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_df))
    
    displayed_df = filtered_df.iloc[start_idx:end_idx].copy()
    displayed_df = displayed_df.reset_index()
    displayed_df['index'] = displayed_df['index'].astype(int)
    
    view_df = displayed_df[['index', 'date_applied', 'company_name', 'position_title', 'status', 'job_posting_link']].copy()
    view_df.columns = ['Index', 'Date Applied', 'Company', 'Position', 'Status', 'Job Posting Link']
    
    st.dataframe(view_df, use_container_width=True)
    
    st.subheader("View/Update Application Details")
    selected_idx = st.number_input("Select application by index", min_value=0, max_value=len(df)-1 if len(df) > 0 else 0, value=int(view_df['Index'].iloc[0]) if not view_df.empty else 0)
    
    if selected_idx >= 0 and selected_idx < len(df):
        selected_app = df.iloc[selected_idx]
        
        with st.form("update_application_form"):
            st.subheader(f"{selected_app['position_title']} at {selected_app['company_name']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                updated_date = st.date_input("Date Applied", pd.to_datetime(selected_app['date_applied']))
                updated_company = st.text_input("Company Name", selected_app['company_name'])
                updated_position = st.text_input("Position Title", selected_app['position_title'])
                updated_location = st.text_input("Location", selected_app['location'])
                updated_job_type = st.selectbox("Job Type", ["Full-time", "Part-time", "Contract", "Internship", "Freelance"], index=["Full-time", "Part-time", "Contract", "Internship", "Freelance"].index(selected_app['job_type']))
                
            with col2:
                all_platforms = JOB_PLATFORMS.copy()
                current_platform = selected_app['platform']
                
                if current_platform not in all_platforms and current_platform != "Other":
                    all_platforms.append(current_platform)
                
                selected_updated_platform = st.selectbox(
                    "Application Platform", 
                    all_platforms,
                    index=all_platforms.index(current_platform) if current_platform in all_platforms else all_platforms.index("Other")
                )
                
                if selected_updated_platform == "Other":
                    custom_updated_platform = st.text_input(
                        "Specify Platform", 
                        value="" if current_platform == "Other" else current_platform if current_platform not in JOB_PLATFORMS else ""
                    )
                    updated_platform = custom_updated_platform if custom_updated_platform else "Other"
                else:
                    updated_platform = selected_updated_platform
                
                updated_status = st.selectbox("Application Status", APPLICATION_STATUSES, index=APPLICATION_STATUSES.index(selected_app['status']) if selected_app['status'] in APPLICATION_STATUSES else 0)
                updated_salary_min = st.number_input("Minimum Salary", min_value=0, value=int(selected_app['salary_min']))
                updated_salary_max = st.number_input("Maximum Salary", min_value=0, value=int(selected_app['salary_max']))
            
            updated_job_description = st.text_area("Job Description", selected_app['job_description'])
            updated_notes = st.text_area("Notes", selected_app['notes'])
            updated_job_posting_link = st.text_input("Job Posting Link", selected_app['job_posting_link'])  # New input for job posting link
            
            col1, col2 = st.columns(2)
            
            with col1:
                update_submitted = st.form_submit_button("Update Application")
            
            with col2:
                delete_submitted = st.form_submit_button("Delete Application")
            
            if update_submitted:
                if selected_updated_platform == "Other" and not updated_platform:
                    st.error("Please specify a platform name.")
                else:
                    updated_data = {
                        "date_applied": updated_date.strftime("%Y-%m-%d"),
                        "company_name": updated_company,
                        "position_title": updated_position,
                        "salary_min": updated_salary_min,
                        "salary_max": updated_salary_max,
                        "platform": updated_platform,
                        "job_description": updated_job_description,
                        "status": updated_status,
                        "location": updated_location,
                        "job_type": updated_job_type,
                        "notes": updated_notes,
                        "job_posting_link": updated_job_posting_link  # Save the updated job posting link
                    }
                    
                    df = update_job_application(selected_idx, updated_data)
                    st.success(f"Updated application for {updated_position} at {updated_company}")
            
            if delete_submitted:
                if st.warning(f"Are you sure you want to delete the application for {selected_app['position_title']} at {selected_app['company_name']}?"):
                    df = delete_job_application(selected_idx)
                    st.success(f"Deleted application for {selected_app['position_title']} at {selected_app['company_name']}")

# Define a function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_bytes = pdf_file.read()
    doc = fitz.open("pdf", pdf_bytes)
    for page in doc:
        text += page.get_text()
    return text

# Define a function to analyze keywords in the CV and cover letter
def analyze_keywords(text, keywords):
    text_lower = text.lower()
    keyword_count = {keyword: text_lower.count(keyword.lower()) for keyword in keywords}
    score = sum(keyword_count.values())
    return score, keyword_count

# Data analysis page
def data_analysis_page(df):
    st.title("📈 Job Application Analysis")
    
    if df.empty:
        st.info("No job applications to analyze. Start by adding a job application.")
        return
    
    st.subheader("CV Analysis")
    
    cv_file = st.file_uploader("Upload Your CV (PDF format only)", type=['pdf'])
    if cv_file is None:
        st.warning("Please upload a PDF file.")
    
    keywords = [
        "python", "java", "javascript", "html", "css", "sql", "nosql", "aws", 
        "azure", "react", "angular", "vue", "node", "express", "django", "flask",
        "docker", "kubernetes", "ci/cd", "git", "agile", "scrum", "data analysis",
        "machine learning", "ai", "tableau", "power bi", "excel", "r", "hadoop",
        "spark", "tensorflow", "pytorch", "nlp", "data science", "cloud computing",
        "devops", "sre", "product management", "ux", "ui", "figma", "sketch",
        "adobe", "photoshop", "illustrator", "project management", "jira",
        "full stack", "frontend", "backend", "mobile", "ios", "android", "api",
        "microservices", "architecture", "leadership", "communication"
    ]
    
    if cv_file:
        cv_text = extract_text_from_pdf(cv_file)
        
        cv_score, cv_keyword_count = analyze_keywords(cv_text, keywords)
        st.subheader("CV Analysis")
        st.write(f"Score: {cv_score}")
        
        st.bar_chart(cv_keyword_count)

        threshold = st.slider("Set Keyword Relevance Threshold", 0, 100, 50)
        filtered_keywords = {k: v for k, v in cv_keyword_count.items() if v >= threshold}
        st.write("Filtered Keyword Counts:")
        st.write(filtered_keywords)
    
    st.subheader("Success Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_apps = len(df)
        st.metric("Total Applications", total_apps)
    
    with col2:
        response_rate = len(df[df['status'] != 'Applied']) / total_apps if total_apps > 0 else 0
        st.metric("Response Rate", f"{response_rate:.1%}")
    
    with col3:
        interview_rate = len(df[df['status'].isin(['Phone Interview', 'Technical Interview', 'Onsite/Final Interview', 'Offer Received', 'Offer Accepted', 'Offer Declined'])]) / total_apps if total_apps > 0 else 0
        st.metric("Interview Rate", f"{interview_rate:.1%}")
    
    st.subheader("Time-based Analysis")
    
    df['date_applied'] = pd.to_datetime(df['date_applied'])
    df['application_week'] = df['date_applied'].dt.strftime('%Y-%U')
    
    weekly_apps = df.groupby('application_week').size().reset_index(name='count')
    weekly_apps['application_week'] = pd.to_datetime(weekly_apps['application_week'].apply(lambda x: f"{x.split('-')[0]}-{int(x.split('-')[1])*7}"), format='%Y-%j')
    
    fig = px.line(
        weekly_apps,
        x='application_week',
        y='count',
        title='Applications per Week',
        labels={'application_week': 'Week', 'count': 'Applications'},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Salary Analysis")
    
    salary_df = df[(df['salary_min'] > 0) | (df['salary_max'] > 0)].copy()
    
    if not salary_df.empty:
        salary_df['avg_salary'] = salary_df.apply(
            lambda x: (x['salary_min'] + x['salary_max']) / 2 if x['salary_min'] > 0 and x['salary_max'] > 0 
            else x['salary_max'] if x['salary_min'] == 0 
            else x['salary_min'],
            axis=1
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                salary_df,
                x='avg_salary',
                nbins=20,
                title='Salary Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            salary_by_type = salary_df.groupby('job_type')['avg_salary'].mean().reset_index()
            
            fig = px.bar(
                salary_by_type,
                x='job_type',
                y='avg_salary',
                title='Average Salary by Job Type',
                color='job_type'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No salary data available for analysis.")
    
    st.subheader("Job Description Analysis")
    
    all_job_descriptions = " ".join(df['job_description'].fillna(""))
    
    if all_job_descriptions.strip():
        wordcloud_fig = generate_word_cloud(all_job_descriptions)
        st.pyplot(wordcloud_fig)
        
        skills = extract_common_skills(all_job_descriptions)
        
        if skills:
            skills_df = pd.DataFrame(list(skills.items()), columns=['Skill', 'Count'])
            skills_df = skills_df.sort_values('Count', ascending=False)
            
            fig = px.bar(
                skills_df.head(15),
                x='Skill',
                y='Count',
                title='Most Common Skills in Job Descriptions',
                color='Count',
                color_continuous_scale=px.colors.sequential.Blues
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No common skills found in job descriptions.")
    else:
        st.info("No job descriptions available for analysis.")
    
    st.subheader("Application Status Analysis")
    
    status_order = {status: i for i, status in enumerate(APPLICATION_STATUSES)}
    df['status_order'] = df['status'].map(status_order)
    
    progress_df = df.groupby('status')['status_order'].count().reset_index()
    progress_df['percentage'] = progress_df['status_order'] / len(df) * 100
    
    fig = go.Figure(go.Funnel(
        y=APPLICATION_STATUSES,
        x=progress_df['percentage'],
        textinfo="value+percent initial"
    ))
    
    fig.update_layout(title_text="Application Funnel")
    st.plotly_chart(fig, use_container_width=True)
    
    platform_success = df.groupby('platform').apply(
        lambda x: len(x[x['status'].isin(['Offer Received', 'Offer Accepted'])])/len(x) if len(x) > 0 else 0
    ).reset_index(name='success_rate')
    
    if len(platform_success) > 0:
        platform_success = platform_success[platform_success['platform'].isin(df['platform'].value_counts().nlargest(5).index)]
        
        if not platform_success.empty:
            fig = px.bar(
                platform_success,
                x='platform',
                y='success_rate',
                title='Success Rate by Platform',
                color='platform',
                labels={'success_rate': 'Success Rate', 'platform': 'Platform'},
                text_auto='.1%'
            )
            
            fig.update_traces(texttemplate='%{text}')
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()