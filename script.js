let applications = [];

function showPage(pageId) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => {
        page.style.display = 'none';
    });
    document.getElementById(pageId).style.display = 'block';
}

document.getElementById('jobApplicationForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const newApplication = {
        dateApplied: document.getElementById('date-applied').value,
        companyName: document.getElementById('company-name').value,
        positionTitle: document.getElementById('position-title').value,
        location: document.getElementById('location').value,
        jobType: document.getElementById('job-type').value,
        jobPostingLink: document.getElementById('job-posting-link').value,
        platform: document.getElementById('platform').value,
        status: document.getElementById('status').value,
        salaryMin: document.getElementById('salary-min').value,
        salaryMax: document.getElementById('salary-max').value,
        jobDescription: document.getElementById('job-description').value,
        notes: document.getElementById('notes').value,
    };
    applications.push(newApplication);
    alert('Job application added successfully!');
    document.getElementById('jobApplicationForm').reset();
});

function analyzeCV() {
    const cvFile = document.getElementById('cv-upload').files[0];
    if (cvFile) {
        // Here you would implement the logic to analyze the CV
        // For now, we will just display a message
        document.getElementById('cv-analysis-results').innerText = 'CV analysis is not implemented yet.';
    } else {
        alert('Please upload a CV file.');
    }
}

// Additional functions to update the dashboard metrics and charts would go here