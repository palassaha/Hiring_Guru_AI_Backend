
import json
from screening import generate_custom_assessment

# Sample test data
test_data = {
    "userId": "test123",
    "skills": ["Python", "React", "AWS"],
    "contributionFreq": "daily", 
    "projectsCount": 5,
    "topLanguages": ["Python", "JavaScript"],
    "recentActivity": {},
    "repositoryStats": {},
    "targetRole": "Software Engineer",
    "dreamCompanies": ["Google", "Microsoft"],
    "skillGaps": ["System Design"],
    "careerPath": ["Senior Engineer"]
}

# Test the function
result = generate_custom_assessment(test_data)
print(json.dumps(result, indent=2))