import os

def load_text_file(filepath):
    """ Load a text file and return its content as a string. """
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    return ""

# Example Usage
job_description = load_text_file("data/job_description.txt")
resume = load_text_file("data/sample_resume.txt")

print("Job Description:\n", job_description)
print("\nResume:\n", resume)
