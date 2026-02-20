# Building an AI-Based Hiring Prediction System 🤖

I built this project to understand how real-world HR analytics tools actually work. When a company receives thousands of resumes, they can't read them all manually. Instead, they use Machine Learning to filter candidates. 

This project simulates that exact process. It's an end-to-end Machine Learning pipeline that takes raw resume data (like skills, experience, and education) and predicts whether a candidate should be "Hired" or "Rejected."

## What the Dataset Looks Like
I used a synthetic dataset of over 1,000 resumes. It includes a great mix of data types that required different preprocessing strategies:
- __Text Data:__ The candidate's technical skills, certifications, and the job role they applied for.
- __Numerical Data:__ Their total years of experience, expected salary, and the number of projects they've completed.
- __Categorical Data:__ Their highest level of education.
- __The Target:__ The actual recruiter decision (Hire/Reject).

## The Tech Stack
- __Python__ (The backbone of the project)
- __Pandas & NumPy__ (For data manipulation and cleaning)
- __Scikit-Learn__ (For building the machine learning models and pipelines)
- __Matplotlib & Seaborn__ (For initial data inspection, though the focus here is heavily on the modeling)

## How I Processed the Data
Real-world data is messy, so I spent a good chunk of time cleaning and engineering features before feeding them to any algorithms:
1. __Handling Missing Info:__ Filled in missing certifications and dropped columns that would cheat the system (like IDs or pre-calculated AI scores).
2. __Text Engineering (NLP):__ I combined the skills, certifications, and job roles into one big text block, cleaned out the special characters, and used `TfidfVectorizer` to turn that text into mathematical weights.
3. __Scaling Numbers:__ Applied `StandardScaler` to things like salary and experience so massive salary numbers wouldn't overshadow smaller numbers (like years of experience) in the distance calculations.
4. __Encoding Categories:__ Used `LabelEncoder` to turn education levels (like B.Sc or PhD) into numbers the model could understand.

## Training the Models
I didn't just want to build one model; I wanted to see which one performed best on this specific mix of text and numerical data. I split the data (80% training, 20% testing) and trained four different classifiers:
- __Logistic Regression__ 
- __Support Vector Machine (SVM)__
- __Random Forest__
- __K-Nearest Neighbors (KNN)__

## The Results
The linear models absolutely crushed it. 
Both __Logistic Regression__ and __SVM__ hit an accuracy of __98.0%__. Random Forest followed closely at 96.5%, and KNN came in at 94.5%. 

Because the dataset relied heavily on the text-based skills (via TF-IDF, which creates a lot of sparse features), it makes sense that Logistic Regression and SVM performed the best—they are notoriously good at handling high-dimensional text data.

As an optional advanced step, I also built a Scikit-Learn `Pipeline` with `GridSearchCV` to tune the hyperparameters (finding the optimal C-value for Logistic Regression), ensuring the model is robust and ready for production without data leakage.

## The Final Product: An AI Recruiter Function
To wrap it all up, I built a custom Python function to simulate the actual product. You can input a brand new candidate's details (their skills, experience, education, etc.), and the function will process their text, scale their numbers, and instantly output a __"Hire"__ or __"Reject"__ decision, along with the probability score.

## Want to try it yourself?
1. Clone this repo to your local machine.
2. Make sure you have the required libraries installed (`pip install pandas numpy scikit-learn matplotlib seaborn`).
3. Open the `AI_Based_Hiring_Prediction_System.ipynb` notebook.
4. Run all the cells. At the very bottom, you can plug in your own resume details into the prediction function to see if the AI would hire you!
