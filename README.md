# EduPredict: Student Performance Prediction

EduPredict is a web application designed to predict student academic performance using machine learning. As a solution architect, I designed and built this project to provide a tool for both students and researchers to gain insights into the factors that influence academic success. The application uses a Django backend and a scikit-learn machine learning pipeline to make predictions.

## Table of Contents

- [The Challenge of Predicting Student Success](#the-challenge-of-predicting-student-success)
- [Key Features](#key-features)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)
- [Tech Stack](#tech-stack)

## The Challenge of Predicting Student Success

Identifying students at risk of academic failure is a critical challenge in education. Early intervention can significantly improve student outcomes, but it requires a data-driven approach to identify those who need support. As a solution architect, I designed EduPredict to address this challenge by providing a platform that leverages machine learning to predict student performance based on a variety of factors.

## Key Features

- **Student-Facing Interface**: An intuitive form for students to input their academic and demographic information.
- **Researcher-Focused Tools**: A dashboard for researchers to upload student data in bulk and receive detailed predictions.
- **Multiple Predictive Models**: Users can choose from several machine learning models (Logistic Regression, Decision Tree, and Random Forest) to see different perspectives on the data.
- **Downloadable Results**: Researchers can download prediction results in CSV format for further analysis.
- **Sample Data**: A sample dataset is provided to help researchers format their data correctly.

## Installation Instructions

### Prerequisites

- [Python](https://www.python.org/) (v3.8 or higher)
- [pip](https://pip.pypa.io/en/stable/installation/)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/edupredict.git
   cd edupredict
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Django development server:**
   ```bash
   python edupredict/manage.py runserver
   ```

## Usage

Once the application is running, open your browser and navigate to `http://127.0.0.1:8000/`.

### Student Flow

1. From the landing page, select the "Student" path.
2. Fill out the form with your information.
3. Choose a prediction model.
4. View your predicted outcome and probability score.

### Researcher Flow

1. From the landing page, select the "Researcher" path.
2. Download the sample CSV to see the required data format.
3. Upload your own CSV file with student data.
4. Choose a prediction model.
5. View the prediction results and download them for further analysis.

## Contributing

We welcome contributions from the community! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.

Please ensure your code adheres to our coding standards and passes all tests.

## License

This project is licensed under the MIT License. This license allows you to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software.

## Contact Information

For questions about the project's design or implementation, feel free to reach out to me. As the solution architect, I'm always open to discussing the architecture and potential improvements.

- **Email**: olu-akinsurojumaxwell@gmail.com
  

## Tech Stack

- **Backend**: Django
- **Machine Learning**: scikit-learn, pandas, numpy, joblib
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite
- **Deployment**: (Not yet implemented)
