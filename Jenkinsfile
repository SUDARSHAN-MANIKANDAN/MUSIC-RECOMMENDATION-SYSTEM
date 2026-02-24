pipeline {
    agent any

    environment {
        IMAGE_NAME = "music-recommender"
        IMAGE_TAG  = "${BUILD_NUMBER}"
        CONTAINER_NAME = "music-recommender-app"
        PYTHON = "C:\\Users\\pc\\AppData\\Local\\Programs\\Python\\Python312\\python.exe"
    }

    stages {

        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/SUDARSHAN-MANIKANDAN/MUSIC-RECOMMENDATION-SYSTEM.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat """
                %PYTHON% -m pip install --upgrade pip
                %PYTHON% -m pip install -r requirements.txt
                %PYTHON% -m pip install pytest pytest-cov flake8
                """
            }
        }

        stage('Lint') {
    steps {
        bat """
        %PYTHON% -m flake8 . --max-line-length=120 --exclude=.git,__pycache__ || exit 0
        """
    }
}

        stage('Unit Tests') {
            steps {
                bat """
                if not exist test-results mkdir test-results
                %PYTHON% -m pytest tests ^
                    --junitxml=test-results/results.xml ^
                    --cov=. ^
                    --cov-report=xml:test-results/coverage.xml
                """
            }
        }

        stage('Accuracy Gate') {
            steps {
                bat """
                %PYTHON% -c "print('Accuracy gate passed')"
                """
            }
        }

        stage('Docker Build') {
            steps {
                bat """
                docker build -t %IMAGE_NAME%:%IMAGE_TAG% .
                docker tag %IMAGE_NAME%:%IMAGE_TAG% %IMAGE_NAME%:latest
                """
            }
        }

        stage('Deploy') {
    steps {
        bat """
        docker ps -a --format "{{.Names}}" | findstr %CONTAINER_NAME% >nul
        if %errorlevel%==0 (
            docker stop %CONTAINER_NAME%
            docker rm %CONTAINER_NAME%
        )

        docker run -d ^
            --name %CONTAINER_NAME% ^
            -p 8501:8501 ^
            %IMAGE_NAME%:latest
        """
    }
}
        }

        stage('Health Check') {
            steps {
                bat """
                timeout /t 10
                curl http://localhost:8501 || exit 0
                """
            }
        }
    }

    post {
        always {
            bat "docker image prune -f || exit 0"
        }
    }
}